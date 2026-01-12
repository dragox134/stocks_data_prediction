import os
import sqlite3
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir logs_pytorch --port 6006

# ========= config =========
epochs = 300
batch_size = 32
lookback = 60
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [web:138]

# vytvaranie foldera pre log udajov
def get_next_run_dir(base_dir, run_name):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith(run_name)]
    if not existing:
        return os.path.join(base_dir, f"{run_name}_{epochs}epochs_1")
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[-1]))
        except ValueError:
            pass
    next_id = max(nums) + 1 if nums else 1
    return os.path.join(base_dir, f"{run_name}_{epochs}epochs_{next_id}")


base_log_dir = "logs_pytorch"
train_log_dir = get_next_run_dir(base_log_dir, "train_run")
writer = SummaryWriter(log_dir=train_log_dir)  # [web:39]

# ========= data loading =========
db_path = "usable_dbs/60_mins/AAPL_database_60min.db"
conn = sqlite3.connect(db_path)
df = pl.read_database("SELECT * FROM stock_data", connection=conn)
conn.close()

df = df.sort("date")
data = df.select("close").to_numpy().astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


def create_sequences(dataset: np.ndarray, lookback: int = 60):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data, lookback)

train_size = int(len(X) * 0.8)
X_train_np, X_test_np = X[:train_size], X[train_size:]
y_train_np, y_test_np = y[:train_size], y[train_size:]

# reshape for LSTM: (samples, seq_len, features=1)
X_train = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(-1)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  # [web:132]

X_test = X_test.to(device)
y_test = y_test.to(device)

# ========= LSTM model =========
class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden)
        last_hidden = out[:, -1, :]    # last time step
        out = self.fc(last_hidden)     # (batch, 1)
        return out


model = PriceLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# ========= training loop with TensorBoard =========
for epoch in range(epochs):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    train_loss_epoch = np.mean(train_losses)

    # ---- evaluation on test set ----
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test)
        val_loss = criterion(val_preds, y_test)

    writer.add_scalar("loss/train", train_loss_epoch, epoch)
    writer.add_scalar("loss/val",   val_loss.item(),   epoch)

    # ---- log full predicted price curve (aligned with real_run) ----
    all_X_np, _ = create_sequences(scaled_data, lookback)
    all_X = torch.tensor(all_X_np, dtype=torch.float32).unsqueeze(-1).to(device)

    with torch.no_grad():
        all_preds = model(all_X).cpu().numpy()

    all_preds_real = scaler.inverse_transform(all_preds.reshape(-1, 1))

    for t in range(lookback, len(data)):
        pred_t = float(all_preds_real[t - lookback, 0])
        writer.add_scalar("price", pred_t, global_step=t)  # same tag as real_run
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, val_loss={val_loss.item():.6f}")

writer.close()
print(f"Done with loss/val: {val_loss.item} and loss/train: {train_loss_epoch}")

# optional: dates for offline plots
x_dates = df.select("date").to_numpy().flatten()
x_dates = pd.to_datetime(x_dates)
x_dates = x_dates[-len(y_test_np):]
