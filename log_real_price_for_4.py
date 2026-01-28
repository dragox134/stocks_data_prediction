import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy as dc
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Load your data (same as main script)
file_id = '1MqY9yaql1XQbodFSngsHxGbyLdWRhVXj'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
data = pd.read_csv(url)
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])

# Same preprocessing functions
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)
shifted_df_as_np = shifted_df.to_numpy()

# Same scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]
X = dc(np.flip(X, axis=1))

# FIXED: Define split_index and X_train/X_test shapes
split_index = int(len(X) * 0.95)

# Get X_train and X_test SHAPES for dummies arrays
X_train_shape = (split_index, lookback)  # [samples, lookback]
X_test_shape = (len(X) - split_index, lookback)

# TRAINING PRICES (days 0 to split_index-1)
y_train = y[:split_index]
dummies = np.zeros((X_train_shape[0], lookback+1))
dummies[:, 0] = y_train
dummies = scaler.inverse_transform(dummies)
new_y_train = dc(dummies[:, 0])

# TEST PRICES (days split_index to end)
y_test = y[split_index:]
dummies = np.zeros((X_test_shape[0], lookback+1))
dummies[:, 0] = y_test
dummies = scaler.inverse_transform(dummies)
new_y_test = dc(dummies[:, 0])

# Auto-incrementing folder
log_dir = 'runs_testing'
run_name = 'real_prices'
run_index = 1
while (Path(log_dir) / f"{run_name}_{run_index}").exists():
    run_index += 1
log_path = f'{log_dir}/{run_name}_{run_index}'
writer = SummaryWriter(log_path)
print(f"Real prices logged to: {log_path}")

# Log TRAINING prices (separate graph)
print(f"Logging {len(new_y_train)} training days...")
for i, price in enumerate(new_y_train):
    writer.add_scalar('Train/Close', price, i)

# Log TEST prices (separate graph) 
print(f"Logging {len(new_y_test)} test days...")
test_start = len(new_y_train)  # Continuous x-axis
for i, price in enumerate(new_y_test):
    writer.add_scalar('Test/Close', price, test_start + i)

writer.close()
print("✅ Train & Test real prices logged separately!")
print("TensorBoard will show 2 graphs: Train/Actual_Close & Test/Actual_Close")