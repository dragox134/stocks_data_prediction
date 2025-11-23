import polars as pl
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from callbacks import CustomModelCheckpoint


epochs = 1000
# === callbacks ===
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# checkpoint_callback = ModelCheckpoint(
#     filepath=os.path.join(checkpoint_dir, "best_model.keras"),  # path to save model
#     monitor="val_loss",      # metric to monitor
#     save_best_only=True,     # save only when val_loss improves
#     save_weights_only=False, # True = save only weights, False = save full model
#     verbose=1
# )

checkpoint_callback = CustomModelCheckpoint(
    filepath=f"checkpoints/best_model.keras",
    monitor="val_loss"
)

# === Load data from SQLite database ===
db_path = "dbs/AAPL_database_60min.db"  # change this to your file name
conn = sqlite3.connect(db_path)

# Read SQL query with Polars
df = pl.read_database("SELECT * FROM stock_data", connection=conn)  # replace 'stock_data' with your table name
conn.close()

# === Preprocess ===
df = df.sort("date")
# df = df.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S"))  # ensure datetime format

# For now, use only the 'close' column
data = df.select("close").to_numpy().astype(float)

# Normalize to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# === Create sequences for LSTM ===
def create_sequences(dataset: np.ndarray, lookback: int = 60):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

lookback = 60
X, y = create_sequences(scaled_data, lookback)

# Reshape for LSTM input [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# === Train/Test Split ===
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# === Build LSTM Model ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# === Train ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=32,
    callbacks = [checkpoint_callback],
    verbose=1
)

# === Predict and Inverse Transform ===
best_model = load_model("checkpoints/best_model.keras")

predictions = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))


x_test = df.select("date").to_numpy().flatten()  # full datetime column
x_test = pd.to_datetime(x_test)
x_test = x_test[-len(y_test_real):] 

# === Plot ===

plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test_real, label="Real Close Price")
plt.plot(x_test, predictions, label="Predicted Close Price")
plt.title("Stock Price Prediction (LSTM + Polars)")
plt.xlabel("Time")
plt.ylabel("Price")
ax = plt.gca()  # get current axes
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)  # rotate x-axis labels for readability
N = max(1, len(x_test)//10)  # adjust number of labels shown
plt.xticks(x_test[::N])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()       # prevent labels from being cut off
plt.savefig("stock_prediction.png")  # save to file
plt.show()