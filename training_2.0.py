import os
import sqlite3
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
from io import BytesIO

import tensorflow as tf  # only for TensorBoard writer             tensorboard --logdir logs_jax --port 6006

# ========= config =========
epochs = 10000
batch_size = 32
lookback = 60
learning_rate = 1e-3


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


base_log_dir  = "logs_jax"
train_log_dir = get_next_run_dir(base_log_dir, "train_run")
summary_writer = tf.summary.create_file_writer(train_log_dir)

# ========= data loading (same as before) =========
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
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# convert to jnp
X_train = jnp.array(X_train)
y_train = jnp.array(y_train).reshape(-1, 1)
X_test  = jnp.array(X_test)
y_test  = jnp.array(y_test).reshape(-1, 1)

# ========= Flax MLP model =========
class PriceMLP(nn.Module):
    hidden_sizes: tuple = (128, 64, 32)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

def create_train_state(rng, learning_rate):
    model = PriceMLP()
    params = model.init(rng, jnp.ones((1, lookback)))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

rng = random.PRNGKey(0)
state = create_train_state(rng, learning_rate)

# ========= batching helpers =========
def get_batches(X, y, batch_size):
    n = X.shape[0]
    idx = np.random.permutation(n)
    for i in range(0, n, batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]

# ========= loss & step functions =========
def mse_loss(params, apply_fn, x, y):
    preds = apply_fn({"params": params}, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        return mse_loss(params, state.apply_fn, x, y)
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

@jax.jit
def eval_step(state, x, y):
    preds = state.apply_fn({"params": state.params}, x)
    loss = jnp.mean((preds - y) ** 2)
    return loss, preds

# ========= training loop with TensorBoard =========
for epoch in range(epochs):
    for xb, yb in get_batches(X_train, y_train, batch_size):
        state = train_step(state, xb, yb)

    val_loss, val_preds = eval_step(state, X_test, y_test)

    with summary_writer.as_default():
        tf.summary.scalar("val_loss", float(val_loss), step=epoch)


    all_X, _ = create_sequences(scaled_data, lookback)
    all_X = jnp.array(all_X)
    _, all_preds = eval_step(state, all_X, jnp.zeros((all_X.shape[0], 1)))

    all_preds_real = scaler.inverse_transform(
        np.array(all_preds).reshape(-1, 1)
    )


    with summary_writer.as_default():
        for t in range(lookback, len(data)):
            pred_t = float(all_preds_real[t - lookback, 0])
            tf.summary.scalar("price", pred_t, step=t)   # <<< SAME TAG "price"

    print(f"Epoch {epoch+1}/{epochs}, val_loss={float(val_loss):.6f}")

# ========= after training: you can still build a pandas index for plotting if needed =========
x_dates = df.select("date").to_numpy().flatten()
x_dates = pd.to_datetime(x_dates)
x_dates = x_dates[-len(y_test):]  # align with test set if you want offline plots
