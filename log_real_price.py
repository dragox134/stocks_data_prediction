# log_real_prices.py
import os
import sqlite3
import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

log_dir = "logs_jax/real_run"
os.makedirs(log_dir, exist_ok=True)
writer = tf.summary.create_file_writer(log_dir)

db_path = "usable_dbs/60_mins/AAPL_database_60min.db"

conn = sqlite3.connect(db_path)
df = pl.read_database("SELECT * FROM stock_data", connection=conn)
conn.close()

df = df.sort("date")
data = df.select("close").to_numpy().astype(float)

# use raw prices on y-axis (no scaler)
with writer.as_default():
    for t, value in enumerate(data.flatten()):
        tf.summary.scalar("price", float(value), step=t)
