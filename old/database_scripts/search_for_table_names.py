import sqlite3


db_path = "dbs/AAPL_database_60min.db"
conn = sqlite3.connect(db_path)

# List all tables in the database
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
print("Tables in database:", tables)

conn.close()