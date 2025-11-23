from load_and_update import insert_data_into_db

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ"]
test = ["TSLA"]


for i in symbols:
  insert_data_into_db(i, f"dbs/60_mins/{i}_database_60min", "TIME_SERIES_INTRADAY")
  insert_data_into_db(i, f"dbs/daily/{i}_database_daily", "TIME_SERIES_DAILY")

