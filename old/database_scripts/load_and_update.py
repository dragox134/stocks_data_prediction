import pandas as pd
import requests
from sqlalchemy import create_engine, Column, Float, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from api.api_functions import find_available, increment_tries

# symbol = GOOGL, df_name = google_database_60min, function = TIME_SERIES_INTRADAY
def insert_data_into_db(symbol, df_name, function):
    # === STEP 1: GET DATA FROM ALPHA VANTAGE ===
    API_KEY = find_available()
    SYMBOL = symbol
    if function == "TIME_SERIES_INTRADAY":
        INTERVAL = "60min"
    else:
        INTERVAL = None
    URL = "https://www.alphavantage.co/query"

    params = {
        "function": function,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "apikey": API_KEY,
        "outputsize": "compact",
        "datatype": "json"
    }

    increment_tries(API_KEY)
    response = requests.get(URL, params=params)
    data = response.json()

    if "Meta Data" not in data:
        print("\n ERROR: Alpha Vantage did not return 'Meta Data'.")
        print("Full response:")
        print(data)
        return

    # Access the time series data using the key provided in the metadata
    if function == "TIME_SERIES_INTRADAY":
        time_series_key = f"Time Series ({data['Meta Data']['4. Interval']})"
    elif function == "TIME_SERIES_DAILY":
        time_series_key = "Time Series (Daily)"
    df = pd.DataFrame.from_dict(data[time_series_key], orient="index", dtype=float)


    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    # === STEP 2: DATABASE SETUP ===
    Base = declarative_base()

    class StockData(Base):
        __tablename__ = 'stock_data'

        date = Column(DateTime, primary_key=True, unique=True)
        open = Column(Float)
        high = Column(Float)
        low = Column(Float)
        close = Column(Float)
        volume = Column(Integer)

        def __init__(self, date, open_price, high_price, low_price, close_price, volume):
            self.date = date
            self.open = open_price
            self.high = high_price
            self.low = low_price
            self.close = close_price
            self.volume = volume

        def __repr__(self):
            return f"| ({self.date}) Open: {self.open}, High: {self.high}, Low: {self.low}, Close: {self.close}, Volume: {self.volume} |"

    # Create SQLite DB and session
    engine = create_engine(f"sqlite:///{df_name}.db", echo=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # === STEP 3: INSERT DATA INTO DATABASE ===
    for idx, row in df.iterrows():
        record = StockData(
            date=idx,
            open_price=row["open"],
            high_price=row["high"],
            low_price=row["low"],
            close_price=row["close"],
            volume=int(row["volume"])
        )
        session.merge(record)

    session.commit()

    print("Data successfully inserted into the database!")
