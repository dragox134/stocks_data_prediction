import pandas as pd
import numpy as np
import yfinance as yf

from torch.utils.data import Dataset, DataLoader
import torch

from copy import deepcopy as dc

from sklearn.preprocessing import MinMaxScaler


DEFAULT_SEED = 42

# loading the datam
def dataset(name):
    data = yf.download(name, period='max', progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned from Yahoo Finance for ticker: {name}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data.dropna(subset=['Close'], inplace=True)
    return data

###########################################################################

# preparing data and removing blank ones
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

###########################################################################

# editing data
def edit_split_data(shifted_df, lookback):
    shifted_df_as_np = shifted_df.to_numpy()


    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    X = dc(np.flip(X, axis=1))

    # splitting and editing for train and test
    split_index = int(len(X) * 0.95)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]


    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))


    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return X_train, y_train, X_test, y_test, scaler, split_index

###########################################################################

# defing dataset in classes
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]



###########################################################################

# loading data into variables
def load_data(batch_size, lookback, name, seed=DEFAULT_SEED):

    data = dataset(name)

    shifted_df = prepare_dataframe_for_lstm(data, lookback)

    X_train, y_train, X_test, y_test, scaler, split_index = edit_split_data(shifted_df, lookback)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    shifted_dates = [timestamp.strftime('%Y-%m-%d') for timestamp in shifted_df.index.to_list()]
    train_dates = shifted_dates[:split_index]
    test_dates = shifted_dates[split_index:]

    last_real_close = float(data['Close'].iloc[split_index + lookback:].iloc[-1])
    last_real_date = shifted_dates[-1]

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return last_real_close, last_real_date, train_dates, test_dates, train_loader, test_loader, X_train, lookback, scaler, X_test, y_train, y_test
###########################################################################