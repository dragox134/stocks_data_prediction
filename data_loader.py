import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

from copy import deepcopy as dc

from sklearn.preprocessing import MinMaxScaler

# loading the datam
url = f'https://drive.google.com/uc?export=download&id=1MqY9yaql1XQbodFSngsHxGbyLdWRhVXj'
data = pd.read_csv(url)
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])

###########################################################################

# preparing data and removing blank ones
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)

###########################################################################

# editing data
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

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)



###########################################################################

# loading data into variables
def load_data(batch_size=16):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train, lookback, scaler, X_test
###########################################################################