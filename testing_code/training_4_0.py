import pandas as pd
import numpy as np
from pathlib import Path

from save_prices import save

import torch
import torch.nn as nn
from copy import deepcopy as dc

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter       #   tensorboard --logdir runs



for i in range(5):
    # loading the datam
    file_id = '1MqY9yaql1XQbodFSngsHxGbyLdWRhVXj'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    data = pd.read_csv(url)
    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])


    # setting device to train
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    #prepears tensorboard for logging LSTM
    log_dir = 'runs'
    run_name = 'trs' #lstm     trs
    run_index = 1
    while (Path(log_dir) / f"{run_name}_stock_{run_index}").exists():
        run_index += 1
    log_path = f'{log_dir}/{run_name}_stock_{run_index}'
    writer = SummaryWriter(log_path)
    print(f"New run saved to: {log_path}")



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
    shifted_df


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
    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ###########################################################################

    # ???
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    ###########################################################################

    # LSTM model for training
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                                batch_first=True)

            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


    ###########################################################################

    # Transformer model for training
    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers
            self.input_size = input_size
            

            self.input_projection = nn.Linear(input_size, hidden_size)
            
    
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_stacked_layers)
            
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            batch_size = x.size(0)
            
            x = self.input_projection(x)
            
            out = self.transformer(x)
            
            out = self.fc(out[:, -1, :])
            return out



    ###########################################################################

    # train one epoch
    def train_one_epoch():
        model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0
        global_step = epoch * len(train_loader)

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                
                writer.add_scalar('Loss/Train', running_loss / 100, global_step + batch_index)

                running_loss = 0.0


        writer.add_scalar('Loss/Train_Epoch', running_loss / len(train_loader), epoch)
        print()

    ###########################################################################

    #validate one epoch
    def validate_one_epoch():
        model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        writer.add_scalar('Loss/Val', running_loss / len(test_loader), epoch)

        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()

    ###########################################################################

    # training loop (choose model)
    model = TransformerModel(1, 4, 1)
    # model = LSTM(1, 4, 1)

    model.to(device)

    learning_rate = 0.001       #lstm = 0.001
    num_epochs = 100
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()
        writer.flush()
        save(model, X_train, device, lookback, scaler, writer, X_test)
