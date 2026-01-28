import torch
import torch.nn as nn

from save_prices import save
from data_loader import load_data
from models import model_switch
from tensorboard_setup import tensorboard
from training_defs import train_one_epoch, validate_one_epoch


# setting device to train
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# name lstm or trs
writer = tensorboard(run_name='lstm')

# load data
train_loader, test_loader, X_train, lookback, scaler, X_test = load_data(batch_size=16)

# choose model
model = model_switch("lstm")    # lstm or trs
model.to(device)




# define
learning_rate = 0.001       #lstm = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch(model, epoch, train_loader, device, loss_function, optimizer, writer)
    validate_one_epoch(model, epoch, test_loader, device, loss_function, writer)
    writer.flush()
    save(model, X_train, device, lookback, scaler, writer, X_test)
