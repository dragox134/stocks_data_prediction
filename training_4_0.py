import torch
import torch.nn as nn
import os
#   tensorboard --logdir runs
from helper_functions.save import save_graphs, save_model
from helper_functions.data_loader import load_data
from helper_functions.models import model_switch
from helper_functions.tensorboard_setup import tensorboard
from helper_functions.training_defs import train_one_epoch, validate_one_epoch
from helper_functions.prediction import predict


# setting device to train
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load data
_, train_loader, test_loader, X_train, lookback, scaler, X_test = load_data(batch_size=16)

# choose model
model_name = "lstm"    # lstm or trs
model = model_switch(model_name)
model.to(device)

# name lstm or trs
writer = tensorboard(run_name=model_name)


# define
learning_rate = 0.001       #lstm = 0.001
num_epochs = 100
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch(model, epoch, train_loader, device, loss_function, optimizer, writer)

    loss = validate_one_epoch(model, epoch, test_loader, device, loss_function, writer)
    try:
        if loss < best_loss:
            best_loss = loss
            os.remove(last_name)
            last_name = save_model(model, optimizer, epoch, best_loss, scaler, lookback, model_name)
    except:
        best_loss = loss
        last_name = save_model(model, optimizer, epoch, best_loss, scaler, lookback, model_name)


    writer.flush()
    save_graphs(model, X_train, device, lookback, scaler, writer, X_test)
