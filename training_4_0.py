from pathlib import Path

from save_prices import save

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter       #   tensorboard --logdir runs

from data_loader import load_data
from models import model_switch


# setting device to train
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#prepears tensorboard for logging LSTM
log_dir = 'runs'
run_name = 'test' #lstm     trs
run_index = 1
while (Path(log_dir) / f"{run_name}_stock_{run_index}").exists():
    run_index += 1
log_path = f'{log_dir}/{run_name}_stock_{run_index}'
writer = SummaryWriter(log_path)
print(f"New run saved to: {log_path}")


train_loader, test_loader, X_train, lookback, scaler, X_test = load_data(batch_size=16)
model = model_switch("trs")    # lstm or trs


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

model.to(device)

learning_rate = 0.001       #lstm = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()
    writer.flush()
    save(model, X_train, device, lookback, scaler, writer, X_test)
