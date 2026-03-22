
import torch



# train one epoch
def train_one_epoch(model, epoch, train_loader, device, loss_function, optimizer, writer):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    total_loss = 0.0
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

            total_loss += running_loss
            running_loss = 0.0

    epoch_avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/Train_Epoch', epoch_avg_loss, epoch)
    return epoch_avg_loss

###########################################################################

#validate one epoch
def validate_one_epoch(model, epoch, test_loader, device, loss_function, writer):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    writer.add_scalar('Loss/Val', avg_loss_across_batches, epoch)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()
    return avg_loss_across_batches

###########################################################################
