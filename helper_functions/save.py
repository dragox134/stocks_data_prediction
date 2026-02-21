import numpy as np
import torch
from copy import deepcopy as dc

###########################################################################
def save_graphs(model, X_train, device, lookback, scaler, writer, X_test):
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

    ###########################################################################

    # saves graph close vs preedicted (actual price) on training data
    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)

    train_predictions = dc(dummies[:, 0])
    # train_predictions

    for i, price in enumerate(train_predictions):
        writer.add_scalar('Train/Close', price, i)

    ###########################################################################

    # saves graph close vs predicted (actual price) on test data
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])
    test_predictions

    train_length = X_train.shape[0]
    for i, price in enumerate(test_predictions):
        writer.add_scalar('Test/Close', price, train_length + i)

    ###########################################################################


def save_model(model, optimizer, epoch, best_val_loss, scaler, lookback, model_name, path="models/"):
    print(f"new best model (val_loss: {best_val_loss:.6f})")
    path = f"{path}{model_name}_{epoch}_loss:{best_val_loss}"

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "scaler": scaler,
        "lookback": lookback,
    }, path)

    return path