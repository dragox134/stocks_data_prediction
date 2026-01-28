import numpy as np
import torch
from copy import deepcopy as dc

###########################################################################
def save(model, X_train, device, lookback, scaler, writer, X_test):
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