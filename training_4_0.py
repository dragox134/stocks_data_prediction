import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
#   tensorboard --logdir runs
from helper_functions.save import save_graphs, save_model
from helper_functions.data_loader import load_data
from helper_functions.models import model_switch
from helper_functions.tensorboard_setup import tensorboard
from helper_functions.training_defs import train_one_epoch, validate_one_epoch
from helper_functions.prediction import predict


DEFAULT_SEED = 40


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def train(stock, model_name, learning_rate, num_epochs, batch_size, lookback, days_to_predict, progress_callback=None, update_every=10, seed=DEFAULT_SEED):
    print("training started")
    set_seed(seed)

    # setting device to train
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load data
    last_real_close, last_real_date, train_dates, test_dates, train_loader, test_loader, X_train, lookback, scaler, X_test, y_train, y_test = load_data(batch_size=batch_size, lookback=lookback, name=stock, seed=seed)

    # choose model
    model = model_switch(model_name)    # lstm or trs
    model.to(device)

    # name lstm or trs
    writer, run_name = tensorboard(run_name=model_name, custom="RandomShowcaseTest")  # add custom if you want some specific name after the official one


    # define
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #lstm = 0.001

    best_loss = None
    last_name = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, epoch, train_loader, device, loss_function, optimizer, writer)
        train_losses.append(float(train_loss))

        val_loss = validate_one_epoch(model, epoch, test_loader, device, loss_function, writer)
        val_losses.append(float(val_loss))

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            if last_name and os.path.exists(last_name):
                os.remove(last_name)
            last_name = save_model(model, optimizer, epoch, best_loss, scaler, lookback, run_name)

        if progress_callback and (((epoch + 1) % update_every == 0) or (epoch + 1 == num_epochs)):
            progress_callback(
                epoch=epoch + 1,
                num_epochs=num_epochs,
                train_losses=train_losses,
                val_losses=val_losses,
            )


        writer.flush()

    # Save prediction graphs once after training (instead of every epoch, which creates many overlapping lines)
    save_graphs(model, X_train, device, lookback, scaler, writer, X_test)

    future_pred = predict(run_name, device, last_real_close, X_train, X_test, days_to_predict=days_to_predict)

    writer.close()

    def _inv(vals):
        d = np.zeros((len(vals), lookback + 1))
        d[:, 0] = vals
        return scaler.inverse_transform(d)[:, 0].astype(float).tolist()

    with torch.no_grad():
        train_preds_scaled = model(X_train.to(device)).cpu().numpy().flatten()
        test_preds_scaled = model(X_test.to(device)).cpu().numpy().flatten()

    train_actual = _inv(y_train.numpy().flatten())
    train_pred = _inv(train_preds_scaled)
    test_actual = _inv(y_test.numpy().flatten())
    test_pred = _inv(test_preds_scaled)

    def _r2_fit_score(actual, predicted):
        actual_np = np.asarray(actual, dtype=float)
        pred_np = np.asarray(predicted, dtype=float)

        if actual_np.size < 2:
            return 0.0

        ss_res = float(np.sum((actual_np - pred_np) ** 2))
        ss_tot = float(np.sum((actual_np - np.mean(actual_np)) ** 2))
        eps = 1e-12
        r2 = 1.0 - (ss_res / (ss_tot + eps))
        return max(0.0, min(100.0, r2 * 100.0))

    train_accuracy = _r2_fit_score(train_actual, train_pred)
    test_accuracy = _r2_fit_score(test_actual, test_pred)

    future_dates = pd.bdate_range(
        start=pd.Timestamp(last_real_date) + pd.offsets.BDay(1),
        periods=days_to_predict,
    ).strftime('%Y-%m-%d').tolist()

    return {
        "message": f"Successfully trained {model_name} model for {stock} with lr={learning_rate}, epochs={num_epochs}, batch={batch_size}, lookback={lookback}. Best val loss: {best_loss:.6f}",
        "best_loss": float(best_loss),
        "run_name": run_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "future_dates": future_dates,
        "train_actual": train_actual,
        "train_pred": train_pred,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_future_pred": list(future_pred),
    }