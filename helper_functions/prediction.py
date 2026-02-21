import torch
import numpy as np
from copy import deepcopy as dc
from torch.utils.tensorboard import SummaryWriter

from helper_functions.models import model_switch
from helper_functions.data_loader import load_data




def load_checkpoint(path, device):

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "lstm" in path:
        model_name = "lstm"
    elif "trs" in path:
        model_name = "trs"
    else:
        raise ValueError("Cannot determine model type from path. Make sure path contains 'lstm' or 'trs'.")

    model = model_switch(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler  = checkpoint["scaler"]
    lookback = checkpoint["lookback"]

    print(f"Loaded '{model_name}' model from epoch {checkpoint['epoch']} "
          f"(val_loss: {checkpoint['best_val_loss']:.6f})")
    print(f"Lookback: {lookback} days\n")

    return model, scaler, lookback


# rolling window prediction function
def predict_future(model, last_window, scaler, lookback, days_to_predict, device):

    rolling_window = list(last_window)   # e.g. 7 scaled floats
    predictions_scaled = []   # we collect the raw scaled outputs here

    for day in range(days_to_predict):

        input_seq = np.array(rolling_window[-lookback:]).reshape(1, lookback, 1)
        input_tensor = torch.tensor(input_seq).float().to(device)

        # --- Run the model ---
        with torch.no_grad():   # no need to track gradients during prediction
            pred_scaled = model(input_tensor).item()   # .item() converts tensor -> plain float

        rolling_window.append(pred_scaled)
        predictions_scaled.append(pred_scaled)

        print(f"  Day {day + 1}: predicted (scaled) = {pred_scaled:.4f}")

    dummies = np.zeros((days_to_predict, lookback + 1))
    dummies[:, 0] = predictions_scaled
    dummies_unscaled = scaler.inverse_transform(dummies)

    predictions_real = dc(dummies_unscaled[:, 0])   # extract only the Close column
    return predictions_real




def log_to_tensorboard(writer, X_train, X_test, predictions_real, last_real_close):

    offset = X_train.shape[0] + X_test.shape[0] - 1

    # Anchor: the actual last known price, pulled directly from the data
    writer.add_scalar('Test/Close', last_real_close, offset)

    for i, price in enumerate(predictions_real):
        writer.add_scalar('Test/Close', price, offset + 1 + i)

    writer.flush()







def predict(name, device, last_real_close, X_train, X_test, days_to_predict=7):

    MODEL_PATH = f"models/{name}_model"
    LOG_PATH = f"runs/predictions/{name}_prediction"

    model, scaler, lookback = load_checkpoint(MODEL_PATH, device)

    last_real_window = X_test[-1].numpy().flatten()   # shape: (lookback,)

    print(f"Starting prediction from the last {lookback} real data points.")
    print(f"Predicting {days_to_predict} days into the future...\n")


    predictions = predict_future(
        model=model,
        last_window=last_real_window,
        scaler=scaler,
        lookback=lookback,
        days_to_predict=days_to_predict,
        device=device
    )


    print(f"\n{'='*40}")
    print(f"  Predicted closing prices (next {days_to_predict} days)")
    print(f"{'='*40}")
    for i, price in enumerate(predictions, start=1):
        print(f"  Day {i:>2}: ${price:.2f}")
    print(f"{'='*40}\n")


    writer = SummaryWriter(f'{LOG_PATH}')
    log_to_tensorboard(writer, X_train, X_test, predictions, last_real_close)
    writer.close()