import tensorflow as tf
import os

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_loss", save_best_only=True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best:
            self.best = current
            self.model.save(self.filepath)
            print(f"   🎉 Epoch {epoch+1}: New best model saved! {self.monitor} = {current:.4f}")