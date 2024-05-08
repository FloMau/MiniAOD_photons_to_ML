import numpy as np
from sklearn.metrics import log_loss

class WeightedValidationCallback(Callback):
    def __init__(self, validation_data, validation_sample_weights):
        super().__init__()
        self.validation_data = validation_data
        self.validation_sample_weights = validation_sample_weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = self.model.predict(self.validation_data[0])
        val_targ = self.validation_data[1]
        _val_loss = log_loss(val_targ, val_predict, sample_weight=self.validation_sample_weights)
        logs['val_loss_weighted'] = _val_loss
        print(f"Epoch: {epoch+1}, Val Loss (Weighted): {_val_loss}")

# Assuming you have your model, training data, and validation data ready
# X_train, y_train: Training data and labels
# X_val, y_val: Validation data and labels
# sample_weights: Sample weights for the training data
# val_sample_weights: Sample weights for the validation data

# Make sure your validation sample weights are normalized properly
# You can normalize weights by making sure that the sum of the weights in each class equals the number of samples in that class

# Create an instance of the custom callback
weighted_val_callback = WeightedValidationCallback(validation_data=(X_val, y_val), validation_sample_weights=val_sample_weights)

# Train your model
history = model.fit(X_train, y_train, sample_weight=sample_weights,
                    validation_data=(X_val, y_val),
                    epochs=10, callbacks=[weighted_val_callback])