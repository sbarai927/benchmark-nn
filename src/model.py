# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics

def train_and_evaluate(params, X_train, y_train, X_val, y_val):
    """
    Build, compile, and train an MLP according to `params`,
    then return the final validation RMSE.
    """
    # Unpack hyperparameters
    n_layers   = params["n_layers"]
    units      = params["units"]
    lr         = params["lr"]
    batch_size = params["batch_size"]

    # Build the model
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for _ in range(n_layers):
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1))  # single-output regression

    # Compile
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=losses.MeanSquaredError(),
        metrics=[metrics.RootMeanSquaredError()]
    )

    # Train for a fixed number of epochs
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=batch_size,
        verbose=0
    )

    # Return the final validation RMSE
    return history.history["val_root_mean_squared_error"][-1]
