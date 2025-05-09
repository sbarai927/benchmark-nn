# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, callbacks
from optuna.integration import TFKerasPruningCallback


def train_and_evaluate(params, X_train, y_train, X_val, y_val, trial=None):
    """
    Build, compile, and train an MLP according to `params`,
    with optional Optuna pruning via `trial`. Returns final val RMSE.
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

    # Prepare callbacks: pruning + early stopping
    cbks = []
    if trial is not None:
        # prune on the validation RMSE
        cbks.append(
            TFKerasPruningCallback(trial, "val_root_mean_squared_error")
        )
    # stop training if val RMSE doesn’t improve for 3 epochs
    cbks.append(
        callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            patience=3,
            restore_best_weights=True
        )
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,               # you can raise beyond 20 now that we prune
        batch_size=batch_size,
        verbose=0,
        callbacks=cbks
    )

    # Return the final validation RMSE
    return history.history["val_root_mean_squared_error"][-1]
