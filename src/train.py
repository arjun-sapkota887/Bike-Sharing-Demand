"""
Neural network training utilities for the final project.

This file implements:
  - Classification MLP for is_peak_hour.
  - Regression MLP for count (using log1p transform + Huber loss).
  - Simple random search for hyperparameters for both tasks.

The models expect:
  - X_* : preprocessed NumPy arrays (after scaling + one-hot encoding)
  - y_class_* : binary labels (0/1)
  - y_reg_* : integer bike counts
"""

import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import set_seed, SEED


# -------------------------------------------------------------------
# Helper: generic MLP builders
# -------------------------------------------------------------------

def build_classification_mlp(input_dim: int,
                             units: int = 64,
                             dropout: float = 0.2,
                             learning_rate: float = 1e-3) -> tf.keras.Model:
    """Binary classification MLP (sigmoid output)."""
    model = Sequential([
        Dense(units, activation="relu", input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(units // 2, activation="relu"),
        Dropout(dropout / 2 if dropout > 0 else 0.0),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regression_mlp(input_dim: int,
                         units: int = 128,
                         depth: int = 3,
                         dropout: float = 0.0,
                         learning_rate: float = 2e-3) -> tf.keras.Model:
    """
    Regression MLP trained on log1p(count), using Huber loss.
    Output is linear; you must expm1() predictions outside.
    """
    layers = []
    # First hidden layer
    layers.append(Dense(units, activation="relu", input_shape=(input_dim,)))
    if dropout > 0:
        layers.append(Dropout(dropout))

    # Additional hidden layers
    for _ in range(depth - 1):
        layers.append(Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(Dropout(dropout))

    # Output
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=["mae"],  # MAE on log scale (monitoring only)
    )
    return model


# -------------------------------------------------------------------
# Random search for hyperparameters
# -------------------------------------------------------------------

def tune_classification_nn(X_train, y_train, X_val, y_val,
                           n_iter: int = 10,
                           verbose: bool = True):
    """
    Random search over a small hyperparameter grid.
    Returns (best_model, best_history, best_params).
    """
    set_seed(SEED)

    param_grid = {
        "units": [64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [1e-3, 5e-4],
    }

    input_dim = X_train.shape[1]
    best_val_acc = 0.0
    best_model = None
    best_history = None
    best_params = None

    for i in range(n_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}

        model = build_classification_mlp(
            input_dim=input_dim,
            units=params["units"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )

        early = EarlyStopping(monitor="val_loss", patience=10,
                              restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=4, verbose=0)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=64,
            callbacks=[early, reduce_lr],
            verbose=0,
        )

        val_acc = max(history.history["val_accuracy"])
        if verbose:
            print(f"[Class NN] Iter {i+1}/{n_iter} {params} -> "
                  f"best val acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_history = history
            best_params = params

    if verbose:
        print("\nBest classification NN params:", best_params)
        print("Best validation accuracy:", best_val_acc)

    return best_model, best_history, best_params


def tune_regression_nn(X_train, y_reg_train,
                       X_val, y_reg_val,
                       n_iter: int = 12,
                       verbose: bool = True):
    """
    Random search for regression NN.

    IMPORTANT:
      - Trains on log1p(y)
      - Uses Huber loss
      - Selects hyperparameters based on validation MAE in ORIGINAL scale.
    """
    set_seed(SEED)

    # Log-transform targets for training
    y_train_log = np.log1p(y_reg_train)
    y_val_log = np.log1p(y_reg_val)

    param_grid = {
        "units": [64, 128, 256],
        "depth": [2, 3, 4],
        "dropout": [0.0, 0.1],
        "learning_rate": [1e-3, 2e-3, 3e-3],
    }

    input_dim = X_train.shape[1]
    best_val_mae = np.inf
    best_model = None
    best_history = None
    best_params = None

    for i in range(n_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}

        model = build_regression_mlp(
            input_dim=input_dim,
            units=params["units"],
            depth=params["depth"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )

        early = EarlyStopping(monitor="val_loss", patience=10,
                              restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=4, verbose=0)

        history = model.fit(
            X_train,
            y_train_log,
            validation_data=(X_val, y_val_log),
            epochs=100,
            batch_size=64,
            callbacks=[early, reduce_lr],
            verbose=0,
        )

        # Evaluate on validation set in ORIGINAL scale
        val_pred_log = model.predict(X_val, verbose=0).flatten()
        val_pred = np.expm1(val_pred_log)
        val_mae = np.mean(np.abs(y_reg_val - val_pred))

        if verbose:
            print(f"[Reg NN] Iter {i+1}/{n_iter} {params} -> "
                  f"val MAE (orig) = {val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_history = history
            best_params = params

    if verbose:
        print("\nBest regression NN params:", best_params)
        print("Best validation MAE (original scale):", best_val_mae)

    return best_model, best_history, best_params


def main():
    """
    Optional: keep a placeholder CLI entrypoint
    so running `python train_nn.py` doesn't break.
    """
    set_seed(SEED)
    print(
        "train_nn.py is now fully implemented for the final project.\n"
        "Use tune_classification_nn(...) and tune_regression_nn(...) from "
        "your Jupyter notebook to train the final models."
    )


if __name__ == "__main__":
    main()
