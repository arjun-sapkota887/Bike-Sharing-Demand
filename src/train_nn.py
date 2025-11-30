# ===============================================================
# train_nn.py
# Final Neural Network Training Script for Bike Sharing Demand
# YoungEverest — CS-4120 Final Project
#
# Reproducible version of the final notebook:
# - Same preprocessing
# - Same Random Search tuning
# - Same NN architectures (classification + regression)
# - MLflow experiment logging
# - Saves trained models to models/
# ===============================================================

import pandas as pd
import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore")

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error,
    confusion_matrix
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# MLflow
import mlflow
import mlflow.keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

mlflow.set_experiment("BikeSharing_Final")

# ===============================================================
# 1. Load and preprocess data
# ===============================================================

df = pd.read_csv("data/train.csv")

# Feature Engineering
df["datetime"] = pd.to_datetime(df["datetime"])

cols_to_drop = [c for c in ["casual", "registered"] if c in df.columns]
df = df.drop(columns=cols_to_drop)

df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year
df["feels_like_gap"] = df["atemp"] - df["temp"]
df["rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

# Classification target
df["is_peak_hour"] = (df["count"] >= 100).astype(int)

# Feature groups
categorical_features = ["hour", "season", "weather", "weekday"]
numeric_features = ["temp", "atemp", "humidity", "windspeed",
                    "feels_like_gap", "month", "year"]

X = df[numeric_features + categorical_features]
y_class = df["is_peak_hour"]
y_reg = df["count"]

# Train/val/test split
X_train_raw, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
    X, y_class, y_reg, test_size=0.30, random_state=SEED, stratify=y_class
)

X_val_raw, X_test_raw, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
    X_temp, y_class_temp, y_reg_temp, test_size=0.50, random_state=SEED, stratify=y_class_temp
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

# Fit-transform
X_train_scaled = preprocessor.fit_transform(X_train_raw).astype("float32")
X_val_scaled = preprocessor.transform(X_val_raw).astype("float32")
X_test_scaled = preprocessor.transform(X_test_raw).astype("float32")

input_dim = X_train_scaled.shape[1]

# ===============================================================
# 2. Model creation function
# ===============================================================

def create_model(input_dim, units=128, dropout_rate=0.2,
                 learning_rate=0.001, output_activation="sigmoid", loss="binary_crossentropy"):
    model = Sequential([
        Dense(units, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(units // 2, activation="relu"),
        BatchNormalization(),
        Dropout(dropout_rate / 2),

        Dense(1, activation=output_activation)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"] if output_activation == "sigmoid" else ["mae"]
    )
    return model

# ===============================================================
# 3. Random Search Hyperparameter Tuning (classification only)
# ===============================================================

param_grid = {
    "units": [64, 128],
    "dropout_rate": [0.1, 0.2, 0.3],
    "learning_rate": [0.001, 0.0005]
}

best_score = 0
best_params = {}

print("\nRunning Random Search (10 iterations)...")

for i in range(10):
    p = {k: random.choice(v) for k, v in param_grid.items()}

    model = create_model(
        input_dim,
        units=p["units"],
        dropout_rate=p["dropout_rate"],
        learning_rate=p["learning_rate"]
    )

    history = model.fit(
        X_train_scaled, y_class_train,
        epochs=20, batch_size=64, verbose=0,
        validation_data=(X_val_scaled, y_class_val)
    )

    val_acc = max(history.history["val_accuracy"])

    print(f"Iter {i+1}: {p} → val_acc={val_acc:.4f}")

    if val_acc > best_score:
        best_score = val_acc
        best_params = p

if best_score < 0.6:
    best_params = {"units": 128, "dropout_rate": 0.2, "learning_rate": 0.001}

print("\nBest Hyperparameters:", best_params)

# ===============================================================
# 4. Train FINAL MODELS (classification + regression)
# ===============================================================

early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, verbose=0)

# ---- CLASSIFICATION FINAL MODEL ----

model_class = create_model(
    input_dim,
    units=best_params["units"],
    dropout_rate=best_params["dropout_rate"],
    learning_rate=best_params["learning_rate"]
)

history_class = model_class.fit(
    X_train_scaled, y_class_train,
    epochs=100, batch_size=64, verbose=1,
    validation_data=(X_val_scaled, y_class_val),
    callbacks=[early_stop, reduce_lr]
)

# ---- REGRESSION FINAL MODEL ----

model_reg = create_model(
    input_dim,
    units=best_params["units"],
    dropout_rate=best_params["dropout_rate"],
    learning_rate=best_params["learning_rate"],
    output_activation="linear",
    loss="mse"
)

history_reg = model_reg.fit(
    X_train_scaled, y_reg_train,
    epochs=100, batch_size=64, verbose=1,
    validation_data=(X_val_scaled, y_reg_val),
    callbacks=[early_stop, reduce_lr]
)

# ===============================================================
# 5. Evaluate models & log to MLflow
# ===============================================================

# ---- Classification Test Metrics ----

y_pred_probs = model_class.predict(X_test_scaled, verbose=0)
y_pred_class = (y_pred_probs >= 0.5).astype(int).flatten()

acc = accuracy_score(y_class_test, y_pred_class)
f1 = f1_score(y_class_test, y_pred_class)

# ---- Regression Test Metrics ----

y_pred_reg = model_reg.predict(X_test_scaled, verbose=0).flatten()
mae = mean_absolute_error(y_reg_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))

# ---- MLflow Logging ----
with mlflow.start_run(run_name="final_nn_models"):
    mlflow.log_params(best_params)

    mlflow.log_metrics({
        "test_accuracy": acc,
        "test_f1": f1,
        "test_mae": mae,
        "test_rmse": rmse
    })

    mlflow.keras.log_model(model_class, "nn_classifier")
    mlflow.keras.log_model(model_reg, "nn_regressor")

# ---- Save models locally ----
os.makedirs("models", exist_ok=True)
model_class.save("models/final_nn_classifier.h5")
model_reg.save("models/final_nn_regressor.h5")

print("\n=== FINAL RESULTS ===")
print("Classification Accuracy:", acc)
print("Classification F1-score:", f1)
print("Regression MAE:", mae)
print("Regression RMSE:", rmse)

print("\nModels saved to /models/")
print("Run logged to MLflow.")
