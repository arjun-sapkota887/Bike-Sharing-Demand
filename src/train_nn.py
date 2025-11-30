
"""
Placeholder for neural network training (final project phase).

Right now this file just defines the structure and can be safely run.
Later, you can implement an MLP that reuses the same encoded + scaled
features as in train_baselines.py.
"""

"""
from utils import set_seed, SEED


def main():
    set_seed(SEED)
    print(
        "train_nn.py placeholder.\n"
        "Neural network training will be implemented for the final report, "
        "using the cleaned and encoded data from the baseline pipeline."
    )


if __name__ == "__main__":
    main()
"""

"""
Final Neural Network Implementation for Bike Sharing Demand.
This script performs:
1. Data Loading & Advanced Preprocessing (One-Hot Encoding)
2. Hyperparameter Tuning (Random Search)
3. Final Model Training (Classification & Regression)
4. Generation of required plots and comparison tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, 
    mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.compose import ColumnTransformer

# TensorFlow Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.inspection import permutation_importance

# Set seeds for reproducibility
SEED = 42
def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def load_and_preprocess_data():
    """
    Loads data and applies advanced preprocessing including One-Hot Encoding
    for cyclic time features which is crucial for NN performance.
    """
    if not os.path.exists("train.csv"):
        raise FileNotFoundError("The file 'train.csv' was not found. Please upload it to the working directory.")

    print("\n--- 1. Loading and Preprocessing Data ---")
    df = pd.read_csv("train.csv")
    print(f"Dataset loaded. Dimensions: {df.shape}")

    # Feature Engineering
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Drop Leakage Columns
    cols_to_drop = [c for c in ['casual', 'registered'] if c in df.columns]
    df = df.drop(columns=cols_to_drop) 

    # Extract Datetime Features
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['feels_like_gap'] = df['atemp'] - df['temp']
    
    # Create Targets
    df['is_peak_hour'] = (df['count'] >= 100).astype(int) # Classification Target
    
    # Check Class Balance
    print("Class Balance (is_peak_hour):")
    print(df['is_peak_hour'].value_counts(normalize=True))

    # Define Feature Sets
    # Note: We treat 'hour', 'season', 'weather', 'weekday' as categorical for OHE
    categorical_features = ['hour', 'season', 'weather', 'weekday']
    numeric_features = ['temp', 'atemp', 'humidity', 'windspeed', 'feels_like_gap', 'month', 'year']

    X = df[numeric_features + categorical_features] 
    y_class = df['is_peak_hour']
    y_reg = df['count']

    # Train/Val/Test Split (70/15/15)
    X_train_raw, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.30, random_state=SEED, stratify=y_class
    )
    X_val_raw, X_test_raw, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.50, random_state=SEED, stratify=y_class_temp
    )

    # Pipeline Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Transform Data
    X_train_scaled = preprocessor.fit_transform(X_train_raw).astype('float32')
    X_val_scaled = preprocessor.transform(X_val_raw).astype('float32')
    X_test_scaled = preprocessor.transform(X_test_raw).astype('float32')

    print(f"Preprocessing Complete. Feature Matrix Shape: {X_train_scaled.shape}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_class_train, y_class_val, y_class_test, 
            y_reg_train, y_reg_val, y_reg_test, 
            preprocessor, categorical_features, numeric_features)

def create_model(input_dim, units=64, dropout_rate=0.2, learning_rate=0.001, output_act='sigmoid', loss='binary_crossentropy'):
    """
    Builds a compiled Keras model based on hyperparameters.
    """
    model = Sequential([
        Dense(units, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units // 2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate / 2),
        Dense(1, activation=output_act)
    ])
    
    metric = 'accuracy' if output_act == 'sigmoid' else 'mae'
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=[metric])
    return model

def run_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_dim):
    print("\n--- 2. Hyperparameter Tuning Summary (Random Search) ---")
    
    param_grid = {
        'units': [64, 128],            
        'dropout_rate': [0.1, 0.2, 0.3], 
        'learning_rate': [0.001, 0.0005] 
    }

    best_score = 0
    best_params = {}

    print("Tuning Classification Model (10 iterations)...")
    for i in range(10): 
        p = {k: random.choice(v) for k, v in param_grid.items()}
        
        # Train quick model to find best params
        model = create_model(input_dim, units=p['units'], dropout_rate=p['dropout_rate'], learning_rate=p['learning_rate'])
        history = model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0, validation_data=(X_val, y_val))
        
        val_acc = max(history.history['val_accuracy'])
        print(f"Iter {i+1}: {p} -> Val Acc: {val_acc:.4f}")
        
        if val_acc > best_score:
            best_score = val_acc
            best_params = p

    print(f"Best Params found: {best_params}")
    return best_params, best_score

class KerasWrapper:
    """Wrapper for Keras models to work with sklearn permutation importance"""
    def __init__(self, model): 
        self.model = model
    def fit(self, X, y): 
        return self
    def predict(self, X): 
        return (self.model.predict(X, verbose=0) >= 0.5).astype(int).flatten()
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def main():
    set_seed(SEED)
    print("TensorFlow Version:", tf.__version__)

    # 1. Load Data
    (X_train, X_val, X_test, 
     y_class_train, y_class_val, y_class_test, 
     y_reg_train, y_reg_val, y_reg_test, 
     preprocessor, cat_feats, num_feats) = load_and_preprocess_data()
    
    input_dim = X_train.shape[1]

    # 2. Tuning
    best_params, best_score = run_hyperparameter_tuning(X_train, y_class_train, X_val, y_class_val, input_dim)

    # Fallback if tuning fails
    if best_score < 0.6:
        print("Warning: Random Search found poor params. Using manual fallback.")
        best_params = {'units': 128, 'dropout_rate': 0.2, 'learning_rate': 0.001}

    # 3. Training Final Models
    print("\n--- 3. Training Final Models ---")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    ]

    # Classification
    model_class = create_model(input_dim, **best_params, output_act='sigmoid', loss='binary_crossentropy')
    hist_class = model_class.fit(
        X_train, y_class_train,
        validation_data=(X_val, y_class_val),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=0
    )

    # Regression
    model_reg = create_model(input_dim, **best_params, output_act='linear', loss='mse')
    hist_reg = model_reg.fit(
        X_train, y_reg_train,
        validation_data=(X_val, y_reg_val),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=0
    )

    # 4. Plots
    # Plot 1
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_class.history['accuracy'], label='Train')
    plt.plot(hist_class.history['val_accuracy'], label='Val')
    plt.title("Plot 1: Classification Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist_class.history['loss'], label='Train')
    plt.plot(hist_class.history['val_loss'], label='Val')
    plt.title("Classification Loss")
    plt.legend()
    plt.show()

    # Plot 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_reg.history['mae'], label='Train')
    plt.plot(hist_reg.history['val_mae'], label='Val')
    plt.title("Plot 2: Regression MAE")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist_reg.history['loss'], label='Train')
    plt.plot(hist_reg.history['val_loss'], label='Val')
    plt.title("Regression Loss")
    plt.legend()
    plt.show()

    # Plot 3
    y_pred_prob = model_class.predict(X_test, verbose=0)
    y_pred_class = (y_pred_prob >= 0.5).astype(int).flatten()
    cm = confusion_matrix(y_class_test, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Off-peak', 'Peak'])
    disp.plot(cmap='Blues')
    plt.title("Plot 3: Confusion Matrix")
    plt.show()

    # Plot 4
    y_pred_reg = model_reg.predict(X_test, verbose=0).flatten()
    residuals = y_reg_test - y_pred_reg
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_reg, residuals, alpha=0.3, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Count")
    plt.ylabel("Residuals")
    plt.title("Plot 4: Residuals vs Predicted")
    plt.show()

    # Plot 5
    print("Calculating Feature Importance...")
    wrapper = KerasWrapper(model_class)
    perm_results = permutation_importance(wrapper, X_val, y_class_val, n_repeats=5, random_state=SEED)
    
    ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_feats)
    all_feats = num_feats + list(ohe_cols)
    imp_df = pd.DataFrame({'feature': all_feats, 'importance': perm_results.importances_mean})
    top_10 = imp_df.sort_values(by='importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_10, orient='h', color='teal')
    plt.title("Plot 5: Feature Importance")
    plt.show()

    # 5. Tables
    nn_acc = accuracy_score(y_class_test, y_pred_class)
    nn_f1 = f1_score(y_class_test, y_pred_class)
    nn_mae = mean_absolute_error(y_reg_test, y_pred_reg)
    nn_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))

    # Baseline Placeholders (Replace with your actual Midpoint values!)
    dt_acc = 0.8959 
    dt_f1 = 0.9154
    dt_mae = 51.52
    dt_rmse = 84.63

    print("\n--- Table 1: Classification Comparison ---")
    tbl1 = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score'],
        'Classical (Decision Tree)': [dt_acc, dt_f1],
        'Neural Network (Final)': [nn_acc, nn_f1]
    })
    print(tbl1)

    print("\n--- Table 2: Regression Comparison ---")
    tbl2 = pd.DataFrame({
        'Metric': ['MAE', 'RMSE'],
        'Classical (Decision Tree)': [dt_mae, dt_rmse],
        'Neural Network (Final)': [nn_mae, nn_rmse]
    })
    print(tbl2)

if __name__ == "__main__":
    main()
