"""
Feature engineering and label creation.

This is a direct modularization of your midpoint notebook:
- engineered features: feels_like_gap, rush_hour, hour_sin, hour_cos
- classification label: is_peak_hour
"""

import numpy as np
import pandas as pd

THRESHOLD_PEAK = 100  # IMPORTANT: this matches your final notebook and report.


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features used in the notebook:
      - feels_like_gap = atemp - temp
      - rush_hour: 1 for {7,8,9,16,17,18}, else 0
      - hour_sin, hour_cos: cyclic encoding of hour
      - is_peak_hour: 1 if count >= THRESHOLD_PEAK, else 0
    """
    df = df.copy()

    # Feels like vs actual temperature
    df["feels_like_gap"] = df["atemp"] - df["temp"]

    # Rush hours (morning + evening commute)
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Classification label
    df["is_peak_hour"] = (df["count"] >= THRESHOLD_PEAK).astype(int)

    return df


def build_feature_matrices(df: pd.DataFrame):
    """
    Build X, y_class, y_reg using the same feature set as the notebook.

    Returns:
        X: full design matrix (pandas DataFrame)
        y_class: binary labels for classification
        y_reg: regression target (count)
        feature_groups: dict with 'categorical', 'binary', 'numeric'
    """
    # Feature groups exactly as in the notebook logic
    categorical_features = ["season", "weather"]
    binary_features = ["holiday", "workingday"]
    numeric_features = [
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "hour",
        "weekday",
        "month",
        "year",
        "feels_like_gap",
        "rush_hour",
        "hour_sin",
        "hour_cos",
    ]

    all_features = categorical_features + binary_features + numeric_features

    X = df[all_features].copy()
    y_class = df["is_peak_hour"].copy()
    y_reg = df["count"].copy()

    feature_groups = {
        "categorical": categorical_features,
        "binary": binary_features,
        "numeric": numeric_features,
        "all": all_features,
    }

    return X, y_class, y_reg, feature_groups
