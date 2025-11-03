# src/features.py
import numpy as np
import pandas as pd

def create_features(df: pd.DataFrame, peak_threshold: int = 100):
    """Generate classification/regression labels and select key features."""
    # Create targets
    df['is_peak_hour'] = (df['count'] >= peak_threshold).astype(int)
    y_class = df['is_peak_hour']
    y_reg = df['count']

    # Minimal interpretable features (team’s choice)
    X = df[['windspeed', 'hour', 'temp', 'humidity']]
    return X, y_class, y_reg
