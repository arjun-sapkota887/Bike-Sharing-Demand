# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import set_seeds

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Kaggle file has 'datetime' & 'count' (sometimes 'cnt' in other variants)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    if 'count' not in df.columns and 'cnt' in df.columns:
        df = df.rename(columns={'cnt':'count'})
    return df

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    dt = df['datetime']
    df['hour']    = dt.dt.hour
    df['weekday'] = dt.dt.weekday
    df['month']   = dt.dt.month
    df['year']    = dt.dt.year
    return df

def add_class_label(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    df['is_peak_hour'] = (df['count'] >= threshold).astype(int)
    return df

def make_splits(df: pd.DataFrame, features: list, seed: int = 42):
    set_seeds(seed)
    X = df[features].copy()
    y_class = df['is_peak_hour'].copy()
    y_reg   = df['count'].copy()

    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.30, random_state=seed, stratify=y_class
    )
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.50, random_state=seed, stratify=y_class_temp
    )
    return (X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, y_reg_train, y_reg_val, y_reg_test)
