"""
Data loading and basic cleaning/splitting logic.

This mirrors the first part of the midpoint notebook:
- load train.csv
- drop leakage columns (casual, registered)
- convert datetime
- extract hour / weekday / month / year
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import SEED


# Paths relative to project root
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"


def load_raw_train(path: Path | str = TRAIN_PATH) -> pd.DataFrame:
    """Load the raw Kaggle train.csv file."""
    df = pd.read_csv(path)
    return df


def basic_time_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same initial cleaning as in the notebook:

    - Drop 'casual' and 'registered' (data leakage)
    - Convert 'datetime' to pandas datetime
    - Add hour, weekday, month, year
    """
    df = df.copy()

    # Drop leakage columns if present
    for col in ["casual", "registered"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Convert datetime and extract time features
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    return df


def load_and_basic_clean(path: Path | str = TRAIN_PATH) -> pd.DataFrame:
    """Convenience function: load raw training data and apply basic cleaning."""
    df_raw = load_raw_train(path)
    df = basic_time_clean(df_raw)

    # In the notebook you printed shape and checked missingness;
    # we keep this info for debugging but don't print by default.
    assert df.isna().sum().sum() == 0, "Unexpected missing values in training data."

    return df


def make_train_val_test_splits(
    X,
    y_class,
    y_reg,
    seed: int = SEED,
    train_frac: float = 0.70,
):
    """
    70 / 15 / 15 split, exactly like the notebook:

    1) First split into (train 70%, temp 30%), stratifying on y_class
    2) Then split temp into (val 15%, test 15%) by splitting 50/50, again stratified.

    Returns:
        X_train, X_val, X_test,
        y_class_train, y_class_val, y_class_test,
        y_reg_train, y_reg_val, y_reg_test
    """

    # 1) 70% train, 30% temp
    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.30,
        random_state=seed,
        stratify=y_class,
    )

    # 2) 15% val, 15% test (50/50 of temp)
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp,
        y_class_temp,
        y_reg_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_class_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_class_train,
        y_class_val,
        y_class_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
    )
