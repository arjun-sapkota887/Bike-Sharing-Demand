
"""
Evaluation and plotting helpers.

These implement the four plots required in the midpoint report:
  - Plot 1: target distribution (bar plot of class counts)
  - Plot 2: correlation heatmap for selected numeric features
  - Plot 3: confusion matrix for best classifier
  - Plot 4: residuals vs predicted for best regressor
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_target_distribution(y_class: pd.Series, threshold: int) -> None:
    """Bar plot of peak vs off-peak counts (Plot 1)."""
    counts = y_class.value_counts().sort_index()

    plt.figure(figsize=(4, 4))
    bars = plt.bar(["Off-peak (0)", "Peak (1)"], counts.values)
    plt.title(f"Target Distribution – is_peak_hour (Threshold = {threshold})")
    plt.xlabel("is_peak_hour")
    plt.ylabel("Number of Hours")
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 str(val), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Correlation heatmap between key numeric features and 'count' (Plot 2).

    Matches the variables in your notebook figure:
      temp, atemp, humidity, windspeed, hour, weekday, month, year, count
    """
    cols = ["temp", "atemp", "humidity", "windspeed", "hour",
            "weekday", "month", "year", "count"]
    corr = df[cols].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        square=True,
    )
    plt.title("Correlation Heatmap – Selected Features and Count")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, threshold: int) -> None:
    """Confusion matrix for best classifier on test set (Plot 3)."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"Confusion Matrix – Decision Tree Classifier (Threshold = {threshold})")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred) -> None:
    """Residuals vs predicted for best regressor (Plot 4)."""
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Count")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted – Decision Tree Regressor")
    plt.tight_layout()
    plt.show()
