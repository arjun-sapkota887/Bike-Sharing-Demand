# src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

def classification_metrics(model, Xv, yv, Xt, yt):
    """Return validation/test accuracy and F1 scores."""
    pv = model.predict(Xv)
    pt = model.predict(Xt)
    return dict(
        val_acc=accuracy_score(yv, pv),
        val_f1=f1_score(yv, pv),
        test_acc=accuracy_score(yt, pt),
        test_f1=f1_score(yt, pt),
        y_pred_test=pt
    )

def regression_metrics(model, Xv, yv, Xt, yt):
    """Return validation/test MAE and RMSE."""
    pv = model.predict(Xv)
    pt = model.predict(Xt)
    return dict(
        val_mae=mean_absolute_error(yv, pv),
        val_rmse=np.sqrt(mean_squared_error(yv, pv)),
        test_mae=mean_absolute_error(yt, pt),
        test_rmse=np.sqrt(mean_squared_error(yt, pt)),
        y_pred_test=pt
    )

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title("Plot 3 – Confusion Matrix (Best Classifier)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_true, y_pred, save_path):
    plt.figure(figsize=(5.5,4))
    plt.scatter(y_pred, y_true - y_pred, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Plot 4 – Residuals vs Predicted (Best Regressor)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
