# src/evaluate.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error
)
from utils import ensure_dir

def classification_metrics(model, Xv, yv, Xt, yt):
    pv = model.predict(Xv); pt = model.predict(Xt)
    out = {
        'val_accuracy': accuracy_score(yv, pv),
        'val_f1': f1_score(yv, pv),
        'test_accuracy': accuracy_score(yt, pt),
        'test_f1': f1_score(yt, pt)
    }
    return out

def regression_metrics(model, Xv, yv, Xt, yt):
    pv = model.predict(Xv); pt = model.predict(Xt)
    out = {
        'val_mae':  mean_absolute_error(yv, pv),
        'val_rmse': mean_squared_error(yv, pv, squared=False),
        'test_mae':  mean_absolute_error(yt, pt),
        'test_rmse': mean_squared_error(yt, pt, squared=False)
    }
    return out

def plot_target_distribution(df, outdir):
    ensure_dir(outdir)
    ax = sns.countplot(x='is_peak_hour', data=df)
    ax.set_title("Plot 1 – Peak vs Off-peak Class Distribution")
    plt.tight_layout(); plt.savefig(f"{outdir}/plot1_target_distribution.png"); plt.close()

def plot_corr(df, features, outdir):
    ensure_dir(outdir)
    plt.figure(figsize=(8,6))
    sns.heatmap(df[features+['count']].corr(), annot=True, cmap='coolwarm')
    plt.title("Plot 2 – Correlation Heatmap")
    plt.tight_layout(); plt.savefig(f"{outdir}/plot2_corr_heatmap.png"); plt.close()

def plot_confusion(model, Xt, yt, outdir):
    ensure_dir(outdir)
    ConfusionMatrixDisplay.from_estimator(model, Xt, yt)
    plt.title("Plot 3 – Confusion Matrix (Best Classifier)")
    plt.tight_layout(); plt.savefig(f"{outdir}/plot3_confusion_matrix.png"); plt.close()

def plot_residuals(model, Xt, yt, outdir):
    ensure_dir(outdir)
    y_pred = model.predict(Xt)
    resid = yt - y_pred
    plt.scatter(y_pred, resid, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted"); plt.ylabel("Residuals")
    plt.title("Plot 4 – Residuals vs Predicted (Best Regressor)")
    plt.tight_layout(); plt.savefig(f"{outdir}/plot4_residuals_vs_pred.png"); plt.close()
