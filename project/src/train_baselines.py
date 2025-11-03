# project/src/train_baselines.py
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

np.random.seed(42)

def load_and_featurize(csv_path):
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Drop leakage
    for col in ['casual','registered']:
        if col in df.columns: df.drop(columns=col, inplace=True)
    # simple features (4)
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    X = df[['windspeed','hour','temp','humidity']].copy()
    y_reg = df['count'].astype(float)
    y_class = (df['count'] >= 100).astype(int)
    return df, X, y_reg, y_class

def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

def main(args):
    ensure_dirs()
    df, X, y_reg, y_class = load_and_featurize(args.data)

    # Plot 1: target distribution (classification)
    plt.figure(figsize=(5,4))
    sns.countplot(x=y_class)
    plt.title("Plot 1 – Target Distribution (is_peak_hour)")
    plt.xlabel("Class (0=off-peak, 1=peak)")
    plt.tight_layout(); plt.savefig("figures/plot1_target_distribution.png"); plt.close()

    # Plot 2: correlation heatmap (key numeric + target)
    num = X.copy()
    num['count'] = y_reg
    plt.figure(figsize=(6,5))
    sns.heatmap(num.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Plot 2 – Correlation Heatmap (Key Numeric Features)")
    plt.tight_layout(); plt.savefig("figures/plot2_corr_heatmap.png"); plt.close()

    # Split 70/15/15 with stratify on classification target
    X_train, X_tmp, y_class_train, y_class_tmp, y_reg_train, y_reg_tmp = train_test_split(
        X, y_class, y_reg, test_size=0.30, random_state=42, stratify=y_class
    )
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_tmp, y_class_tmp, y_reg_tmp, test_size=0.50, random_state=42, stratify=y_class_tmp
    )

    # === Classification baselines ===
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42).fit(X_train_s, y_class_train)
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_class_train)

    def cls_metrics(model, Xv, yv, Xt, yt, scaled=False):
        Xv_ = X_val_s if scaled else Xv
        Xt_ = X_test_s if scaled else Xt
        pv = model.predict(Xv_); pt = model.predict(Xt_)
        return dict(
            val_acc=accuracy_score(yv, pv),
            val_f1=f1_score(yv, pv),
            test_acc=accuracy_score(yt, pt),
            test_f1=f1_score(yt, pt),
            yhat_test=pt
        )

    m_lr   = cls_metrics(log_reg, X_val, y_class_val, X_test, y_class_test, scaled=True)
    m_tree = cls_metrics(tree_clf, X_val, y_class_val, X_test, y_class_test, scaled=False)

    best_clf, use_scaled = (tree_clf, False) if m_tree['test_f1'] >= m_lr['test_f1'] else (log_reg, True)
    X_for_cm = X_test_s if use_scaled else X_test
    y_hat = best_clf.predict(X_for_cm)

    # Plot 3: confusion matrix (best classifier)
    cm = confusion_matrix(y_class_test, y_hat)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title("Plot 3 – Confusion Matrix (Best Classifier)")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.tight_layout(); plt.savefig("figures/plot3_confusion_matrix.png"); plt.close()

    # === Regression baselines ===
    lin_reg = LinearRegression().fit(X_train, y_reg_train)
    tree_reg = DecisionTreeRegressor(random_state=42).fit(X_train, y_reg_train)

    def reg_metrics(model, Xv, yv, Xt, yt):
        pv = model.predict(Xv); pt = model.predict(Xt)
        return dict(
            val_mae=mean_absolute_error(yv, pv),
            val_rmse=np.sqrt(mean_squared_error(yv, pv)),
            test_mae=mean_absolute_error(yt, pt),
            test_rmse=np.sqrt(mean_squared_error(yt, pt)),
            yhat_test=pt
        )

    mr_lin  = reg_metrics(lin_reg,  X_val, y_reg_val, X_test, y_reg_test)
    mr_tree = reg_metrics(tree_reg, X_val, y_reg_val, X_test, y_reg_test)

    best_reg = tree_reg if mr_tree['test_rmse'] <= mr_lin['test_rmse'] else lin_reg
    y_pred = best_reg.predict(X_test)

    # Plot 4: residuals vs predicted (best regressor)
    plt.figure(figsize=(5.5,4))
    plt.scatter(y_pred, y_reg_test - y_pred, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted"); plt.ylabel("Residual")
    plt.title("Plot 4 – Residuals vs Predicted (Best Regressor)")
    plt.tight_layout(); plt.savefig("figures/plot4_residuals_vs_pred.png"); plt.close()

    # Tables
    t1 = pd.DataFrame([
        {"Model":"Logistic Regression", "Val Accuracy":m_lr['val_acc'], "Val F1":m_lr['val_f1'],
         "Test Accuracy":m_lr['test_acc'], "Test F1":m_lr['test_f1']},
        {"Model":"Decision Tree Classifier", "Val Accuracy":m_tree['val_acc'], "Val F1":m_tree['val_f1'],
         "Test Accuracy":m_tree['test_acc'], "Test F1":m_tree['test_f1']},
    ]).round(4)
    t1.to_csv("tables/Table1_Classification.csv", index=False)

    t2 = pd.DataFrame([
        {"Model":"Linear Regression", "Val MAE":mr_lin['val_mae'], "Val RMSE":mr_lin['val_rmse'],
         "Test MAE":mr_lin['test_mae'], "Test RMSE":mr_lin['test_rmse']},
        {"Model":"Decision Tree Regressor", "Val MAE":mr_tree['val_mae'], "Val RMSE":mr_tree['val_rmse'],
         "Test MAE":mr_tree['test_mae'], "Test RMSE":mr_tree['test_rmse']},
    ]).round(3)
    t2.to_csv("tables/Table2_Regression.csv", index=False)

    print("Saved: figures/[plot1..plot4].png and tables/Table1_Classification.csv, Table2_Regression.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/train.csv")   # run from src/
    args = parser.parse_args()
    main(args)
