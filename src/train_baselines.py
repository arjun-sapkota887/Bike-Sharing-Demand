
"""
Train classical baselines for both tasks (classification + regression)
using the *same logic* as the midpoint notebook, with MLflow tracking.

Models:
  - Classification:
      * Logistic Regression (with scaling)
      * Decision Tree Classifier (no scaling)
  - Regression:
      * Linear Regression (with scaling)
      * Decision Tree Regressor (no scaling)

This script also produces the 4 required plots via evaluate.py.
"""

import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from data import load_and_basic_clean, make_train_val_test_splits
from features import add_engineered_features, build_feature_matrices, THRESHOLD_PEAK
from evaluate import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_residuals,
)
from utils import set_seed, log_with_mlflow, SEED


def one_hot_encode_train_val_test(X_train, X_val, X_test, categorical_cols):
    """
    Fit OneHotEncoder on training categorical columns and transform
    train/val/test. Returns encoded matrices and the fitted encoder.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    X_train_cat = encoder.transform(X_train[categorical_cols])
    X_val_cat = encoder.transform(X_val[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])

    # numeric part = everything else
    X_train_num = X_train.drop(columns=categorical_cols)
    X_val_num = X_val.drop(columns=categorical_cols)
    X_test_num = X_test.drop(columns=categorical_cols)

    X_train_enc = np.hstack([X_train_num.values, X_train_cat])
    X_val_enc = np.hstack([X_val_num.values, X_val_cat])
    X_test_enc = np.hstack([X_test_num.values, X_test_cat])

    return (
        X_train_enc,
        X_val_enc,
        X_test_enc,
        encoder,
    )


def scale_train_val_test(X_train_enc, X_val_enc, X_test_enc):
    """
    Standardize features for models that need scaling (logistic + linear).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def main():
    # 1. Reproducibility
    set_seed(SEED)

    # 2. Load + basic cleaning (drop leakage, add time columns)
    df = load_and_basic_clean()

    # 3. Add engineered features + labels
    df = add_engineered_features(df)

    # 4. Build X, y_class, y_reg
    X, y_class, y_reg, feature_groups = build_feature_matrices(df)

    # 5. Train/Val/Test split (70/15/15, stratified on y_class)
    (
        X_train,
        X_val,
        X_test,
        y_class_train,
        y_class_val,
        y_class_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
    ) = make_train_val_test_splits(X, y_class, y_reg)

    # 6. One-hot encode categorical (season, weather)
    cat_cols = feature_groups["categorical"]
    (
        X_train_enc,
        X_val_enc,
        X_test_enc,
        encoder,
    ) = one_hot_encode_train_val_test(X_train, X_val, X_test, cat_cols)

    # 7. Scale encoded features for linear / logistic models
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    ) = scale_train_val_test(X_train_enc, X_val_enc, X_test_enc)

    # ------------------------------------------------------------------
    # 8. Baseline Models – Classification
    # ------------------------------------------------------------------

    # 8.1 Logistic Regression (with scaling)
    log_reg = LogisticRegression(max_iter=10000, solver="lbfgs", random_state=SEED)
    log_reg.fit(X_train_scaled, y_class_train)

    y_pred_val_log = log_reg.predict(X_val_scaled)
    y_pred_test_log = log_reg.predict(X_test_scaled)

    log_acc_val = accuracy_score(y_class_val, y_pred_val_log)
    log_f1_val = f1_score(y_class_val, y_pred_val_log)
    log_acc_test = accuracy_score(y_class_test, y_pred_test_log)
    log_f1_test = f1_score(y_class_test, y_pred_test_log)

    print("Logistic Regression – Val Accuracy:", log_acc_val)
    print("Logistic Regression – Val F1:", log_f1_val)
    print("Logistic Regression – Test Accuracy:", log_acc_test)
    print("Logistic Regression – Test F1:", log_f1_test)

    log_with_mlflow(
        "logistic_regression_baseline",
        log_reg,
        params={"model": "LogisticRegression", "scaled": True},
        metrics={
            "val_accuracy": log_acc_val,
            "val_f1": log_f1_val,
            "test_accuracy": log_acc_test,
            "test_f1": log_f1_test,
        },
    )

    # 8.2 Decision Tree Classifier (no scaling)
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=SEED)
    tree_clf.fit(X_train_enc, y_class_train)

    y_pred_val_tree = tree_clf.predict(X_val_enc)
    y_pred_test_tree = tree_clf.predict(X_test_enc)

    tree_acc_val = accuracy_score(y_class_val, y_pred_val_tree)
    tree_f1_val = f1_score(y_class_val, y_pred_val_tree)
    tree_acc_test = accuracy_score(y_class_test, y_pred_test_tree)
    tree_f1_test = f1_score(y_class_test, y_pred_test_tree)

    print("\nDecision Tree Classifier – Val Accuracy:", tree_acc_val)
    print("Decision Tree Classifier – Val F1:", tree_f1_val)
    print("Decision Tree Classifier – Test Accuracy:", tree_acc_test)
    print("Decision Tree Classifier – Test F1:", tree_f1_test)

    log_with_mlflow(
        "decision_tree_classifier_baseline",
        tree_clf,
        params={"model": "DecisionTreeClassifier", "max_depth": 5},
        metrics={
            "val_accuracy": tree_acc_val,
            "val_f1": tree_f1_val,
            "test_accuracy": tree_acc_test,
            "test_f1": tree_f1_test,
        },
    )

    # ------------------------------------------------------------------
    # 9. Baseline Models – Regression
    # ------------------------------------------------------------------

    # 9.1 Linear Regression (with scaling)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_reg_train)

    y_pred_val_lin = lin_reg.predict(X_val_scaled)
    y_pred_test_lin = lin_reg.predict(X_test_scaled)

    lin_mae_val = mean_absolute_error(y_reg_val, y_pred_val_lin)
    lin_rmse_val = np.sqrt(mean_squared_error(y_reg_val, y_pred_val_lin))
    lin_mae_test = mean_absolute_error(y_reg_test, y_pred_test_lin)
    lin_rmse_test = np.sqrt(mean_squared_error(y_reg_test, y_pred_test_lin))

    print("\nLinear Regression – Val MAE:", lin_mae_val, "Val RMSE:", lin_rmse_val)
    print("Linear Regression – Test MAE:", lin_mae_test, "Test RMSE:", lin_rmse_test)

    log_with_mlflow(
        "linear_regression_baseline",
        lin_reg,
        params={"model": "LinearRegression", "scaled": True},
        metrics={
            "val_mae": lin_mae_val,
            "val_rmse": lin_rmse_val,
            "test_mae": lin_mae_test,
            "test_rmse": lin_rmse_test,
        },
    )

    # 9.2 Decision Tree Regressor (no scaling)
    tree_reg = DecisionTreeRegressor(max_depth=8, random_state=SEED)
    tree_reg.fit(X_train_enc, y_reg_train)

    y_pred_val_tree_reg = tree_reg.predict(X_val_enc)
    y_pred_test_tree_reg = tree_reg.predict(X_test_enc)

    tree_mae_val = mean_absolute_error(y_reg_val, y_pred_val_tree_reg)
    tree_rmse_val = np.sqrt(mean_squared_error(y_reg_val, y_pred_val_tree_reg))
    tree_mae_test = mean_absolute_error(y_reg_test, y_pred_test_tree_reg)
    tree_rmse_test = np.sqrt(mean_squared_error(y_reg_test, y_pred_test_tree_reg))

    print("\nDecision Tree Regressor – Val MAE:", tree_mae_val, "Val RMSE:", tree_rmse_val)
    print("Decision Tree Regressor – Test MAE:", tree_mae_test, "Test RMSE:", tree_rmse_test)

    log_with_mlflow(
        "decision_tree_regressor_baseline",
        tree_reg,
        params={"model": "DecisionTreeRegressor", "max_depth": 8},
        metrics={
            "val_mae": tree_mae_val,
            "val_rmse": tree_rmse_val,
            "test_mae": tree_mae_test,
            "test_rmse": tree_rmse_test,
        },
    )

    # ------------------------------------------------------------------
    # 10. Plots (reproduce figures for midpoint)
    # ------------------------------------------------------------------

    # Plot 1: target distribution of is_peak_hour (classification label)
    plot_target_distribution(df["is_peak_hour"], threshold=THRESHOLD_PEAK)

    # Plot 2: correlation heatmap for key numeric features
    plot_correlation_heatmap(df)

    # Plot 3: confusion matrix for best classifier (Decision Tree)
    plot_confusion_matrix(tree_clf, X_test_enc, y_class_test, threshold=THRESHOLD_PEAK)

    # Plot 4: residuals vs predicted for best regressor (Decision Tree Regressor)
    plot_residuals(y_reg_test, y_pred_test_tree_reg)


if __name__ == "__main__":
    main()
