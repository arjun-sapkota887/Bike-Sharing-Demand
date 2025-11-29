
"""
Shared utilities: random seed handling and MLflow logging helper.
This mirrors the helper used in the midpoint notebook.
"""

import os
import random
import numpy as np
import mlflow
import mlflow.sklearn


SEED = 42
EXPERIMENT_NAME = "BikeSharing_midpoint"


def set_seed(seed: int = SEED) -> None:
    """Set numpy and python random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def ensure_experiment(name: str = EXPERIMENT_NAME) -> None:
    """Create / select MLflow experiment."""
    mlflow.set_experiment(name)


def log_with_mlflow(
    run_name: str,
    model,
    params: dict | None = None,
    metrics: dict | None = None,
    experiment_name: str = EXPERIMENT_NAME,
    model_artifact_name: str = "model",
) -> None:
    """
    Minimal MLflow wrapper similar to what you used in the notebook.

    - Creates/selects an experiment.
    - Starts a run with `run_name`.
    - Logs params, metrics, and the sklearn model.
    """
    ensure_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

        # This will emit a deprecation warning in MLflow 3.x,
        # but is perfectly fine for this project.
        mlflow.sklearn.log_model(model, artifact_path=model_artifact_name)
