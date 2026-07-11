"""Thin wrapper around the MLflow tracking API."""

import json
import logging
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


logger = logging.getLogger(__name__)

# Packages whose versions are logged for reproducibility.
_TRACKED_PACKAGES = [
    "mlflow",
    "scikit-learn",
    "xgboost",
    "optuna",
    "pandas",
    "numpy",
    "click",
]


def setup_mlflow(tracking_uri, experiment_name):
    """Configure the MLflow tracking URI and target experiment.

    Parameters
    ----------
    tracking_uri : str
        URI of the MLflow tracking server (or local path).
    experiment_name : str
        Name of the experiment to log runs into.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def start_run(run_name=None, nested=False):
    """Start an MLflow run.

    Parameters
    ----------
    run_name : str, optional
        Human-readable name for the run.
    nested : bool, default False
        If ``True``, open a nested run inside the current active run.

    Returns
    -------
    mlflow.entities.Run
        The started run object.
    """
    return mlflow.start_run(run_name=run_name, nested=nested)


def _run_cmd(args):
    """Run a shell command and return stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_git_info():
    """Log git commit hash and dirty state as MLflow parameters."""
    git_hash = _run_cmd(["git", "rev-parse", "HEAD"])
    if git_hash:
        mlflow.log_param("git_commit", git_hash)
    else:
        logger.warning("Failed to retrieve git commit hash")

    git_dirty = _run_cmd(["git", "status", "--porcelain"])
    mlflow.log_param("git_dirty", bool(git_dirty))


def log_package_versions():
    """Log versions of key dependencies as MLflow parameters."""
    for pkg in _TRACKED_PACKAGES:
        try:
            mlflow.log_param(f"version_{pkg}", version(pkg))
        except Exception:
            logger.warning("Could not retrieve version for %s", pkg)


def log_config(config):
    """Log the full merged config as a JSON artifact."""
    mlflow.log_text(json.dumps(config, indent=2), "config.json")


def log_features(numeric, categorical):
    """Log the feature lists used in the run."""
    mlflow.log_param("features_numeric", json.dumps(numeric))
    mlflow.log_param("features_categorical", json.dumps(categorical))
    mlflow.log_param("n_features", len(numeric) + len(categorical))


def log_data_shape(X_train, y_train):
    """Log training data dimensions."""
    mlflow.log_param("n_samples", len(X_train))
    mlflow.log_param("n_input_features", X_train.shape[1])


def log_model_params(params):
    """Log model hyperparameters."""
    for key, value in params.items():
        mlflow.log_param(f"param_{key}", value)


def save_model(model, path="models/model.pkl"):
    """Log a model to MLflow and persist it to disk.

    Parameters
    ----------
    model : object
        A fitted estimator that implements the scikit-learn API.
    path : str, default ``'models/model.pkl'``
        Local file path to save the serialised model.
    """
    mlflow.sklearn.log_model(model, "model")

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, dest)
    logger.info("Model saved to %s", dest)
