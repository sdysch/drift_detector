"""Thin wrapper around the MLflow tracking API."""

import logging
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


logger = logging.getLogger(__name__)


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


def log_source_versions():
    """Log the current git commit hash as an MLflow parameter."""
    git_hash = _run_cmd(["git", "rev-parse", "HEAD"])
    if git_hash:
        mlflow.log_param("git_commit", git_hash)
    else:
        logger.warning("Failed to retrieve git commit hash")


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
