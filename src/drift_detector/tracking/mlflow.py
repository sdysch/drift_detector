"""Thin wrapper around the MLflow tracking API."""

import logging
from pathlib import Path

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


def log_config(config):
    """Log a nested config dict as flat ``section.key`` parameters.

    Parameters
    ----------
    config : dict[str, dict]
        Top-level keys are section names; nested key-value pairs are
        logged as ``section.key = value``.
    """
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                mlflow.log_param(f"{section}.{key}", value)


def log_metrics(metrics):
    """Log a dictionary of metric name-value pairs.

    Parameters
    ----------
    metrics : dict[str, float]
        Mapping of metric names to their numeric values.
    """
    for name, value in metrics.items():
        mlflow.log_metric(name, value)


def log_model(model, name="model"):
    """Persist a scikit-learn-compatible model to MLflow.

    Parameters
    ----------
    model : object
        A fitted estimator that implements the scikit-learn API.
    name : str, default ``'model''``
        Artifact path under which the model is stored.
    """
    mlflow.sklearn.log_model(model, name)


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

    import joblib

    joblib.dump(model, dest)
    logger.info("Model saved to %s", dest)
