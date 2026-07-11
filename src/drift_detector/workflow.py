"""Orchestration layer for training and optimisation workflows."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from drift_detector.data.load import load_training_data
from drift_detector.metrics import compute_metrics
from drift_detector.models.train import train_model
from drift_detector.optimisation.optuna import run_optimisation
from drift_detector.tracking.mlflow import (
    log_config,
    log_data_shape,
    log_features,
    log_git_info,
    log_metrics,
    log_model_params,
    log_optuna_plots,
    log_package_versions,
    save_model,
    setup_mlflow,
    start_run,
)
from drift_detector.utils.config import load_experiment_config

logger = logging.getLogger(__name__)


def load_config(configs_dir, model_name, optimise=False):
    """Load merged experiment config and set up MLflow tracking.

    Parameters
    ----------
    configs_dir : str or Path
        Directory containing the YAML config files.
    model_name : str
        Model key used to locate ``configs/models/{model_name}.yml``.
    optimise : bool, default False
        When ``True``, the Optuna config is also loaded and merged.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    configs_dir = Path(configs_dir)
    model_config_path = configs_dir / "models" / f"{model_name}.yml"
    optuna_config_path = configs_dir / "optuna.yml" if optimise else None

    config = load_experiment_config(
        train_path=configs_dir / "train.yml",
        model_path=model_config_path,
        optuna_path=optuna_config_path,
    )

    tracking = config.get("tracking", {})
    setup_mlflow(
        tracking_uri=tracking.get("uri", "mlruns"),
        experiment_name=tracking.get("experiment_name", config["experiment"]["name"]),
    )

    return config


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _prepare_data(config):
    """Load training data and extract feature lists."""
    X_train, y_train = load_training_data(config)
    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]
    return X_train, y_train, numeric, categorical


def run_train(configs_dir, model_name):
    """Train a model with config defaults and persist the result.

    Parameters
    ----------
    configs_dir : str or Path
        Directory containing the YAML config files.
    model_name : str
        Model key (e.g. ``"random_forest"``).
    """
    config = load_config(configs_dir, model_name)
    X_train, y_train, numeric, categorical = _prepare_data(config)

    with start_run(run_name=f"{model_name}-train-{_timestamp()}"):
        log_git_info()
        log_package_versions()
        log_config(config)
        log_features(numeric, categorical)
        log_data_shape(X_train, y_train)
        log_model_params(config["params"])

        pipeline = train_model(
            model_name=model_name,
            params=config["params"],
            X_train=X_train,
            y_train=y_train,
            numeric_features=numeric,
            categorical_features=categorical,
        )
        save_model(pipeline)

        y_pred = pipeline.predict(X_train)
        metrics = compute_metrics(y_train, y_pred)
        log_metrics(metrics)
        logger.info("Training metrics: %s", metrics)


def run_optimise(configs_dir, model_name):
    """Run Optuna hyper-parameter search and log the best result.

    Parameters
    ----------
    configs_dir : str or Path
        Directory containing the YAML config files.
    model_name : str
        Model key (e.g. ``"random_forest"``).
    """
    config = load_config(configs_dir, model_name, optimise=True)
    X_train, y_train, numeric, categorical = _prepare_data(config)

    with start_run(run_name=f"{model_name}-optimise-{_timestamp()}"):
        log_git_info()
        log_package_versions()
        log_config(config)
        log_features(numeric, categorical)
        log_data_shape(X_train, y_train)

        study = run_optimisation(
            X_train,
            y_train,
            config,
            numeric,
            categorical,
        )
        log_optuna_plots(study)
        logger.info("Best params: %s", study.best_params)
