"""Orchestration layer for training and optimisation workflows."""

import logging
import time
from datetime import datetime, timezone

import mlflow

from drift_detector.data.load import load_training_data
from drift_detector.models.evaluate import run_evaluation
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
from drift_detector.utils.config import load_config as load_config_yaml

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load consolidated config from a single YAML file and set up MLflow tracking.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config = load_config_yaml(config_path)
    model_name = config.get("model", {}).get("name", "unknown")

    tracking = config.get("tracking", {})
    setup_mlflow(
        tracking_uri=tracking.get("uri", "mlruns"),
        experiment_name=f"{model_name}_experiments",
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


def run_train(config_path):
    """Train a model with config defaults and persist the result.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.
    """
    config = load_config(config_path)
    model_name = config["model"]["name"]
    X_train, y_train, numeric, categorical = _prepare_data(config)

    with start_run(run_name=f"{model_name}-train-{_timestamp()}"):
        log_git_info()
        log_package_versions()
        log_config(config)
        log_features(numeric, categorical)
        log_data_shape(X_train, y_train)
        log_model_params(config["params"])

        t0 = time.perf_counter()
        obj_metric = config.get("metric") or (config.get("optuna", {}) or {}).get(
            "metric"
        )
        pipeline = train_model(
            model_name=model_name,
            params=config["params"],
            X_train=X_train,
            y_train=y_train,
            numeric_features=numeric,
            categorical_features=categorical,
            objective_metric=obj_metric,
            target_transform=config.get("target_transform"),
        )
        train_time = time.perf_counter() - t0
        suffix = config.get("save_name_suffix")
        save_model(pipeline, register_name=model_name, name_suffix=suffix)

        y_pred = pipeline.predict(X_train)
        metrics = compute_metrics(y_train, y_pred)
        metrics["train_time_s"] = round(train_time, 3)
        log_metrics(metrics)
        logger.info("Training metrics: %s", metrics)


def run_optimise(config_path):
    """Run Optuna hyper-parameter search and log the best result.

    Each trial is logged to a ``{model_name}_experiments`` MLflow
    experiment. Study-level metadata (git, config, plots) is logged to
    a final summary run in the same experiment.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.
    """
    config = load_config(config_path)
    model_name = config["model"]["name"]
    X_train, y_train, numeric, categorical = _prepare_data(config)

    study = run_optimisation(
        X_train,
        y_train,
        config,
        numeric,
        categorical,
    )

    with start_run(run_name=f"{model_name}-summary-{_timestamp()}"):
        log_git_info()
        log_package_versions()
        log_config(config)
        log_features(numeric, categorical)
        log_data_shape(X_train, y_train)
        log_optuna_plots(study)
        log_metrics(study.best_params)
        logger.info("Best params: %s", study.best_params)


def run_best(config_path, metric="rmse", direction="minimize"):
    """Find the best run in a model's MLflow experiment.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.
    metric : str, default ``"rmse"``
        Metric to sort by.
    direction : str, default ``"minimize"``
        ``"minimize"`` for metrics where lower is better,
        ``"maximize"`` for metrics where higher is better.

    Returns
    -------
    tuple[dict, dict]
        ``(params, metrics)`` from the best run.
    """
    config = load_config(config_path)
    model_name = config["model"]["name"]

    order = f"metrics.{metric} {'ASC' if direction == 'minimize' else 'DESC'}"
    runs = mlflow.search_runs(
        experiment_names=[f"{model_name}_experiments"],
        order_by=[order],
        max_results=1,
    )

    if runs.empty:
        logger.warning("No runs found for %s", model_name)
        return {}, {}

    run = runs.iloc[0]

    params = {
        k.removeprefix("params."): v for k, v in run.items() if k.startswith("params.")
    }
    metrics = {
        k.removeprefix("metrics."): v
        for k, v in run.items()
        if k.startswith("metrics.")
    }

    logger.info("Best run: %s", run["run_id"])
    logger.info("Params: %s", params)
    logger.info("Metrics: %s", metrics)

    return params, metrics


def run_eval(model_path, data_path, target_column, output_dir="plots"):
    """Evaluate a fitted model against a test CSV.

    The fitted pipeline's preprocessor resolves features by column
    name, so the CSV must contain the training feature columns (order
    is independent).

    Parameters
    ----------
    model_path : str or Path
        Path to the fitted pipeline pickle file.
    data_path : str or Path
        Path to the evaluation CSV.
    target_column : str
        Name of the target column.
    output_dir : str or Path
        Root directory for evaluation outputs.
    """
    import pandas as pd

    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    run_evaluation(model_path, X=X, y=y, output_dir=output_dir)
