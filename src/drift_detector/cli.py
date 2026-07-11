"""Command-line interface for drift_detector."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import optuna

from drift_detector.data.load import load_training_data
from drift_detector.models.train import train_model
from drift_detector.optimisation.optuna import run_optimisation
from drift_detector.tracking.mlflow import (
    log_source_versions,
    save_model,
    setup_mlflow,
    start_run,
)
from drift_detector.utils.config import load_experiment_config
from drift_detector.utils.logging import configure_logging

logger = logging.getLogger(__name__)

MODELS = ["linear", "random_forest", "xgboost", "ridge"]


def _load_config(configs_dir, model_name, optimise=False):
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


@click.group()
def main():
    """drift_detector — train and optimise regression models."""
    configure_logging()


@main.command()
@click.option(
    "--model",
    "model_name",
    default="random_forest",
    type=click.Choice(MODELS),
    help="Model to optimise.",
)
@click.option(
    "--configs-dir",
    default="configs",
    type=click.Path(),
    help="Directory containing YAML config files.",
)
def optimise(model_name, configs_dir):
    """Run an Optuna hyper-parameter search with MLflow tracking.

    Reads search-space bounds from the model config and the Optuna
    config, runs the configured number of trials, and logs the best
    parameters found.
    """
    optuna.logging.set_verbosity(optuna.logging.INFO)
    configure_logging()

    config = _load_config(configs_dir, model_name, optimise=True)
    X_train, y_train = load_training_data(config)

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    with start_run(run_name=f"{model_name}-optimise-{timestamp}"):
        log_source_versions()
        study = run_optimisation(
            X_train,
            y_train,
            config,
            numeric_features,
            categorical_features,
        )
        logger.info("Best params: %s", study.best_params)


@main.command()
@click.option(
    "--model",
    "model_name",
    default="random_forest",
    type=click.Choice(MODELS),
    help="Model to train.",
)
@click.option(
    "--configs-dir",
    default="configs",
    type=click.Path(),
    help="Directory containing YAML config files.",
)
def train(model_name, configs_dir):
    """Train a model using the default hyper-parameters from the config.

    Builds a preprocessing + model pipeline, fits it on the training
    data, and persists the result to MLflow and disk.
    """
    config = _load_config(configs_dir, model_name)
    X_train, y_train = load_training_data(config)

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    with start_run(run_name=f"{model_name}-train-{timestamp}"):
        log_source_versions()
        pipeline = train_model(
            model_name=model_name,
            params=config["params"],
            X_train=X_train,
            y_train=y_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        save_model(pipeline)
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
