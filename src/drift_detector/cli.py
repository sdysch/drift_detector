"""Command-line interface for drift_detector."""

import logging
from pathlib import Path

import click

from drift_detector.data.load import load_training_data
from drift_detector.models.train import train_model
from drift_detector.optimisation.optuna import run_optimisation
from drift_detector.tracking.mlflow import (
    save_model,
    setup_mlflow,
    start_run,
)
from drift_detector.utils.config import load_experiment_config
from drift_detector.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    "model_name",
    default="random_forest",
    type=click.Choice(["linear", "random_forest", "xgboost", "ridge"]),
    help="Model to train or optimise.",
)
@click.option(
    "--configs-dir",
    default="configs",
    type=click.Path(),
    help="Directory containing YAML config files.",
)
@click.option("--optimise", is_flag=True, help="Run Optuna hyper-parameter search.")
@click.option("--train", "do_train", is_flag=True, help="Train a final model.")
def main(model_name, configs_dir, optimise, do_train):
    """drift_detector — train and optimise regression models."""
    configure_logging()

    if not optimise and not do_train:
        click.echo("Nothing to do.  Pass --optimise or --train (or both).")
        raise SystemExit(1)

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

    X_train, y_train = load_training_data(config)

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    best_params = config["params"]

    if optimise:
        with start_run(run_name=f"{model_name}-optimise"):
            study = run_optimisation(
                X_train,
                y_train,
                config,
                numeric_features,
                categorical_features,
            )
            best_params = study.best_params
            logger.info("Best params: %s", best_params)

    if do_train:
        with start_run(run_name=f"{model_name}-train"):
            pipeline = train_model(
                model_name=model_name,
                params=best_params,
                X_train=X_train,
                y_train=y_train,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            save_model(pipeline)
            logger.info("Training complete.")


if __name__ == "__main__":
    main()
