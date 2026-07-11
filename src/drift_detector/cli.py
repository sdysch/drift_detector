"""Command-line interface for drift_detector."""

import click
import optuna

from drift_detector.utils.logging import configure_logging
from drift_detector.workflow import run_optimise, run_train

MODELS = ["linear", "random_forest", "xgboost", "ridge"]


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
    """Run an Optuna hyper-parameter search with MLflow tracking."""
    optuna.logging.set_verbosity(optuna.logging.INFO)
    configure_logging()
    run_optimise(configs_dir, model_name)


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
    """Train a model using the default hyper-parameters from the config."""
    run_train(configs_dir, model_name)


if __name__ == "__main__":
    main()
