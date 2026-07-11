"""Command-line interface for drift_detector."""

import click
import optuna

from drift_detector.models.models import MODELS
from drift_detector.utils.logging import configure_logging
from drift_detector.workflow import run_best, run_optimise, run_train

MODEL_NAMES = sorted(MODELS)


@click.group()
def main():
    """drift_detector — train and optimise regression models."""
    configure_logging()


@main.command()
@click.option(
    "--model",
    "model_name",
    default="random_forest",
    type=click.Choice(MODEL_NAMES),
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
    type=click.Choice(MODEL_NAMES),
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


@main.command()
@click.option(
    "--model",
    "model_name",
    default="random_forest",
    type=click.Choice(MODEL_NAMES),
    help="Model to find the best run for.",
)
@click.option(
    "--metric",
    default="rmse",
    help="Metric to sort by.",
)
@click.option(
    "--direction",
    default="minimize",
    type=click.Choice(["minimize", "maximize"]),
    help="Whether lower or higher is better.",
)
@click.option(
    "--configs-dir",
    default="configs",
    type=click.Path(),
    help="Directory containing YAML config files.",
)
def best(model_name, metric, direction, configs_dir):
    """Find the best run for a model from MLflow."""
    run_best(configs_dir, model_name, metric, direction)


if __name__ == "__main__":
    main()
