"""Command-line interface for drift_detector."""

import click
import optuna

from drift_detector.utils.logging import configure_logging
from drift_detector.workflow import run_best, run_eval, run_optimise, run_train


def _config_option():
    return click.option(
        "--config",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to the YAML config file.",
    )


@click.group()
def main():
    """drift_detector — train and optimise regression models."""
    configure_logging()


@main.command()
@_config_option()
def optimise(config):
    """Run an Optuna hyper-parameter search with MLflow tracking."""
    optuna.logging.set_verbosity(optuna.logging.INFO)
    configure_logging()
    run_optimise(config)


@main.command()
@_config_option()
def train(config):
    """Train a model using the hyper-parameters from the config."""
    run_train(config)


@main.command()
@_config_option()
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
def best(config, metric, direction):
    """Find the best run for a model from MLflow."""
    run_best(config, metric, direction)


@main.command()
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to fitted .pkl pipeline.",
)
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to evaluation CSV.",
)
@click.option(
    "--target",
    default="target",
    help="Name of the target column in the CSV.",
)
@click.option(
    "--output-dir",
    default="plots",
    type=click.Path(),
    help="Root directory for evaluation outputs.",
)
def eval(model, data, target, output_dir):
    """Evaluate a trained model against test data."""
    run_eval(model, data, target, output_dir)


if __name__ == "__main__":
    main()
