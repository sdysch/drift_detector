"""Command-line interface for drift_detector."""

import json
import os

import click
import optuna
import uvicorn

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
    params, metrics = run_best(config, metric, direction)
    print(json.dumps({"params": params, "metrics": metrics}, indent=2))


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


@main.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to fitted .pkl pipeline. Defaults to models/xgboost_best_v1.pkl.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the server to.",
)
def serve(model_path, host, port):
    """Start the model serving API (FastAPI/Uvicorn)."""
    from drift_detector.api.server import app, _MODEL_PATH_ENV

    if model_path:
        os.environ[_MODEL_PATH_ENV] = str(model_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
