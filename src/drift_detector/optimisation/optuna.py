"""Optuna-based hyper-parameter optimisation with MLflow tracking."""

import logging
import time

import mlflow
import optuna
from sklearn.model_selection import cross_val_predict, KFold

from drift_detector.metrics import compute_metrics
from drift_detector.models.pipeline import build_pipeline
from drift_detector.optimisation.search_spaces import get_search_space
from drift_detector.tracking.mlflow import setup_mlflow

logger = logging.getLogger(__name__)


_DIRECTION = {"r2": "maximize"}


def _get_metric(config):
    """Return (metric_name, direction) from the optuna config."""
    metric = config["optuna"]["metric"]
    direction = _DIRECTION.get(metric, "minimize")
    return metric, direction


def create_objective(
    X_train,
    y_train,
    config,
    numeric_features,
    categorical_features,
):
    """Build an Optuna objective function for the configured model.

    Each trial creates a preprocessing + model pipeline, evaluates it via
    5-fold cross-validation, and logs parameters / metrics to MLflow
    under a model-specific experiment.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    config : dict
        Full training configuration.
    numeric_features : list[str]
        Names of numeric columns for preprocessing.
    categorical_features : list[str]
        Names of categorical columns for preprocessing.

    Returns
    -------
    callable
        ``objective(trial) -> float`` compatible with
        ``optuna.study.Study.optimize``.
    """
    metric, _ = _get_metric(config)
    model_name = config["model"]["name"]

    def objective(trial):
        params = get_search_space(
            model_name,
            trial,
            config["optuna"]["search_space"],
        )

        pipeline = build_pipeline(
            model_name=model_name,
            model_params=params,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        t0 = time.perf_counter()
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv)
        trial_time = time.perf_counter() - t0
        metrics = compute_metrics(y_train, y_pred)
        metrics["trial_time_s"] = round(trial_time, 3)

        setup_mlflow(
            tracking_uri=config.get("tracking", {}).get("uri", "mlruns"),
            experiment_name=f"{model_name}_experiments",
        )

        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_param("model", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        logger.info(
            "Trial %d | MAE: %.4f | RMSE: %.4f | R²: %.4f | %.1fs (params: %s)",
            trial.number,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            trial_time,
            params,
        )

        return metrics[metric]

    return objective


def run_optimisation(
    X_train,
    y_train,
    config,
    numeric_features,
    categorical_features,
):
    """Run an Optuna hyper-parameter search.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    config : dict
        Full training configuration (must include ``study`` and
        ``n_trials`` keys).
    numeric_features : list[str]
        Names of numeric columns for preprocessing.
    categorical_features : list[str]
        Names of categorical columns for preprocessing.

    Returns
    -------
    optuna.study.Study
        Completed study with optimisation results.
    """
    _, direction = _get_metric(config)

    study = optuna.create_study(
        direction=direction,
        study_name=config["study"]["name"],
        storage=config["study"].get("storage"),
        load_if_exists=True,
    )

    objective = create_objective(
        X_train,
        y_train,
        config,
        numeric_features,
        categorical_features,
    )

    study.optimize(
        objective,
        n_trials=config["optuna"]["n_trials"],
        n_jobs=config["optuna"].get("n_jobs", 1),
    )

    return study
