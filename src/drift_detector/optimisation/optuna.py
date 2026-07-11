"""Optuna-based hyper-parameter optimisation with MLflow tracking."""

import logging

import mlflow
import optuna

from sklearn.model_selection import cross_val_score

from drift_detector.models.pipeline import build_pipeline
from drift_detector.optimisation.search_spaces import get_search_space

logger = logging.getLogger(__name__)


def create_objective(
    X_train,
    y_train,
    config,
    numeric_features,
    categorical_features,
):
    """Build an Optuna objective function for the configured model.

    Each trial creates a preprocessing + model pipeline, evaluates it via
    5-fold cross-validation on RMSE, and logs parameters / metrics to
    MLflow.

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

    def objective(trial):
        model_name = config["model"]["name"]

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

        with mlflow.start_run(nested=True):
            mlflow.log_param(
                "model",
                model_name,
            )

            mlflow.log_params(params)

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring="neg_root_mean_squared_error",
            )

            rmse = -scores.mean()

            mlflow.log_metric(
                "rmse",
                rmse,
            )

        logger.info(
            "Trial %d finished with value: %.4f (params: %s)",
            trial.number,
            rmse,
            params,
        )

        return rmse

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
    study = optuna.create_study(
        direction="minimize",
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
        n_trials=config["n_trials"],
        n_jobs=config.get("n_jobs", 1),
    )

    return study
