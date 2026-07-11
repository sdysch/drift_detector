"""Model training utilities."""

import copy

import pandas as pd
from sklearn.pipeline import Pipeline

from drift_detector.models.models import objective_for_model
from drift_detector.models.pipeline import build_pipeline


def train_model(
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    objective_metric: str | None = None,
    target_transform: str | None = None,
) -> Pipeline:
    """Build and fit a preprocessing + model pipeline.

    Parameters
    ----------
    model_name : str
        Model key (``"linear"``, ``"random_forest"``, ``"xgboost"``,
        ``"ridge"``).
    params : dict
        Keyword arguments forwarded to the model constructor.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    numeric_features : list[str]
        Names of numeric columns for preprocessing.
    categorical_features : list[str]
        Names of categorical columns for preprocessing.
    objective_metric : str, optional
        If provided, the model's loss objective is set to match this eval
        metric (e.g. ``"mae"`` → absolute error for XGBoost).
    target_transform : str, optional
        Optional target transformation (e.g. ``"yeojohnson"``).

    Returns
    -------
    Pipeline
        Fitted scikit-learn pipeline with preprocessing and model steps.
    """
    model_params = copy.copy(params)
    obj = (
        objective_for_model(model_name, objective_metric) if objective_metric else None
    )
    if obj:
        model_params["objective"] = obj

    pipeline = build_pipeline(
        model_name,
        model_params,
        numeric_features,
        categorical_features,
        target_transform=target_transform,
    )

    pipeline.fit(X_train, y_train)

    return pipeline
