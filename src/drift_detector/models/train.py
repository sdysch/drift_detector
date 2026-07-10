"""Model training utilities."""

import pandas as pd
from sklearn.pipeline import Pipeline

from drift_detector.models.pipeline import build_pipeline


def train_model(
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
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

    Returns
    -------
    Pipeline
        Fitted scikit-learn pipeline with preprocessing and model steps.
    """
    pipeline = build_pipeline(
        model_name,
        params,
        numeric_features,
        categorical_features,
    )

    pipeline.fit(X_train, y_train)

    return pipeline
