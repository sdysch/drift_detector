"""Common preprocessing and modelling pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from drift_detector.models.models import build_model


def create_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a column transformer that imputes and one-hot encodes categorical features.

    TODO, what if I want to specify the type of encoding? Impact, ordinal etc

    Args:
        numeric_features: Column names to treat as numeric.
        categorical_features: Column names to treat as categorical.

    Returns:
        A fitted-ready ``ColumnTransformer``.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("encoder", OneHotEncoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_pipeline(
    model_name: str,
    model_params: dict,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Create a full preprocessing + modelling pipeline.

    Combines :func:`create_preprocessor` with a model from
    :func:`drift_detector.models.models.build_model` into a single
    scikit-learn ``Pipeline`` that can be fit and predicted on raw DataFrames.

    Args:
        model_name: Model key (``"linear"``, ``"random_forest"``, ``"xgboost"``).
        model_params: Keyword arguments forwarded to the model constructor.
        numeric_features: Column names to treat as numeric.
        categorical_features: Column names to treat as categorical.

    Returns:
        A ``Pipeline`` with a ``"preprocessor"`` and ``"model"`` step.
    """
    return Pipeline(
        [
            (
                "preprocessor",
                create_preprocessor(numeric_features, categorical_features),
            ),
            ("model", build_model(model_name, model_params)),
        ]
    )
