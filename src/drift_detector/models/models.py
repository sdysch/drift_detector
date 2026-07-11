"""Model registry for regression baselines."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from xgboost import XGBRegressor


_METRIC_TO_XGBOOST_OBJECTIVE = {
    "mae": "reg:absoluteerror",
    "rmse": "reg:squarederror",
    "r2": "reg:squarederror",
    "mape": "reg:absoluteerror",
    "wmape": "reg:absoluteerror",
    "smape": "reg:absoluteerror",
}


def objective_for_model(model_name: str, metric: str) -> str | None:
    """Return an objective/loss string for the given model and eval metric.

    Only XGBoost supports custom objectives. Other models (linear, ridge,
    random forest) ignore this.
    """
    if model_name == "xgboost":
        return _METRIC_TO_XGBOOST_OBJECTIVE.get(metric)
    return None


@dataclass(frozen=True)
class ModelSpec:
    """Immutable descriptor for a registered model."""

    name: str
    constructor: type


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec("linear", LinearRegression),
        ModelSpec("random_forest", RandomForestRegressor),
        ModelSpec("xgboost", XGBRegressor),
        ModelSpec("ridge", Ridge),
    ]
}


def build_model(name: str, params: dict):
    """Instantiate a regression model by name.

    Args:
        name: One of the registered model names.
        params: Keyword arguments forwarded to the model constructor.

    Returns:
        An unfitted scikit-learn-compatible regressor.
    """
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {', '.join(sorted(MODELS))}"
        )

    return MODELS[name].constructor(**params)
