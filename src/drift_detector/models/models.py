"""Model factory for regression baselines."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from xgboost import XGBRegressor


def build_model(name: str, params: dict):
    """Instantiate a regression model by name.

    Args:
        name: One of ``"linear"``, ``"random_forest"``, ``"xgboost"``,
            ``"ridge"``.
        params: Keyword arguments forwarded to the model constructor.

    Returns:
        An unfitted scikit-learn-compatible regressor.
    """
    models = {
        "linear": LinearRegression,
        "random_forest": RandomForestRegressor,
        "xgboost": XGBRegressor,
        "ridge": Ridge,
    }

    return models[name](**params)
