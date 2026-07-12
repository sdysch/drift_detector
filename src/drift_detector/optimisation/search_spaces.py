"""Optuna search space definitions for each model."""

from typing import Literal

from pydantic import BaseModel, RootModel, model_validator


class SearchSpaceEntry(BaseModel):
    """A single hyper-parameter search-space range."""

    type: Literal["int", "float"]
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self):
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        return self

    @model_validator(mode="after")
    def _validate_log_with_int(self):
        if self.log and self.type == "int":
            raise ValueError("log scale is not supported for integer parameters")
        return self


class SearchSpaceConfig(RootModel):
    """Validated search-space configuration for a model.

    Accepts the raw ``search_space`` dict from the YAML config, where
    each key is a hyper-parameter name mapping to its range definition.
    """

    root: dict[str, SearchSpaceEntry]


def _suggest_param(
    trial,
    entry: SearchSpaceEntry,
    name: str,
):
    """Suggest a value for a single hyper-parameter."""
    if entry.type == "int":
        return trial.suggest_int(name, int(entry.low), int(entry.high))
    return trial.suggest_float(name, entry.low, entry.high, log=entry.log)


def _build_params(trial, space: SearchSpaceConfig, names: list[str]):
    """Build a param dict by suggesting each named parameter."""
    return {name: _suggest_param(trial, space.root[name], name) for name in names}


def random_forest_search_space(trial, space: SearchSpaceConfig):
    """Suggest hyper-parameters for a Random Forest regressor."""
    return _build_params(trial, space, ["n_estimators", "max_depth"])


def xgboost_search_space(trial, space: SearchSpaceConfig):
    """Suggest hyper-parameters for an XGBoost regressor."""
    return _build_params(
        trial,
        space,
        [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ],
    )


def ridge_search_space(trial, space: SearchSpaceConfig):
    """Suggest hyper-parameters for a Ridge regressor."""
    return _build_params(trial, space, ["alpha"])


SEARCH_SPACES = {
    "random_forest": random_forest_search_space,
    "xgboost": xgboost_search_space,
    "ridge": ridge_search_space,
}


def get_search_space(model_name, trial, config):
    """Validate the search-space config and suggest parameters.

    Parameters
    ----------
    model_name : str
        Key into the ``SEARCH_SPACES`` registry.
    trial : optuna.trial.Trial
        Active Optuna trial.
    config : dict
        Raw search-space configuration from the YAML config.

    Returns
    -------
    dict
        Hyper-parameter suggestions for the requested model.
    """
    space = SearchSpaceConfig.model_validate(config)
    return SEARCH_SPACES[model_name](trial, space)
