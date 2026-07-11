"""Optuna search space definitions for each model."""


def random_forest_search_space(
    trial,
    config,
):
    """Suggest hyper-parameters for a Random Forest regressor.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial used to suggest values.
    config : dict
        Search-space configuration read from the YAML config.

    Returns
    -------
    dict
        Parameter dict accepted by ``RandomForestRegressor``.
    """
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            config["n_estimators"]["low"],
            config["n_estimators"]["high"],
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            config["max_depth"]["low"],
            config["max_depth"]["high"],
        ),
    }


def xgboost_search_space(
    trial,
    config,
):
    """Suggest hyper-parameters for an XGBoost regressor.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial used to suggest values.
    config : dict
        Search-space configuration read from the YAML config.

    Returns
    -------
    dict
        Parameter dict accepted by ``XGBRegressor``.
    """
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            config["n_estimators"]["low"],
            config["n_estimators"]["high"],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config["learning_rate"]["low"],
            config["learning_rate"]["high"],
            log=config["learning_rate"].get("log", False),
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            config["max_depth"]["low"],
            config["max_depth"]["high"],
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            config["min_child_weight"]["low"],
            config["min_child_weight"]["high"],
        ),
        "subsample": trial.suggest_float(
            "subsample",
            config["subsample"]["low"],
            config["subsample"]["high"],
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            config["colsample_bytree"]["low"],
            config["colsample_bytree"]["high"],
        ),
        "gamma": trial.suggest_float(
            "gamma",
            config["gamma"]["low"],
            config["gamma"]["high"],
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            config["reg_alpha"]["low"],
            config["reg_alpha"]["high"],
            log=config["reg_alpha"].get("log", False),
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            config["reg_lambda"]["low"],
            config["reg_lambda"]["high"],
            log=config["reg_lambda"].get("log", False),
        ),
    }


def ridge_search_space(
    trial,
    config,
):
    """Suggest hyper-parameters for a Ridge regressor.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial used to suggest values.
    config : dict
        Search-space configuration read from the YAML config.

    Returns
    -------
    dict
        Parameter dict accepted by ``Ridge``.
    """
    return {
        "alpha": trial.suggest_float(
            "alpha",
            config["alpha"]["low"],
            config["alpha"]["high"],
            log=config["alpha"].get(
                "log",
                False,
            ),
        ),
    }


SEARCH_SPACES = {
    "random_forest": random_forest_search_space,
    "xgboost": xgboost_search_space,
    "ridge": ridge_search_space,
}


def get_search_space(
    model_name,
    trial,
    config,
):
    """Return the search-space function for *model_name* and call it.

    Parameters
    ----------
    model_name : str
        Key into the ``SEARCH_SPACES`` registry.
    trial : optuna.trial.Trial
        Active Optuna trial.
    config : dict
        Search-space configuration from the YAML config.

    Returns
    -------
    dict
        Hyper-parameter suggestions for the requested model.
    """
    return SEARCH_SPACES[model_name](
        trial,
        config,
    )
