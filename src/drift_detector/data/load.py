"""Data loading utilities for training and evaluation."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_training_data(config):
    """Load training features and target from the configured CSV path.

    Parameters
    ----------
    config : dict
        Merged configuration containing ``data.path``, ``data.target``,
        ``features.numeric``, and ``features.categorical``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X_train, y_train)`` ready for model training.
    """
    path = config["data"]["path"]
    target = config["data"]["target"]

    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]

    logger.info("Loading training data from %s", path)

    df = pd.read_csv(path)

    feature_cols = numeric + categorical
    X_train = df[feature_cols]
    y_train = df[target]

    logger.info(
        "Loaded %d rows, %d features, target='%s'",
        len(X_train),
        len(feature_cols),
        target,
    )

    return X_train, y_train


def load_eval_data(config, data_path=None):
    """Load evaluation features and target.

    Parameters
    ----------
    config : dict
        Configuration containing ``data.target``, ``features.numeric``,
        and ``features.categorical``.
    data_path : str, optional
        Path to the CSV file. Falls back to ``config.data.path``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` ready for evaluation.
    """
    path = data_path or config["data"]["path"]
    target = config["data"]["target"]

    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]

    logger.info("Loading evaluation data from %s", path)

    df = pd.read_csv(path)

    feature_cols = numeric + categorical
    X = df[feature_cols]
    y = df[target]

    logger.info(
        "Loaded %d rows, %d features, target='%s'",
        len(X),
        len(feature_cols),
        target,
    )

    return X, y
