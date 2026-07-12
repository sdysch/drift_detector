"""Data loading utilities for training and evaluation."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_training_data(config):
    """Load training features and target from the configured CSV path.

    Parameters
    ----------
    config : Config
        Validated configuration containing ``data.path``, ``data.target``,
        ``features.numeric``, and ``features.categorical``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X_train, y_train)`` ready for model training.
    """
    path = config.data.path
    target = config.data.target

    numeric = config.features.numeric
    categorical = config.features.categorical

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
