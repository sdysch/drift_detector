"""YAML configuration loading utilities."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(path):
    """Load and parse a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(path)

    logger.info("Loading config from %s", path)

    with path.open("r") as f:
        return yaml.safe_load(f)


def merge_configs(*configs):
    """Merge multiple config dictionaries into one.

    Later values overwrite earlier ones for duplicate keys.

    Parameters
    ----------
    *configs : dict
        Configuration dictionaries to merge, in priority order.

    Returns
    -------
    dict
        Combined configuration dictionary.
    """
    merged = {}

    for config in configs:
        merged.update(config)

    return merged


def load_experiment_config(train_path, model_path, optuna_path=None):
    """Load and merge train, model, and optional Optuna configs.

    Parameters
    ----------
    train_path : str or Path
        Path to the training YAML config.
    model_path : str or Path
        Path to the model YAML config.
    optuna_path : str or Path, optional
        Path to the Optuna YAML config.  When ``None`` only train and
        model configs are merged.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    configs = [
        load_config(train_path),
        load_config(model_path),
    ]

    if optuna_path:
        configs.append(load_config(optuna_path))

    logger.info("Merging %d config files", len(configs))

    return merge_configs(*configs)
