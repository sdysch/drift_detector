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
