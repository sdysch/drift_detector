"""YAML configuration loading and validation."""

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrackingConfig(BaseModel):
    uri: str = "mlruns"


class DataConfig(BaseModel):
    path: Path
    target: str


class FeaturesConfig(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    name: Literal["linear", "ridge", "random_forest", "xgboost"]


class StudyConfig(BaseModel):
    name: str
    storage: str | None = None


class OptunaConfig(BaseModel):
    metric: str
    n_trials: int = 100
    n_jobs: int = 1
    search_space: dict = Field(default_factory=dict)


class Config(BaseModel):
    """Top-level validated configuration."""

    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    data: DataConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig
    params: dict = Field(default_factory=dict)
    random_state: int = 42
    save_name_suffix: str | None = None
    study: StudyConfig | None = None
    optuna: OptunaConfig | None = None
    metric: str | None = None

    @property
    def resolved_metric(self) -> str:
        """Resolve metric with fallback: top-level → optuna → 'rmse'."""
        return self.metric or (self.optuna.metric if self.optuna else None) or "rmse"

    def model_dump_serializable(self) -> dict:
        """Dump to a plain dict suitable for JSON serialization."""
        return self.model_dump(mode="json")


def load_config(path):
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the YAML file.

    Returns
    -------
    Config
        Validated configuration object.
    """
    path = Path(path)

    logger.info("Loading config from %s", path)

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    return Config.model_validate(raw)
