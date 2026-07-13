"""FastAPI server for serving a trained regression pipeline."""

from __future__ import annotations

from pathlib import Path

import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from drift_detector.utils.logging import configure_logging

_MODEL_PATH_ENV = "DRIFT_DETECTOR_MODEL_PATH"
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "models" / "xgboost_best_v1.pkl"
)

app = FastAPI(
    title="Drift Detector",
    description="Serving API for trained regression models",
    version="0.1.0",
)

_pipeline = None
_model_path = None


class Features(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_gaussian: float
    feature_lognormal: float
    feature_exponential: float
    category: str
    type: str


class PredictResponse(BaseModel):
    predictions: list[float]


def load_pipeline(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def _resolve_model_path(override: str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get(_MODEL_PATH_ENV)
    if env:
        return Path(env)
    default = _DEFAULT_MODEL_PATH
    if default.exists():
        return default
    alt = Path.cwd() / "models" / "xgboost_best_v1.pkl"
    if alt.exists():
        return alt
    return default


@app.on_event("startup")
def startup():
    global _pipeline, _model_path
    configure_logging()
    _model_path = _resolve_model_path()
    _pipeline = load_pipeline(_model_path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": {
            "loaded": _pipeline is not None,
            "path": str(_model_path) if _model_path else None,
            "pipeline_steps": list(_pipeline.named_steps) if _pipeline else None,
        },
        "features": {
            "names": list(Features.model_fields),
            "schema": Features.model_json_schema(),
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(features: Features) -> PredictResponse:
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")

    X = pd.DataFrame([features.model_dump()])
    y_pred = _pipeline.predict(X)
    return PredictResponse(predictions=y_pred.tolist())
