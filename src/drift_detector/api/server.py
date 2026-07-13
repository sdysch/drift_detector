"""FastAPI server for serving a trained regression pipeline."""

from __future__ import annotations

from pathlib import Path

import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from starlette.responses import JSONResponse

from drift_detector.api.store import PredictionStore
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
_store: PredictionStore | None = None


_FEATURE_NAMES = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_gaussian",
    "feature_lognormal",
    "feature_exponential",
    "category",
    "type",
]


class Features(BaseModel):
    model_config = {"extra": "forbid"}

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


@app.exception_handler(ValidationError)
def _validation_handler(request, exc: ValidationError):
    unknown = [e for e in exc.errors() if e["type"] == "extra_forbidden"]
    missing = [e for e in exc.errors() if e["type"] == "missing"]

    messages = []
    if unknown:
        names = [e["loc"][-1] for e in unknown]
        messages.append(
            f"Unrecognised feature(s): {', '.join(names)}. "
            f"Valid features: {', '.join(_FEATURE_NAMES)}."
        )
    for e in missing:
        messages.append(f"Missing required feature: '{e['loc'][-1]}'.")

    others = [e for e in exc.errors() if e not in unknown and e not in missing]
    for e in others:
        field = ".".join(str(p) for p in e["loc"])
        messages.append(f"Invalid value for '{field}': {e['msg']}.")

    return JSONResponse(
        status_code=422,
        content={"detail": messages},
    )


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
    global _pipeline, _model_path, _store
    configure_logging()
    _model_path = _resolve_model_path()
    _pipeline = load_pipeline(_model_path)
    _store = PredictionStore()


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
    prediction = float(y_pred[0])

    if _store is not None:
        _store.insert(features.model_dump(), prediction)

    return PredictResponse(predictions=[prediction])
