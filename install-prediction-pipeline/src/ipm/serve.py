"""Low-latency REST inference service (extra credit).

Loads the packaged ONNX graph once at start-up and serves single or batched
predictions. The same :mod:`ipm.featurizer` used in training runs here, so there
is zero train/serve skew in the feature logic.

Run:  uvicorn ipm.serve:app --host 0.0.0.0 --port 8080
      (artifacts directory via the IPM_ARTIFACTS env var, default ./artifacts)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import featurizer, model, schema

logger = logging.getLogger("ipm.serve")

ARTIFACTS_DIR = Path(os.environ.get("IPM_ARTIFACTS", "artifacts"))


class Impression(BaseModel):
    """One ad impression to score. Unknown/missing fields are allowed —
    the featurizer handles nulls and unseen categories gracefully."""

    user_id: str | None = None
    country: str | None = None
    device_os: str | None = None
    count_user_impressions_7: float | None = None
    appid: str | None = None
    sdkappid: str | None = None
    memory_total: float | None = None
    count_user_clicks_7: float | None = None
    session_count_7d: float | None = None
    user_install_profile: str | None = None
    timestamp: int | None = Field(default=None, description="utc epoch milliseconds")


class PredictRequest(BaseModel):
    instances: list[Impression]


class PredictResponse(BaseModel):
    predictions: list[float]
    latency_ms: float


class _Model:
    """Holds the ONNX session + companion config; loaded once per process."""

    def __init__(self, artifacts_dir: Path):
        onnx_path = artifacts_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No model.onnx in {artifacts_dir}. Train first (python -m ipm.train)."
            )
        # Single-thread sessions give the most predictable per-request latency
        # for online, one-row-at-a-time scoring.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        cfg_path = artifacts_dir / "featurizer_config.json"
        self.featurizer_config = (
            json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        )
        logger.info("Loaded model from %s", onnx_path)

    def predict(self, instances: list[Impression]) -> np.ndarray:
        raw = pd.DataFrame([i.model_dump() for i in instances])
        # Ensure every raw column the featurizer expects exists.
        for col in schema.ALL_COLUMNS:
            if col != schema.TARGET and col not in raw.columns:
                raw[col] = None
        feats = featurizer.transform(raw)
        return model.predict_proba_onnx(self.session, feats)


app = FastAPI(title="Install Prediction Model", version="0.1.0")
_state: dict[str, _Model] = {}


@app.on_event("startup")
def _startup() -> None:
    _state["model"] = _Model(ARTIFACTS_DIR)


@app.get("/health")
def health() -> dict:
    ready = "model" in _state
    return {"status": "ok" if ready else "loading", "artifacts": str(ARTIFACTS_DIR)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not request.instances:
        raise HTTPException(status_code=400, detail="no instances provided")
    start = time.perf_counter()
    probs = _state["model"].predict(request.instances)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return PredictResponse(predictions=[float(p) for p in probs], latency_ms=latency_ms)
