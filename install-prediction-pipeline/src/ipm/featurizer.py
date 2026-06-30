"""Deterministic feature engineering (raw row -> model-ready columns).

Why a separate, *stateless* featurizer instead of doing everything inside the
sklearn pipeline?

* ONNX's ``ai.onnx.ml`` operators cannot split strings, parse timestamps or
  count tokens in a sequence. Those operations have to happen in Python.
* By making this layer **stateless** (it learns nothing from the data — only
  fixed arithmetic and string ops) we guarantee that the exact same code path
  runs at training time and at serving time. There is no fitted state that could
  silently drift between the two.

Everything that *is* learned (medians for imputation, scaler statistics,
one-hot vocabularies, the GBDT itself) lives in the sklearn pipeline and is
therefore captured inside the single ONNX graph.

Insights from the data analysis that shaped these features
----------------------------------------------------------
* Install base-rate is ~3.25% -> heavy imbalance (handled in the model).
* Missingness is informative: rows where ``count_user_clicks_7`` is null convert
  at 1.5% vs 3.5% overall, so we emit explicit ``*_missing`` flags rather than
  silently imputing the signal away.
* ``memory_total`` is in bytes (1.8-17 GB) -> rescaled to GB.
* Count features are heavy-tailed (p99 ~ 30, max in the hundreds) -> log1p.
* ``user_install_profile`` is a space-delimited app list -> summarised by its
  length and a presence flag (a cheap, robust proxy for user engagement; richer
  embeddings are noted as future work in the design doc).
* ``timestamp`` -> cyclical hour-of-day / day-of-week, never used as a raw
  magnitude (which would not generalise to future dates).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema

# Output feature groups. Order matters: it defines the ONNX input order and the
# serving contract, so it is declared once here and reused everywhere.
NUMERIC_FEATURES: list[str] = [
    "memory_gb",
    "log_impressions_7",
    "log_clicks_7",
    "log_sessions_7d",
    "ctr_7",                 # clicks / (impressions + 1)
    "profile_len",           # number of previously installed apps
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Informative-missingness flags (1.0 = the source value was missing).
    "clicks_missing",
    "memory_missing",
    "sessions_missing",
    "profile_missing",
]

CATEGORICAL_FEATURES: list[str] = [
    schema.COUNTRY,
    schema.DEVICE_OS,
    schema.APPID,
    schema.SDKAPPID,
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class FeaturizerConfig:
    """Serialisable description of the featurizer's output contract.

    The featurizer itself has no learned parameters; this object exists so the
    serving layer can validate that it is paired with a compatible model and so
    the feature contract is versioned alongside the artifacts.
    """

    version: str = "1.0"
    numeric_features: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
        }


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw impressions frame to the engineered feature frame.

    Returns a DataFrame with exactly ``ALL_FEATURES`` columns: ``float32`` for
    numerics (NaN preserved for the in-pipeline imputer) and ``object``/string
    for categoricals (missing values replaced by a fixed sentinel).
    """
    out = pd.DataFrame(index=df.index)

    # --- Numeric, with informative-missingness flags ------------------------
    memory = pd.to_numeric(df.get(schema.MEMORY_TOTAL), errors="coerce")
    out["memory_gb"] = memory / 1e9
    out["memory_missing"] = memory.isna().astype("float32")

    impressions = pd.to_numeric(df.get(schema.COUNT_IMPRESSIONS_7), errors="coerce")
    out["log_impressions_7"] = np.log1p(impressions.clip(lower=0))

    clicks = pd.to_numeric(df.get(schema.COUNT_CLICKS_7), errors="coerce")
    out["log_clicks_7"] = np.log1p(clicks.clip(lower=0))
    out["clicks_missing"] = clicks.isna().astype("float32")

    sessions = pd.to_numeric(df.get(schema.SESSION_COUNT_7D), errors="coerce")
    out["log_sessions_7d"] = np.log1p(sessions.clip(lower=0))
    out["sessions_missing"] = sessions.isna().astype("float32")

    # Click-through rate over the 7-day window. Missing clicks are treated as 0
    # engagement here (the dedicated flag preserves the "unknown" information).
    out["ctr_7"] = clicks.fillna(0) / (impressions.fillna(0) + 1.0)

    # --- Sequence feature: previously installed apps ------------------------
    profile = df.get(schema.USER_INSTALL_PROFILE)
    profile_len = profile.apply(_token_count) if profile is not None else 0
    out["profile_len"] = pd.to_numeric(profile_len, errors="coerce").fillna(0.0)
    out["profile_missing"] = (
        profile.isna().astype("float32") if profile is not None else np.float32(1.0)
    )

    # --- Time: cyclical encodings (no raw magnitude) ------------------------
    ts = pd.to_datetime(
        pd.to_numeric(df.get(schema.TIMESTAMP), errors="coerce"),
        unit="ms", utc=True,
    )
    hour = ts.dt.hour.fillna(0).to_numpy()
    dow = ts.dt.dayofweek.fillna(0).to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    for col in NUMERIC_FEATURES:
        out[col] = out[col].astype("float32")

    # --- Categorical: fixed sentinel for missing, stable string dtype -------
    for col in CATEGORICAL_FEATURES:
        series = df.get(col)
        if series is None:
            series = pd.Series([pd.NA] * len(df), index=df.index)
        out[col] = (
            series.astype("string").fillna(schema.CATEGORICAL_MISSING).astype(object)
        )

    return out[ALL_FEATURES]


def _token_count(value) -> float:
    """Count space-delimited tokens in a sequence string; NaN-safe."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return 0.0
    return float(len(str(value).split()))


def split_onnx_inputs(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reshape the feature frame into the per-column feed ONNX expects.

    Each model input is a single column ``[batch, 1]`` tensor: ``float32`` for
    numerics, ``object`` (python str) for categoricals.
    """
    feed: dict[str, np.ndarray] = {}
    for col in NUMERIC_FEATURES:
        feed[col] = features[col].to_numpy(dtype=np.float32).reshape(-1, 1)
    for col in CATEGORICAL_FEATURES:
        feed[col] = features[col].to_numpy(dtype=object).reshape(-1, 1)
    return feed
