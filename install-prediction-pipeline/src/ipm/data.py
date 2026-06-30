"""Robust data loading and the temporal train/valid/test split.

Design notes
------------
* Format-agnostic: the same code reads the 1.5 GB ``dataset.parquet`` and the
  small anonymised CSV used for tests, chosen by file extension.
* Column projection: only the columns in ``schema.ALL_COLUMNS`` are read from
  parquet, which avoids materialising columns we never use.
* Temporal split: rows are ordered by ``timestamp`` and the most recent slice is
  held out. Predicting the future from the past is exactly the production
  setting, and it prevents look-ahead leakage that a random split would hide.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

logger = logging.getLogger(__name__)


def load_raw(path: str | Path, max_rows: int = 0) -> pd.DataFrame:
    """Load the raw impressions table from parquet or csv.

    Parameters
    ----------
    path : str | Path
        A ``.parquet`` file/directory or a ``.csv`` file.
    max_rows : int
        If > 0, keep only the first ``max_rows`` rows (after time sorting) for
        fast local iteration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Point data.path at dataset.parquet "
            f"or generate a synthetic sample with scripts/generate_synthetic_data.py."
        )

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"} or path.is_dir():
        # Push the column projection down to the parquet reader so we never
        # decode columns we do not need.
        try:
            df = pd.read_parquet(path, columns=schema.ALL_COLUMNS)
        except (ValueError, KeyError):
            # Fall back to reading everything if the projection fails (e.g. a
            # column is missing in this particular snapshot).
            df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}' for '{path}'.")

    df = _coerce_types(df)
    df = df.sort_values(schema.TIMESTAMP, kind="stable").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Make column dtypes deterministic and robust to source quirks."""
    # Target -> {0,1}. Accept bool, "True"/"False" strings, or 0/1.
    if df[schema.TARGET].dtype == bool:
        df[schema.TARGET] = df[schema.TARGET].astype("int8")
    else:
        df[schema.TARGET] = (
            df[schema.TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "1": 1, "false": 0, "0": 0})
            .fillna(0)
            .astype("int8")
        )

    # Numerics -> float; non-parseable values become NaN and are imputed later.
    for col in [
        schema.COUNT_IMPRESSIONS_7,
        schema.COUNT_CLICKS_7,
        schema.SESSION_COUNT_7D,
        schema.MEMORY_TOTAL,
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[schema.TIMESTAMP] = pd.to_numeric(df[schema.TIMESTAMP], errors="coerce")
    # Drop rows with an unusable timestamp — we cannot place them in time.
    df = df[df[schema.TIMESTAMP].notna()].copy()

    # String columns -> pandas string dtype (keeps NaN distinct from "").
    for col in schema.RAW_STRING_COLUMNS:
        df[col] = df[col].astype("string")
    return df


def temporal_split(
    df: pd.DataFrame, valid_fraction: float, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a *time-sorted* frame into train / valid / test by recency.

    The last ``test_fraction`` of rows is the test set, the slice before it is
    validation, and everything earlier is training.
    """
    if not 0 <= valid_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if valid_fraction + test_fraction >= 1:
        raise ValueError("valid_fraction + test_fraction must be < 1")

    n = len(df)
    n_test = int(round(n * test_fraction))
    n_valid = int(round(n * valid_fraction))
    n_train = n - n_valid - n_test
    if min(n_train, n_valid, n_test) <= 0:
        raise ValueError(
            f"Split produced an empty partition (n={n}). Lower the fractions "
            f"or provide more data."
        )

    train = df.iloc[:n_train].reset_index(drop=True)
    valid = df.iloc[n_train : n_train + n_valid].reset_index(drop=True)
    test = df.iloc[n_train + n_valid :].reset_index(drop=True)
    logger.info(
        "Temporal split -> train=%d valid=%d test=%d (install rate %.4f/%.4f/%.4f)",
        len(train), len(valid), len(test),
        train[schema.TARGET].mean(), valid[schema.TARGET].mean(), test[schema.TARGET].mean(),
    )
    return train, valid, test
