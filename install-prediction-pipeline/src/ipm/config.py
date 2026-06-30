"""Typed configuration loaded from a YAML file.

Using a dataclass keeps the config self-documenting and lets us validate values
once, at start-up, instead of scattering ``cfg.get("...", default)`` calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    # Path to the training data. Either a parquet file/dir or a csv file.
    path: str = "data/dataset.parquet"
    # Fraction of the most recent rows (by timestamp) used for validation/test.
    # The split is *temporal* to mimic production: train on the past, score the
    # future. This is the honest way to estimate online performance.
    valid_fraction: float = 0.15
    test_fraction: float = 0.15
    # Cap rows for quick local iterations (0 = use all rows).
    max_rows: int = 0


@dataclass
class ModelConfig:
    # LightGBM hyper-parameters. Deliberately modest — the assignment is about
    # engineering, not squeezing the last AUC point.
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 42
    # When True, scale_pos_weight = n_negative / n_positive to up-weight the rare
    # positives. Default False: a GBDT trained on the natural ~3% distribution
    # yields *calibrated* P(install), which the bidding system needs. Reweighting
    # improves nothing here and degrades calibration (see docs/DESIGN.md).
    balance_classes: bool = False


@dataclass
class TrainConfig:
    experiment_name: str = "install-prediction-model"
    run_name: str = "ipm-lightgbm"
    # Where the packaged artifacts are written.
    output_dir: str = "artifacts"
    onnx_opset: int = 17
    # Early stopping patience (rounds) on the validation set.
    early_stopping_rounds: int = 50


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return Config(
            data=DataConfig(**(raw.get("data") or {})),
            model=ModelConfig(**(raw.get("model") or {})),
            train=TrainConfig(**(raw.get("train") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
