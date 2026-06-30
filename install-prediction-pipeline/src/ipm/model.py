"""Model assembly and ONNX packaging.

The serving graph is built in two co-operating pieces:

1. The stateless :mod:`ipm.featurizer` (Python) turns a raw row into the
   engineered feature columns.
2. A single sklearn ``Pipeline`` — ``ColumnTransformer`` (median-impute + scale
   numerics, one-hot encode categoricals) followed by a ``LGBMClassifier`` — is
   fitted on those columns and then exported **whole** to one ONNX graph.

Putting the imputer, scaler, one-hot vocabularies and the GBDT in a single ONNX
file means the serving runtime has exactly one artifact to load and there is no
train/serve skew in the learned preprocessing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import featurizer
from .config import ModelConfig


def build_pipeline(cfg: ModelConfig, scale_pos_weight: float) -> Pipeline:
    """Construct the (unfitted) preprocessing + LightGBM pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            # Numerics may still carry NaN (e.g. unknown clicks); the dedicated
            # *_missing flags already preserved the missingness signal, so a
            # median fill here is safe and ONNX-convertible.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # handle_unknown="ignore" -> categories unseen at training (a brand-new
    # appid/country at serving time) map to an all-zero vector instead of
    # crashing. This is the key robustness property for online serving.
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(featurizer.NUMERIC_FEATURES)),
            ("categorical", categorical_encoder, list(featurizer.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )

    classifier = LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        scale_pos_weight=scale_pos_weight if cfg.balance_classes else 1.0,
        n_jobs=-1,
        verbose=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """n_negative / n_positive — counters the ~3% install base-rate."""
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return float(n_neg / max(n_pos, 1))


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
def export_onnx(pipeline: Pipeline, opset: int, output_path: str | Path) -> Path:
    """Convert the fitted sklearn pipeline to a single ONNX graph.

    LightGBM is not part of sklearn, so we register the onnxmltools converter
    with skl2onnx before conversion. Each engineered feature becomes its own
    named ONNX input (float for numerics, string for categoricals), which gives
    the serving layer a clear, typed contract.
    """
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
        convert_lightgbm,
    )
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    from skl2onnx.common.shape_calculator import (
        calculate_linear_classifier_output_shapes,
    )

    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    initial_types = [
        (name, FloatTensorType([None, 1])) for name in featurizer.NUMERIC_FEATURES
    ] + [
        (name, StringTensorType([None, 1])) for name in featurizer.CATEGORICAL_FEATURES
    ]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        # zipmap=False -> probabilities come out as a plain [batch, 2] tensor
        # instead of a list-of-dicts, which is faster and simpler to serve.
        options={id(pipeline): {"zipmap": False}},
        # ai.onnx.ml must be pinned to 3; this skl2onnx build does not emit v5.
        target_opset={"": opset, "ai.onnx.ml": 3},
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())
    return output_path


def predict_proba_onnx(session, features: pd.DataFrame) -> np.ndarray:
    """Run the ONNX session and return P(install) for each row.

    ``session`` is an ``onnxruntime.InferenceSession``. Kept here so training,
    tests and the server all score identically.
    """
    feed = featurizer.split_onnx_inputs(features)
    outputs = session.run(None, feed)
    # Output 0 = predicted label, output 1 = [batch, 2] probabilities.
    proba = outputs[1]
    return np.asarray(proba)[:, 1]
