"""End-to-end training entrypoint.

    load -> temporal split -> featurize -> fit (early stopping) -> evaluate
         -> export ONNX -> verify parity -> log everything to MLflow

Run:  python -m ipm.train --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from . import data, featurizer, model, schema
from .config import Config
from .featurizer import FeaturizerConfig
from .metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("ipm.train")


def run(config_path: str) -> dict:
    cfg = Config.load(config_path)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + temporal split -------------------------------------------------
    df = data.load_raw(cfg.data.path, max_rows=cfg.data.max_rows)
    train_df, valid_df, test_df = data.temporal_split(
        df, cfg.data.valid_fraction, cfg.data.test_fraction
    )

    # 2. Featurize -------------------------------------------------------------
    X_train, y_train = featurizer.transform(train_df), train_df[schema.TARGET].to_numpy()
    X_valid, y_valid = featurizer.transform(valid_df), valid_df[schema.TARGET].to_numpy()
    X_test, y_test = featurizer.transform(test_df), test_df[schema.TARGET].to_numpy()

    spw = model.compute_scale_pos_weight(y_train)
    logger.info("scale_pos_weight=%.2f (train install rate %.4f)", spw, y_train.mean())

    # 3. Fit with early stopping on the temporal validation set ---------------
    pipeline = model.build_pipeline(cfg.model, scale_pos_weight=spw)
    # Fit the preprocessor on train only, then feed transformed validation data
    # to LightGBM for early stopping (avoids leaking valid stats into scaling).
    from lightgbm import early_stopping, log_evaluation

    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    X_train_t = pre.fit_transform(X_train, y_train)
    X_valid_t = pre.transform(X_valid)
    clf.fit(
        X_train_t,
        y_train,
        eval_set=[(X_valid_t, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(cfg.train.early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )
    logger.info("Best iteration: %s", clf.best_iteration_)

    # 4. Evaluate (sklearn pipeline) ------------------------------------------
    def proba(X):
        return pipeline.predict_proba(X)[:, 1]

    metrics = {
        **{f"valid_{k}": v for k, v in compute_metrics(y_valid, proba(X_valid)).items()},
        **{f"test_{k}": v for k, v in compute_metrics(y_test, proba(X_test)).items()},
    }
    logger.info(
        "TEST  logloss=%.5f  roc_auc=%.4f  pr_auc=%.4f  brier=%.5f",
        metrics["test_log_loss"], metrics["test_roc_auc"],
        metrics["test_pr_auc"], metrics["test_brier"],
    )

    # 5. Package: ONNX graph + featurizer contract + metadata -----------------
    onnx_path = out_dir / "model.onnx"
    model.export_onnx(pipeline, cfg.train.onnx_opset, onnx_path)

    feat_cfg = FeaturizerConfig()
    (out_dir / "featurizer_config.json").write_text(
        json.dumps(feat_cfg.to_dict(), indent=2)
    )

    # 6. Verify sklearn <-> ONNX parity on the test set (guards the export) ---
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_proba = model.predict_proba_onnx(sess, X_test)
    parity = float(np.abs(onnx_proba - proba(X_test)).max())
    metrics["onnx_sklearn_max_abs_diff"] = parity
    logger.info("ONNX/sklearn max abs prob diff = %.2e", parity)
    if parity > 1e-4:
        raise RuntimeError(f"ONNX parity check failed (diff={parity:.2e})")

    metadata = {
        "model_version": "0.1.0",
        "framework": "lightgbm->onnx",
        "best_iteration": int(clf.best_iteration_ or cfg.model.n_estimators),
        "n_features": len(featurizer.ALL_FEATURES),
        "numeric_features": featurizer.NUMERIC_FEATURES,
        "categorical_features": featurizer.CATEGORICAL_FEATURES,
        "scale_pos_weight": spw,
        "onnx_opset": cfg.train.onnx_opset,
        "train_rows": int(len(train_df)),
        "metrics": metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # 7. Experiment tracking ---------------------------------------------------
    _log_to_mlflow(cfg, metrics, out_dir, onnx_path)

    logger.info("Artifacts written to %s", out_dir.resolve())
    return metrics


def _log_to_mlflow(cfg: Config, metrics: dict, out_dir: Path, onnx_path: Path) -> None:
    mlflow.set_experiment(cfg.train.experiment_name)
    with mlflow.start_run(run_name=cfg.train.run_name):
        # Hyper-parameters & data config.
        mlflow.log_params(cfg.model.__dict__)
        mlflow.log_params({f"data_{k}": v for k, v in cfg.data.__dict__.items()})
        # Metrics (skip NaNs which MLflow rejects).
        mlflow.log_metrics({k: v for k, v in metrics.items() if not _is_nan(v)})
        # Artifacts: the ONNX model and its companion config/metadata.
        mlflow.log_artifact(str(onnx_path), artifact_path="model")
        mlflow.log_artifact(str(out_dir / "featurizer_config.json"), artifact_path="model")
        mlflow.log_artifact(str(out_dir / "metadata.json"), artifact_path="model")


def _is_nan(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Install Prediction Model.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
