"""Prediction-quality metrics for a binary, heavily-imbalanced target.

We report more than the requested logloss/AUC because, at a ~3% base-rate,
ROC-AUC alone is optimistic and uninformative about ranking the positive class:

* ``log_loss``  -> the assignment's headline metric; punishes mis-calibration.
* ``roc_auc``   -> the assignment's headline metric; threshold-free ranking.
* ``pr_auc``    -> average precision; the honest ranking metric under imbalance.
* ``brier``     -> calibration quality, which matters because bids are computed
                   from the *probability*, not a 0/1 decision.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)

    metrics = {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(y_true.mean()),
        "mean_predicted": float(y_prob.mean()),
        "n": int(len(y_true)),
    }
    # AUC metrics are only defined when both classes are present.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics
