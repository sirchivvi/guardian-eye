"""
Metric computation utilities for GuardianEye unimodal models.

Implements zero-denominator honesty:
- A metric with a zero denominator is left undefined (represented as None / null in JSON),
  never coerced to 0.0.
- A view or class with no test files is recorded as NOT YET RUN with a specific reason.
- Audio positive class is gunshot (1); negative class is background (0).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Return numerator / denominator if denominator > 0, otherwise None."""
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def compute_binary_audio_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute binary classification metrics for the audio branch (gunshot as positive class).

    Returns
    -------
    dict[str, Any]
        Dictionary with count, accuracy, precision, recall, f1, fpr, roc_auc, and reasons.

    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    n_samples = int(len(y_t))
    if n_samples == 0:
        return {
            "status": "NOT YET RUN",
            "reason": "Test set contains 0 samples.",
            "count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "fpr": None,
            "roc_auc": None,
        }

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))

    accuracy = float((tp + tn) / n_samples)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = None

    # FPR = FP / (FP + TN)
    fpr = safe_divide(fp, fp + tn)

    # ROC-AUC requires probabilities and at least one positive and one negative sample
    roc_auc = None
    roc_auc_reason = None
    if y_probs is not None:
        probs = np.asarray(y_probs, dtype=float)
        if n_pos > 0 and n_neg > 0:
            try:
                roc_auc = float(roc_auc_score(y_t, probs))
            except Exception as exc:
                roc_auc_reason = f"ROC-AUC calculation error: {exc}"
        else:
            roc_auc_reason = "Single class present in ground truth; ROC-AUC undefined."

    result: dict[str, Any] = {
        "status": "OK",
        "count": n_samples,
        "n_gunshot": n_pos,
        "n_background": n_neg,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
    }
    if roc_auc_reason:
        result["roc_auc_reason"] = roc_auc_reason

    return result


def compute_multiclass_image_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Compute multiclass classification metrics for the image branch.

    Returns per-class precision, recall, F1, overall accuracy, and macro F1.
    Classes with no ground-truth samples are reported with status NOT YET RUN.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    n_samples = int(len(y_t))
    if n_samples == 0:
        return {
            "status": "NOT YET RUN",
            "reason": "Test set contains 0 samples.",
            "count": 0,
            "accuracy": None,
            "f1_macro": None,
            "per_class": {
                c: {
                    "status": "NOT YET RUN",
                    "reason": "No test instances.",
                    "precision": None,
                    "recall": None,
                    "f1": None,
                }
                for c in class_names
            },
        }

    accuracy = float(np.sum(y_t == y_p) / n_samples)
    per_class: dict[str, Any] = {}
    valid_f1s: list[float] = []

    for idx, class_name in enumerate(class_names):
        n_c = int(np.sum(y_t == idx))
        if n_c == 0:
            per_class[class_name] = {
                "status": "NOT YET RUN",
                "reason": f"No ground-truth test instances for class '{class_name}'.",
                "support": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            continue

        tp = int(np.sum((y_t == idx) & (y_p == idx)))
        fp = int(np.sum((y_t != idx) & (y_p == idx)))
        fn = int(np.sum((y_t == idx) & (y_p != idx)))

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = float(2 * precision * recall / (precision + recall))
            valid_f1s.append(f1)
        else:
            f1 = None

        per_class[class_name] = {
            "status": "OK",
            "support": n_c,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    f1_macro = float(np.mean(valid_f1s)) if valid_f1s else None

    return {
        "status": "OK",
        "count": n_samples,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "per_class": per_class,
    }


def compute_image_view_metrics(
    df: Any,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Compute metrics broken down by view (aerial_thermal vs camera_trap).

    df must have columns 'label_idx', 'pred_idx', and 'view'.
    """
    views = ["aerial_thermal", "camera_trap"]
    view_metrics: dict[str, Any] = {}

    for v in views:
        view_df = df[df["view"] == v]
        if len(view_df) == 0:
            view_metrics[v] = {
                "status": "NOT YET RUN",
                "reason": f"No test samples found for view '{v}'.",
                "count": 0,
                "accuracy": None,
                "f1_macro": None,
                "per_class": {},
            }
        else:
            y_t = view_df["label_idx"].values
            y_p = view_df["pred_idx"].values
            view_metrics[v] = compute_multiclass_image_metrics(
                y_t, y_p, class_names=class_names
            )

    return view_metrics
