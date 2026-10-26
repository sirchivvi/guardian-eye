"""
Unit tests for unimodal evaluation, zero-denominator honesty, and NOT YET RUN handling.

Verifies:
- Safe division and zero-denominator honesty (metrics left as None, never 0.0)
- Audio binary metrics: gunshot as positive class, FPR = FP / (FP + TN)
- Single-class test sets reporting ROC-AUC as undefined (None) with reason
- Image multiclass metrics: unsupported classes reported as NOT YET RUN with reason
- Image view breakdown: views with no test files reported as NOT YET RUN
- Clean exit and unimodal_status.json generation on unpopulated data
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from scripts.train_audio import run_audio_training_for_seed
from scripts.train_image import run_image_training_for_seed
from src.config import load_config
from src.evaluation.metrics import (
    compute_binary_audio_metrics,
    compute_image_view_metrics,
    compute_multiclass_image_metrics,
    safe_divide,
)
from src.models import IMAGE_CLASSES


def test_safe_divide() -> None:
    """safe_divide returns float when denom > 0, and None when denom <= 0."""
    assert safe_divide(10.0, 2.0) == 5.0
    assert safe_divide(0.0, 5.0) == 0.0
    assert safe_divide(5.0, 0.0) is None
    assert safe_divide(5.0, -1.0) is None


def test_audio_binary_metrics_zero_denominators() -> None:
    """Audio metrics with zero denominators must evaluate to None, never 0.0."""
    # Case 1: Ground truth has 0 gunshots (all background)
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    metrics = compute_binary_audio_metrics(y_true, y_pred)

    assert metrics["count"] == 4
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 4

    # Precision: 0 TP, 0 FP -> TP + FP == 0 -> None
    assert metrics["precision"] is None
    # Recall: 0 TP, 0 FN -> TP + FN == 0 -> None
    assert metrics["recall"] is None
    # F1 -> None
    assert metrics["f1"] is None
    # FPR: 0 FP / (0 FP + 4 TN) = 0.0
    assert metrics["fpr"] == 0.0

    # Case 2: Ground truth has only gunshots (no background)
    y_true_all_pos = [1, 1, 1]
    y_pred_all_pos = [1, 1, 1]
    metrics_all_pos = compute_binary_audio_metrics(y_true_all_pos, y_pred_all_pos)

    # FPR: 0 FP, 0 TN -> FP + TN == 0 -> None (undefined)
    assert metrics_all_pos["fpr"] is None
    assert metrics_all_pos["recall"] == 1.0


def test_audio_binary_metrics_roc_auc_single_class() -> None:
    """When test set contains only one class, ROC-AUC must be None with reason."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    y_probs = [0.1, 0.2, 0.3]
    metrics = compute_binary_audio_metrics(y_true, y_pred, y_probs=y_probs)

    assert metrics["roc_auc"] is None
    assert "Single class present" in metrics.get("roc_auc_reason", "")


def test_image_multiclass_metrics_missing_class() -> None:
    """Classes with 0 test samples must report status NOT YET RUN and None metrics."""
    # Only human (0) and animal (2) present in test data
    y_true = [0, 0, 2, 2]
    y_pred = [0, 0, 2, 2]
    metrics = compute_multiclass_image_metrics(y_true, y_pred, class_names=IMAGE_CLASSES)

    assert metrics["accuracy"] == 1.0
    assert metrics["status"] == "OK"

    per_class = metrics["per_class"]
    # human (0) and animal (2)
    assert per_class["human"]["status"] == "OK"
    assert per_class["human"]["f1"] == 1.0
    assert per_class["animal"]["status"] == "OK"
    assert per_class["animal"]["f1"] == 1.0

    # vehicle (1) and empty (3) had 0 test samples
    assert per_class["vehicle"]["status"] == "NOT YET RUN"
    assert per_class["vehicle"]["f1"] is None
    assert per_class["empty"]["status"] == "NOT YET RUN"
    assert per_class["empty"]["f1"] is None


def test_image_view_metrics_missing_view() -> None:
    """If a view has no test files, its status must be written as NOT YET RUN."""
    # Only aerial_thermal present
    df = pd.DataFrame(
        {
            "label_idx": [0, 2],
            "pred_idx": [0, 2],
            "view": ["aerial_thermal", "aerial_thermal"],
        }
    )
    view_metrics = compute_image_view_metrics(df, class_names=IMAGE_CLASSES)

    assert view_metrics["aerial_thermal"]["status"] == "OK"
    assert view_metrics["camera_trap"]["status"] == "NOT YET RUN"
    assert "No test samples found" in view_metrics["camera_trap"]["reason"]


def test_unpopulated_manifest_exits_cleanly_with_status_json(tmp_path: Path) -> None:
    """When manifest is NOT YET RUN, training writes unimodal_status.json and exits 0."""
    cfg = load_config()
    cfg.paths.results_root = tmp_path / "results"
    run_dir = cfg.paths.results_root / "runs" / "seed_27"
    manifest_dir = run_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Write Phase 1 NOT YET RUN manifest_status.json
    manifest_status = {
        "seed": 27,
        "status": "NOT YET RUN",
        "reason": "No raw dataset files found.",
    }
    with (manifest_dir / "manifest_status.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_status, f)

    device = torch.device("cpu")

    # Run image training
    code_img = run_image_training_for_seed(cfg, seed=27, device=device)
    assert code_img == 0

    # Run audio training
    code_audio = run_audio_training_for_seed(cfg, seed=27, device=device)
    assert code_audio == 0

    # Confirm unimodal_status.json written
    status_file = run_dir / "unimodal_status.json"
    assert status_file.exists()

    with status_file.open("r", encoding="utf-8") as f:
        unimodal_status = json.load(f)

    assert unimodal_status["status"] == "NOT YET RUN"
    assert unimodal_status["image_trained"] is False
    assert unimodal_status["audio_trained"] is False
    assert "NOT YET RUN" in unimodal_status["reason"]
