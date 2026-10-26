"""
Evaluation routine for GuardianEye image branch.

Computes:
- Overall accuracy and macro F1
- Per-class precision, recall, and F1 (human, vehicle, animal, empty)
- Per-view breakdown: aerial_thermal vs camera_trap
- Strict zero-denominator honesty: undefined metrics are stored as None/null,
  views or classes with 0 test instances are recorded as NOT YET RUN.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    compute_image_view_metrics,
    compute_multiclass_image_metrics,
)
from src.image.dataset import ImageDataset
from src.models import IMAGE_CLASSES

logger = logging.getLogger(__name__)


def evaluate_image_model(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    data_root: Path | str,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Evaluate trained ImageEncoder exclusively on test_df.

    Returns dictionary formatted for serialization to image_metrics.json.
    """
    model.eval()

    if len(test_df) == 0:
        logger.warning("Empty test DataFrame passed to evaluate_image_model.")
        return {
            "status": "NOT YET RUN",
            "reason": "Test set contains 0 samples.",
            "overall": {
                "count": 0,
                "accuracy": None,
                "f1_macro": None,
                "per_class": {
                    c: {"status": "NOT YET RUN", "reason": "No test samples."}
                    for c in IMAGE_CLASSES
                },
            },
            "by_view": {
                "aerial_thermal": {
                    "status": "NOT YET RUN",
                    "reason": "No test samples.",
                },
                "camera_trap": {
                    "status": "NOT YET RUN",
                    "reason": "No test samples.",
                },
            },
        }

    dataset = ImageDataset(test_df, data_root=data_root, is_train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_views: list[str] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"]
            views = batch["view"]

            logits = model(imgs, return_embedding=False)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(labels.tolist())
            all_views.extend(views)

    # Compute overall multiclass metrics
    overall_metrics = compute_multiclass_image_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=IMAGE_CLASSES,
    )

    # Compute per-view breakdown
    eval_df = pd.DataFrame(
        {
            "label_idx": all_targets,
            "pred_idx": all_preds,
            "view": all_views,
        }
    )
    view_metrics = compute_image_view_metrics(eval_df, class_names=IMAGE_CLASSES)

    return {
        "status": "OK",
        "count": len(all_targets),
        "overall": overall_metrics,
        "by_view": view_metrics,
    }
