"""
Evaluation routine for GuardianEye audio branch.

Computes:
- Binary metrics (accuracy, precision, recall, F1, FPR, ROC-AUC) on test split
- In-domain evaluation on pooled Belize + Vietnam test split
- Leakage-free cross-site evaluations:
  * Belize -> Vietnam: evaluated strictly on Vietnam test split
  * Vietnam -> Belize: evaluated strictly on Belize test split
- Strict zero-denominator honesty: undefined metrics stored as null.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.audio.dataset import AudioDataset
from src.audio.features import AudioProcessor
from src.evaluation.metrics import compute_binary_audio_metrics

logger = logging.getLogger(__name__)


def evaluate_audio_dataset(
    model: torch.nn.Module,
    df: pd.DataFrame,
    data_root: Path | str,
    device: torch.device,
    processor: AudioProcessor | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate audio model on a given subset DataFrame."""
    model.eval()

    if len(df) == 0:
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

    dataset = AudioDataset(
        df,
        data_root=data_root,
        processor=processor,
        is_train=False,  # Never apply SpecAugment during evaluation
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_gunshot_probs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            labels = batch["label"]

            logits = model(audio, return_embedding=False)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(labels.tolist())
            all_gunshot_probs.extend(probs[:, 1].cpu().tolist())

    return compute_binary_audio_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_gunshot_probs,
    )


def evaluate_audio_branch(
    in_domain_model: torch.nn.Module,
    manifest_df: pd.DataFrame,
    data_root: Path | str,
    device: torch.device,
    processor: AudioProcessor | None = None,
    cross_site_models: dict[str, torch.nn.Module] | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run full audio evaluation including in-domain test and cross-site tests."""
    results: dict[str, Any] = {}

    # In-domain test set (Belize + Vietnam test split, excluding thunder_wind)
    in_domain_test_df = manifest_df[
        (manifest_df["modality"] == "audio")
        & (manifest_df["split"] == "test")
        & (manifest_df["noise_tag"] != "thunder_wind")
        & (manifest_df["source"].isin(["belize", "vietnam"]))
    ]
    results["in_domain"] = evaluate_audio_dataset(
        in_domain_model,
        in_domain_test_df,
        data_root=data_root,
        device=device,
        processor=processor,
        batch_size=batch_size,
    )

    # Cross-site evaluations if models provided
    cross_site_results: dict[str, Any] = {}
    if cross_site_models:
        # Belize -> Vietnam: trained on Belize, evaluated on Vietnam test split
        if "belize_to_vietnam" in cross_site_models:
            vietnam_test_df = manifest_df[
                (manifest_df["modality"] == "audio")
                & (manifest_df["split"] == "test")
                & (manifest_df["source"] == "vietnam")
                & (manifest_df["noise_tag"] != "thunder_wind")
            ]
            cross_site_results["belize_to_vietnam"] = evaluate_audio_dataset(
                cross_site_models["belize_to_vietnam"],
                vietnam_test_df,
                data_root=data_root,
                device=device,
                processor=processor,
                batch_size=batch_size,
            )

        # Vietnam -> Belize: trained on Vietnam, evaluated on Belize test split
        if "vietnam_to_belize" in cross_site_models:
            belize_test_df = manifest_df[
                (manifest_df["modality"] == "audio")
                & (manifest_df["split"] == "test")
                & (manifest_df["source"] == "belize")
                & (manifest_df["noise_tag"] != "thunder_wind")
            ]
            cross_site_results["vietnam_to_belize"] = evaluate_audio_dataset(
                cross_site_models["vietnam_to_belize"],
                belize_test_df,
                data_root=data_root,
                device=device,
                processor=processor,
                batch_size=batch_size,
            )

    results["cross_site"] = cross_site_results
    return results
