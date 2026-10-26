"""
Serialization utilities for GuardianEye model outputs.

Extracts logits, probabilities, and 128-d penultimate embeddings for EVERY valid
manifest item after training, and writes atomically to:
results/runs/seed_<n>/outputs/image_outputs.pt
results/runs/seed_<n>/outputs/audio_outputs.pt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.audio.dataset import AudioDataset
from src.audio.features import AudioProcessor
from src.image.dataset import ImageDataset

logger = logging.getLogger(__name__)


def save_serialized_outputs(
    data: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """
    Save outputs dictionary atomically.

    Parameters
    ----------
    data:
        Dictionary containing manifest_indices, paths, logits, probabilities,
        embeddings, labels, and splits.
    output_path:
        Target .pt file path.

    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")

    torch.save(data, tmp)
    tmp.replace(target)
    logger.info("Serialized outputs saved atomically to %s", target)
    return target


def extract_image_outputs(
    model: torch.nn.Module,
    manifest_df: pd.DataFrame,
    data_root: Path | str,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Extract predictions and embeddings for all image files in manifest_df."""
    model.eval()
    dataset = ImageDataset(manifest_df, data_root=data_root, is_train=False)

    if len(dataset) == 0:
        return {
            "manifest_indices": torch.empty((0,), dtype=torch.long),
            "paths": [],
            "logits": torch.empty((0, 4), dtype=torch.float32),
            "probabilities": torch.empty((0, 4), dtype=torch.float32),
            "embeddings": torch.empty((0, 128), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
            "splits": [],
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # strictly deterministic order
        num_workers=0,
    )

    all_indices: list[int] = []
    all_paths: list[str] = []
    all_logits: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"]
            indices = batch["manifest_idx"]
            paths = batch["path"]

            logits, embeddings = model(imgs, return_embedding=True)
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_indices.extend(indices.tolist())
            all_paths.extend(paths)

    # Get split information corresponding to each row in the dataset
    splits = [
        str(dataset.df.iloc[i].get("split", "unknown")) for i in range(len(dataset))
    ]

    return {
        "manifest_indices": torch.tensor(all_indices, dtype=torch.long),
        "paths": all_paths,
        "logits": torch.cat(all_logits, dim=0),
        "probabilities": torch.cat(all_probs, dim=0),
        "embeddings": torch.cat(all_embeddings, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "splits": splits,
    }


def extract_audio_outputs(
    model: torch.nn.Module,
    manifest_df: pd.DataFrame,
    data_root: Path | str,
    device: torch.device,
    processor: AudioProcessor | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Extract predictions and embeddings for all audio files in manifest_df.

    Includes held-out thunder_wind clips.
    """
    model.eval()
    dataset = AudioDataset(
        manifest_df,
        data_root=data_root,
        processor=processor,
        is_train=False,  # Never apply SpecAugment during output extraction
    )

    if len(dataset) == 0:
        return {
            "manifest_indices": torch.empty((0,), dtype=torch.long),
            "paths": [],
            "logits": torch.empty((0, 2), dtype=torch.float32),
            "probabilities": torch.empty((0, 2), dtype=torch.float32),
            "embeddings": torch.empty((0, 128), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
            "splits": [],
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # strictly deterministic order
        num_workers=0,
    )

    all_indices: list[int] = []
    all_paths: list[str] = []
    all_logits: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            labels = batch["label"]
            indices = batch["manifest_idx"]
            paths = batch["path"]

            logits, embeddings = model(audio, return_embedding=True)
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_indices.extend(indices.tolist())
            all_paths.extend(paths)

    splits = [
        str(dataset.df.iloc[i].get("split", "unknown")) for i in range(len(dataset))
    ]

    return {
        "manifest_indices": torch.tensor(all_indices, dtype=torch.long),
        "paths": all_paths,
        "logits": torch.cat(all_logits, dim=0),
        "probabilities": torch.cat(all_probs, dim=0),
        "embeddings": torch.cat(all_embeddings, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "splits": splits,
    }
