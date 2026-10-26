"""
Training routine for GuardianEye audio branch.

Supports:
- In-domain training on pooled Belize + Vietnam train split (excluding thunder_wind)
- Checkpoint selection on pooled Belize + Vietnam val split (excluding thunder_wind)
- Leakage-free cross-site training:
  * Belize -> Vietnam: trained ONLY on Belize train, validated ONLY on Belize val,
    zero exposure to Vietnam during training/tuning.
  * Vietnam -> Belize: trained ONLY on Vietnam train, validated ONLY on Vietnam val,
    zero exposure to Belize during training/tuning.
- Class-weighted CrossEntropyLoss: w_c = N / (K * n_c), missing classes get weight 0.0
- SpecAugment applied during training ONLY.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.audio.dataset import AudioDataset
from src.audio.features import AudioProcessor
from src.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from src.config import GuardianEyeConfig
from src.evaluation.metrics import compute_binary_audio_metrics
from src.models import AUDIO_CLASSES, AudioEncoder

logger = logging.getLogger(__name__)


def compute_audio_class_weights(
    df: pd.DataFrame,
    class_names: list[str] = AUDIO_CLASSES,
) -> torch.Tensor:
    """
    Compute inverse frequency class weights: w_c = N / (K * n_c).

    If a class has 0 samples, logs a warning and assigns weight 0.0
    to avoid division by zero and degenerate loss behavior.
    """
    k = len(class_names)
    weights = torch.zeros(k, dtype=torch.float32)

    if len(df) == 0:
        logger.warning("Empty training DataFrame provided for audio class weights.")
        return weights

    counts: dict[str, int] = df["label"].value_counts().to_dict()
    total_samples = len(df)

    for idx, c in enumerate(class_names):
        count = counts.get(c, 0)
        if count == 0:
            logger.warning(
                "Audio training data has 0 instances of '%s'. Setting weight 0.0.",
                c,
            )
            weights[idx] = 0.0
        else:
            weights[idx] = float(total_samples / (k * count))

    return weights


def train_audio_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for a single epoch and return average loss."""
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(audio, return_embedding=False)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps if steps > 0 else 0.0


def validate_audio_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float | None]:
    """Validate on validation set; return (val_loss, val_f1)."""
    model.eval()
    total_loss = 0.0
    steps = 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_gunshot_probs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            logits = model(audio, return_embedding=False)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            steps += 1

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())
            all_gunshot_probs.extend(probs[:, 1].cpu().tolist())

    val_loss = total_loss / steps if steps > 0 else 0.0
    metrics = compute_binary_audio_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_gunshot_probs,
    )
    val_f1 = metrics.get("f1")

    return val_loss, val_f1


def train_audio_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_root: Path | str,
    cfg: GuardianEyeConfig,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path | str,
    processor: AudioProcessor | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    best_filename: str = "best_model.pt",
    resume: bool = True,
) -> tuple[AudioEncoder, dict[str, Any]]:
    """Train AudioEncoder on train_df, validate on val_df, and save checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = epochs or cfg.audio.epochs
    bs = batch_size or cfg.audio.batch_size
    proc = processor or AudioProcessor()

    model = AudioEncoder(
        num_classes=cfg.audio.num_classes,
        embedding_dim=cfg.audio.embedding_dim,
        use_pretrained=cfg.audio.pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.audio.lr,
        weight_decay=cfg.audio.weight_decay,
    )

    class_weights = compute_audio_class_weights(train_df, AUDIO_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = AudioDataset(
        train_df, data_root=data_root, processor=proc, is_train=True
    )
    val_dataset = AudioDataset(
        val_df, data_root=data_root, processor=proc, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)

    start_epoch = 0
    best_val_metric: float = -1.0
    history: list[dict[str, Any]] = []

    if resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            state = load_checkpoint(latest)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            start_epoch = state["epoch"] + 1
            best_val_metric = state.get("best_val_metric", -1.0)
            logger.info("Resumed audio training from epoch %d", state["epoch"])

    for epoch in range(start_epoch, n_epochs):
        train_loss = train_audio_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_f1 = validate_audio_epoch(
            model, val_loader, criterion, device
        )

        metric_for_selection = val_f1 if val_f1 is not None else -val_loss
        is_best = metric_for_selection > best_val_metric
        if is_best:
            best_val_metric = metric_for_selection

        logger.info(
            "Audio Epoch [%d/%d] Train Loss: %.4f | Val Loss: %.4f | Val F1: %s (best: %.4f)",
            epoch,
            n_epochs - 1,
            train_loss,
            val_loss,
            f"{val_f1:.4f}" if val_f1 is not None else "null",
            best_val_metric,
        )

        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.model_dump(),
            "seed": seed,
            "best_val_metric": best_val_metric,
            "metrics": {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
            },
        }

        # Save regular epoch checkpoint
        save_checkpoint(state, ckpt_dir, epoch)

        # Save best model
        if is_best:
            save_checkpoint(state, ckpt_dir, epoch, filename=best_filename)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )

    best_path = ckpt_dir / best_filename
    if best_path.exists():
        best_state = load_checkpoint(best_path)
        model.load_state_dict(best_state["model_state"])
        logger.info("Loaded best audio model from %s", best_path)

    return model, {"history": history, "best_val_metric": best_val_metric}


def train_in_domain_audio(
    manifest_df: pd.DataFrame,
    data_root: Path | str,
    cfg: GuardianEyeConfig,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path | str,
    processor: AudioProcessor | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> tuple[AudioEncoder, dict[str, Any]]:
    """
    Train in-domain audio model on pooled Belize + Vietnam train split.

    Excludes thunder_wind clips. Validates on pooled val split.
    """
    # Filter out thunder_wind clips and non-audio rows
    audio_df = manifest_df[
        (manifest_df["modality"] == "audio")
        & (manifest_df["noise_tag"] != "thunder_wind")
        & (manifest_df["source"].isin(["belize", "vietnam"]))
    ]

    train_df = audio_df[audio_df["split"] == "train"]
    val_df = audio_df[audio_df["split"] == "val"]

    logger.info(
        "In-domain audio training pool: %d train samples, %d val samples",
        len(train_df),
        len(val_df),
    )

    return train_audio_model(
        train_df=train_df,
        val_df=val_df,
        data_root=data_root,
        cfg=cfg,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        processor=processor,
        epochs=epochs,
        batch_size=batch_size,
        best_filename="best_model.pt",
    )


def train_cross_site_audio(
    manifest_df: pd.DataFrame,
    source_site: str,
    data_root: Path | str,
    cfg: GuardianEyeConfig,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path | str,
    processor: AudioProcessor | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> tuple[AudioEncoder, dict[str, Any]]:
    """
    Train cross-site audio model strictly on source_site train split.

    Selects checkpoints strictly on source_site val split.
    Zero target-site data is touched during training or tuning.
    """
    source_df = manifest_df[
        (manifest_df["modality"] == "audio")
        & (manifest_df["source"] == source_site)
        & (manifest_df["noise_tag"] != "thunder_wind")
    ]

    train_df = source_df[source_df["split"] == "train"]
    val_df = source_df[source_df["split"] == "val"]

    logger.info(
        "Cross-site audio training for source '%s': %d train, %d val",
        source_site,
        len(train_df),
        len(val_df),
    )

    filename = f"cross_site_{source_site}_best.pt"
    return train_audio_model(
        train_df=train_df,
        val_df=val_df,
        data_root=data_root,
        cfg=cfg,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        processor=processor,
        epochs=epochs,
        batch_size=batch_size,
        best_filename=filename,
        resume=False,
    )
