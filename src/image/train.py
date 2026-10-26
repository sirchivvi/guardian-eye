"""
Training routine for GuardianEye image branch.

Features:
- Class-weighted CrossEntropyLoss: w_c = N / (K * n_c), missing classes get weight 0.0
- Epoch checkpointing with atomic saving and resumption
- Model selection via validation split macro F1
- Resumable optimizer, epoch, config, and seed tracking
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

from src.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from src.config import GuardianEyeConfig
from src.evaluation.metrics import compute_multiclass_image_metrics
from src.image.dataset import ImageDataset
from src.models import IMAGE_CLASSES, ImageEncoder

logger = logging.getLogger(__name__)


def compute_image_class_weights(
    df: pd.DataFrame,
    class_names: list[str] = IMAGE_CLASSES,
) -> torch.Tensor:
    """
    Compute inverse frequency class weights: w_c = N / (K * n_c).

    If a class has 0 samples, logs a warning and assigns weight 0.0
    to avoid division by zero and degenerate loss behavior.
    """
    k = len(class_names)
    weights = torch.zeros(k, dtype=torch.float32)

    if len(df) == 0:
        logger.warning("Empty training DataFrame provided for class weights.")
        return weights

    counts: dict[str, int] = df["label"].value_counts().to_dict()
    total_samples = len(df)

    for idx, c in enumerate(class_names):
        count = counts.get(c, 0)
        if count == 0:
            logger.warning(
                "Training data has 0 instances of '%s'. Setting weight 0.0.",
                c,
            )
            weights[idx] = 0.0
        else:
            weights[idx] = float(total_samples / (k * count))

    return weights


def train_image_epoch(
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
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, return_embedding=False)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps if steps > 0 else 0.0


def validate_image_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float | None]:
    """Validate on validation set; return (val_loss, val_macro_f1)."""
    model.eval()
    total_loss = 0.0
    steps = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, return_embedding=False)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            steps += 1

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

    val_loss = total_loss / steps if steps > 0 else 0.0
    metrics = compute_multiclass_image_metrics(
        all_targets, all_preds, class_names=IMAGE_CLASSES
    )
    val_f1 = metrics.get("f1_macro")

    return val_loss, val_f1


def train_image_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_root: Path | str,
    cfg: GuardianEyeConfig,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path | str,
    epochs: int | None = None,
    batch_size: int | None = None,
    resume: bool = True,
) -> tuple[ImageEncoder, dict[str, Any]]:
    """Train ImageEncoder on train_df, validate on val_df, and save checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = epochs or cfg.image.epochs
    bs = batch_size or cfg.image.batch_size

    # Instantiate model
    model = ImageEncoder(
        num_classes=cfg.image.num_classes,
        embedding_dim=cfg.image.embedding_dim,
        use_pretrained=cfg.image.pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.image.lr,
        weight_decay=cfg.image.weight_decay,
    )

    # Compute class weights
    class_weights = compute_image_class_weights(train_df, IMAGE_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Prepare datasets and loaders
    train_dataset = ImageDataset(train_df, data_root=data_root, is_train=True)
    val_dataset = ImageDataset(val_df, data_root=data_root, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)

    start_epoch = 0
    best_val_metric: float = -1.0
    history: list[dict[str, Any]] = []

    # Check for resumable checkpoint
    if resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            state = load_checkpoint(latest)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            start_epoch = state["epoch"] + 1
            best_val_metric = state.get("best_val_metric", -1.0)
            logger.info("Resumed image training from epoch %d", state["epoch"])

    for epoch in range(start_epoch, n_epochs):
        train_loss = train_image_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_f1 = validate_image_epoch(
            model, val_loader, criterion, device
        )

        metric_for_selection = val_f1 if val_f1 is not None else -val_loss
        is_best = metric_for_selection > best_val_metric
        if is_best:
            best_val_metric = metric_for_selection

        logger.info(
            "Epoch [%d/%d] Train Loss: %.4f | Val Loss: %.4f | Val F1: %s (best: %.4f)",
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

        # Save best checkpoint
        if is_best:
            save_checkpoint(state, ckpt_dir, epoch, filename="best_model.pt")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )

    # Load best checkpoint if saved
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        best_state = load_checkpoint(best_path)
        model.load_state_dict(best_state["model_state"])
        logger.info("Loaded best image model from %s", best_path)

    return model, {"history": history, "best_val_metric": best_val_metric}
