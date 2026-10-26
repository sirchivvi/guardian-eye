"""
Unit tests for unimodal training routines, class weighting, and checkpoint resumption.

Verifies:
- Inverse frequency class weights with absent training classes getting weight 0.0
- Avoidance of degenerate loss behavior when a class is absent
- Resumable epoch checkpointing (model, optimizer, epoch, config, seed)
- Single-epoch training step updates weights and decreases loss on fixture data
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.audio.train import compute_audio_class_weights
from src.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from src.config import load_config
from src.image.train import compute_image_class_weights
from src.models import AUDIO_CLASSES, IMAGE_CLASSES, ImageEncoder


def test_image_class_weights_balanced() -> None:
    """Class weights must follow inverse frequency: w_c = N / (K * n_c)."""
    # 4 classes: 10 human, 10 vehicle, 10 animal, 10 empty -> total 40
    data = {"label": ["human"] * 10 + ["vehicle"] * 10 + ["animal"] * 10 + ["empty"] * 10}
    df = pd.DataFrame(data)
    weights = compute_image_class_weights(df, IMAGE_CLASSES)

    assert len(weights) == 4
    # All counts equal -> all weights equal 1.0 (40 / (4 * 10) = 1.0)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0, 1.0, 1.0]))


def test_image_class_weights_missing_class(caplog: pytest.LogCaptureFixture) -> None:
    """Missing class must get weight 0.0, log a warning, and not cause division by zero."""
    # human: 20, animal: 20. (vehicle and empty are absent)
    data = {"label": ["human"] * 20 + ["animal"] * 20}
    df = pd.DataFrame(data)

    with caplog.at_level(logging.WARNING):
        weights = compute_image_class_weights(df, IMAGE_CLASSES)

    # vehicle (idx 1) and empty (idx 3) must be 0.0
    assert weights[1].item() == 0.0
    assert weights[3].item() == 0.0

    # human (idx 0) and animal (idx 2) must be positive: 40 / (4 * 20) = 0.5
    assert weights[0].item() == pytest.approx(0.5)
    assert weights[2].item() == pytest.approx(0.5)

    # Warning must be logged
    assert "vehicle" in caplog.text
    assert "empty" in caplog.text


def test_audio_class_weights_imbalanced() -> None:
    """Audio class weights must reflect class imbalance: w_c = N / (K * n_c)."""
    # 90 background, 10 gunshot -> total 100
    data = {"label": ["background"] * 90 + ["gunshot"] * 10}
    df = pd.DataFrame(data)
    weights = compute_audio_class_weights(df, AUDIO_CLASSES)

    # background (idx 0): 100 / (2 * 90) = 0.5555...
    # gunshot (idx 1): 100 / (2 * 10) = 5.0
    assert weights[0].item() == pytest.approx(100.0 / 180.0)
    assert weights[1].item() == pytest.approx(5.0)


def test_loss_with_zero_weight_does_not_nan() -> None:
    """CrossEntropyLoss with 0-weight for absent classes must evaluate without NaN."""
    weights = torch.tensor([1.0, 0.0, 1.0, 0.0])  # classes 1 and 3 missing
    criterion = nn.CrossEntropyLoss(weight=weights)

    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 2, 0, 2], dtype=torch.long)  # only present classes

    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_checkpoint_saving_and_resumption(tmp_path: Path) -> None:
    """Checkpoint must save all required state elements and restore model parameters exactly."""
    cfg = load_config()
    model = ImageEncoder(num_classes=4, embedding_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = tmp_path / "checkpoints"
    epoch = 3
    seed = 27

    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg.model_dump(),
        "seed": seed,
        "scheduler_state": None,
        "scaler_state": None,
        "best_val_metric": 0.85,
        "metrics": {"train_loss": 0.25, "val_loss": 0.30},
    }

    saved_path = save_checkpoint(state, ckpt_dir, epoch)
    assert saved_path.exists()
    assert find_latest_checkpoint(ckpt_dir) == saved_path

    # Restore into a fresh model
    new_model = ImageEncoder(num_classes=4, embedding_dim=128)
    loaded_state = load_checkpoint(saved_path)

    new_model.load_state_dict(loaded_state["model_state"])

    # Verify model parameters match exactly
    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=False):
        assert torch.equal(p1, p2)

    assert loaded_state["epoch"] == epoch
    assert loaded_state["seed"] == seed
    assert loaded_state["best_val_metric"] == pytest.approx(0.85)


def test_training_step_reduces_loss() -> None:
    """Running multiple optimizer steps on a fixed fixture batch must reduce loss."""
    torch.manual_seed(42)
    model = ImageEncoder(num_classes=4, embedding_dim=128)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    # Initial loss in train mode
    optimizer.zero_grad()
    initial_loss = criterion(model(x), y)
    initial_loss.backward()
    optimizer.step()

    # Train for 5 iterations
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss.item(), (
        f"Loss did not decrease: {initial_loss.item()} -> {final_loss}"
    )
