"""Unit tests for src/checkpoint.py.

All tests run on CPU using temporary directories (no CUDA required).
Covers:
- save_checkpoint writes the expected file
- Atomic write pattern (no .tmp left behind)
- load_checkpoint round-trips state correctly
- find_latest_checkpoint returns None for empty / missing dirs
- find_latest_checkpoint returns the highest-epoch file
- _validate_state raises ValueError for missing required keys
- save_checkpoint raises ValueError for bad state
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.checkpoint import (
    _validate_state,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_state(epoch: int = 0) -> dict[str, Any]:
    """Return a minimal valid checkpoint state (no real tensors needed)."""
    return {
        "model_state": {"weight": [1.0, 2.0]},
        "optimizer_state": {"lr": 0.0001},
        "epoch": epoch,
        "config": {"project": {"name": "guardian-eye"}},
        "seed": 27,
    }


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_creates_file(tmp_path: Path) -> None:
    """save_checkpoint must create epoch_0000.pt in the given directory."""

    state = _minimal_state(epoch=0)
    path = save_checkpoint(state, tmp_path, epoch=0)
    assert path.exists()
    assert path.name == "epoch_0000.pt"


def test_save_checkpoint_no_tmp_file_left(tmp_path: Path) -> None:
    """No .tmp file must remain after a successful save."""
    state = _minimal_state(epoch=1)
    save_checkpoint(state, tmp_path, epoch=1)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"


def test_save_checkpoint_creates_dir_if_missing(tmp_path: Path) -> None:
    """save_checkpoint must create the checkpoint directory if it does not exist."""
    ckpt_dir = tmp_path / "nested" / "checkpoints"
    assert not ckpt_dir.exists()
    save_checkpoint(_minimal_state(0), ckpt_dir, epoch=0)
    assert ckpt_dir.exists()


def test_save_checkpoint_returns_correct_path(tmp_path: Path) -> None:
    """save_checkpoint must return the final .pt path, not the tmp path."""
    path = save_checkpoint(_minimal_state(3), tmp_path, epoch=3)
    assert path == tmp_path / "epoch_0003.pt"


def test_save_checkpoint_missing_key_raises(tmp_path: Path) -> None:
    """save_checkpoint must raise ValueError when required keys are absent."""
    bad_state = {"model_state": {}, "epoch": 0}  # missing optimizer_state, config, seed
    with pytest.raises(ValueError, match="missing required keys"):
        save_checkpoint(bad_state, tmp_path, epoch=0)


def test_save_checkpoint_overwrites_existing_file(tmp_path: Path) -> None:
    """save_checkpoint must overwrite an existing checkpoint atomically without error."""
    state1 = _minimal_state(epoch=0)
    state1["seed"] = 100
    path1 = save_checkpoint(state1, tmp_path, epoch=0)
    assert path1.exists()

    state2 = _minimal_state(epoch=0)
    state2["seed"] = 200
    path2 = save_checkpoint(state2, tmp_path, epoch=0)
    assert path2 == path1

    loaded = load_checkpoint(path2)
    assert loaded["seed"] == 200


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


def test_load_checkpoint_round_trips(tmp_path: Path) -> None:
    """load_checkpoint must return the same state that was saved."""
    state = _minimal_state(epoch=5)
    state["scheduler_state"] = {"last_epoch": 5}
    save_checkpoint(state, tmp_path, epoch=5)
    loaded = load_checkpoint(tmp_path / "epoch_0005.pt")

    assert loaded["epoch"] == 5
    assert loaded["seed"] == 27
    assert "scheduler_state" in loaded


def test_load_checkpoint_file_not_found(tmp_path: Path) -> None:
    """load_checkpoint must raise FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


# ---------------------------------------------------------------------------
# find_latest_checkpoint
# ---------------------------------------------------------------------------


def test_find_latest_returns_none_for_missing_dir(tmp_path: Path) -> None:
    """find_latest_checkpoint returns None when the directory does not exist."""
    result = find_latest_checkpoint(tmp_path / "nonexistent")
    assert result is None


def test_find_latest_returns_none_for_empty_dir(tmp_path: Path) -> None:
    """find_latest_checkpoint returns None when no .pt files are present."""
    (tmp_path / "checkpoints").mkdir()
    result = find_latest_checkpoint(tmp_path / "checkpoints")
    assert result is None


def test_find_latest_returns_highest_epoch(tmp_path: Path) -> None:
    """find_latest_checkpoint must return the epoch_NNNN.pt with the largest N."""
    for epoch in [0, 1, 5, 3]:
        save_checkpoint(_minimal_state(epoch), tmp_path, epoch=epoch)

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "epoch_0005.pt"


def test_find_latest_single_checkpoint(tmp_path: Path) -> None:
    """find_latest_checkpoint works with a single checkpoint present."""
    save_checkpoint(_minimal_state(0), tmp_path, epoch=0)
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "epoch_0000.pt"


def test_find_latest_ignores_non_checkpoint_files(tmp_path: Path) -> None:
    """find_latest_checkpoint must ignore files that don't match epoch_NNNN.pt."""
    save_checkpoint(_minimal_state(2), tmp_path, epoch=2)
    (tmp_path / "best.pt").write_text("not a checkpoint")
    (tmp_path / "readme.txt").write_text("ignore me")

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "epoch_0002.pt"


# ---------------------------------------------------------------------------
# _validate_state
# ---------------------------------------------------------------------------


def test_validate_state_passes_for_complete_state() -> None:
    """_validate_state must not raise for a complete state dict."""
    _validate_state(_minimal_state())


def test_validate_state_raises_for_each_missing_key() -> None:
    """_validate_state must raise ValueError for each individually missing key."""
    required = {"model_state", "optimizer_state", "epoch", "config", "seed"}
    full = _minimal_state()
    for key in required:
        partial = {k: v for k, v in full.items() if k != key}
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_state(partial)


def test_validate_state_allows_extra_optional_keys() -> None:
    """_validate_state must allow optional keys like scheduler_state."""
    state = _minimal_state()
    state["scheduler_state"] = {"lr": 0.001}
    state["scaler_state"] = {"scale": 65536.0}
    _validate_state(state)  # must not raise


# ---------------------------------------------------------------------------
# Resume-from-latest integration
# ---------------------------------------------------------------------------


def test_resume_from_latest_pattern(tmp_path: Path) -> None:
    """Simulate a real training resume: save epoch 0, 1, discover 1, load it."""
    # Simulated epoch 0
    save_checkpoint(_minimal_state(0), tmp_path, epoch=0)
    # Simulated epoch 1
    save_checkpoint(_minimal_state(1), tmp_path, epoch=1)

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None

    loaded = load_checkpoint(latest)
    start_epoch = loaded["epoch"] + 1  # resume convention
    assert start_epoch == 2
