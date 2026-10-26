"""
Shared checkpointing utilities for GuardianEye training modules.

Convention
----------
All training scripts in later phases **must** use these helpers to ensure a
consistent checkpoint format and safe atomic writes.

Checkpoint format
-----------------
Each checkpoint is a ``dict`` saved with ``torch.save`` containing at minimum:

``model_state``
    ``model.state_dict()`` — the model weights.
``optimizer_state``
    ``optimizer.state_dict()`` — the optimiser state.
``epoch``
    Integer epoch index (0-based) that just completed.
``config``
    Serialisable snapshot of the training configuration (e.g. a
    ``GuardianEyeConfig.model_dump()`` dict).
``seed``
    Integer random seed used for this run.

Optional keys (include when applicable):
``scheduler_state``
    ``scheduler.state_dict()`` for LR schedulers.
``scaler_state``
    ``scaler.state_dict()`` for ``torch.cuda.amp.GradScaler``.

Naming convention
-----------------
Checkpoints are stored under ``{results_root}/checkpoints/{prefix}/`` and
follow the pattern ``epoch_{epoch:04d}.pt``.  The *latest* checkpoint for a
given prefix is the one with the highest epoch number.

Atomic writes
-------------
Each checkpoint is first written to a ``.tmp`` file alongside the target path
and then renamed, so an interrupted write never corrupts the previous saved
state.

Usage example (in a future training module)
-------------------------------------------
>>> from src.checkpoint import save_checkpoint, find_latest_checkpoint, load_checkpoint
>>> ckpt_dir = cfg.paths.results_root / "checkpoints" / "image_branch"
>>>
>>> # Resume if possible
>>> latest = find_latest_checkpoint(ckpt_dir)
>>> start_epoch = 0
>>> if latest:
...     state = load_checkpoint(latest)
...     model.load_state_dict(state["model_state"])
...     optimizer.load_state_dict(state["optimizer_state"])
...     start_epoch = state["epoch"] + 1
...     logger.info("Resumed from epoch %d", state["epoch"])
>>> else:
...     logger.info("Starting training from scratch")
>>>
>>> for epoch in range(start_epoch, cfg.image.epochs):
...     # ... training loop ...
...     save_checkpoint(
...         state={
...             "model_state": model.state_dict(),
...             "optimizer_state": optimizer.state_dict(),
...             "epoch": epoch,
...             "config": cfg.model_dump(),
...             "seed": cfg.reproducibility.seeds[0],
...         },
...         checkpoint_dir=ckpt_dir,
...         epoch=epoch,
...     )
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EPOCH_RE = re.compile(r"^epoch_(\d{4,})\.pt$")


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: Path | str,
    epoch: int,
    filename: str | None = None,
) -> Path:
    """
    Save a training checkpoint atomically.

    The checkpoint is written to a temporary file first, then renamed to the
    final path.  This ensures that an interrupted write does not corrupt the
    previous checkpoint.

    Parameters
    ----------
    state:
        Dictionary containing at least ``model_state``, ``optimizer_state``,
        ``epoch``, ``config``, and ``seed``.
    checkpoint_dir:
        Directory in which to save the checkpoint.  Created if absent.
    epoch:
        Zero-based epoch index that just completed.
    filename:
        Optional filename. If not provided, defaults to ``epoch_{epoch:04d}.pt``.

    Returns
    -------
    Path
        Path to the saved checkpoint file.

    Raises
    ------
    ValueError
        If required keys are missing from ``state``.

    """
    import torch  # deferred import — keeps CPU-only imports fast

    _validate_state(state)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        target = ckpt_dir / filename
    else:
        target = ckpt_dir / f"epoch_{epoch:04d}.pt"
    tmp = target.with_suffix(".tmp")

    torch.save(state, tmp)
    tmp.replace(target)  # atomic overwrite across POSIX and Windows

    logger.info("Checkpoint saved: %s (epoch %d)", target, epoch)
    return target


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """
    Load a checkpoint from disk.

    Parameters
    ----------
    path:
        Path to a ``.pt`` checkpoint file.

    Returns
    -------
    dict[str, Any]
        The checkpoint state dictionary.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    """
    import torch  # deferred import

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state: dict[str, Any] = torch.load(
        ckpt_path, map_location="cpu", weights_only=False
    )
    logger.info("Checkpoint loaded: %s (epoch %d)", ckpt_path, state.get("epoch", "?"))
    return state


def find_latest_checkpoint(checkpoint_dir: Path | str) -> Path | None:
    """
    Return the path to the most recent checkpoint in a directory.

    Scans ``checkpoint_dir`` for files matching ``epoch_NNNN.pt`` and returns
    the one with the highest epoch number.  Returns ``None`` when the directory
    does not exist or contains no valid checkpoints.

    Parameters
    ----------
    checkpoint_dir:
        Directory to search.

    Returns
    -------
    Path | None
        Path to the latest checkpoint, or ``None``.

    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        logger.debug("Checkpoint directory not found: %s", ckpt_dir)
        return None

    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = _EPOCH_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        logger.debug("No checkpoints found in: %s", ckpt_dir)
        return None

    candidates.sort(key=lambda t: t[0])
    latest_epoch, latest_path = candidates[-1]
    logger.info(
        "Latest checkpoint: %s (epoch %d)", latest_path, latest_epoch
    )
    return latest_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"model_state", "optimizer_state", "epoch", "config", "seed"}
)


def _validate_state(state: dict[str, Any]) -> None:
    """Raise ``ValueError`` if required checkpoint keys are missing."""
    missing = _REQUIRED_KEYS - set(state)
    if missing:
        raise ValueError(
            f"Checkpoint state is missing required keys: {sorted(missing)}"
        )
