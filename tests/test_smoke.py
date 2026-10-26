"""Unit tests for smoke mode wiring and device detection.

These tests verify:
- get_device() returns a valid torch.device without CUDA
- Device detection does not raise on CPU-only machines
- load_config() smoke section is present and valid
- scripts/run.py --smoke flag is wired (import-level check)
- Smoke settings are sensibly bounded for CPU execution
"""

from __future__ import annotations

import sys


def test_get_device_returns_torch_device() -> None:
    """get_device() must return a torch.device (cuda or cpu) without raising."""
    import torch

    from src.logging_config import get_device

    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "cpu"}


def test_get_device_cpu_when_no_cuda(monkeypatch) -> None:
    """get_device() must return CPU when CUDA is not available."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from src.logging_config import get_device

    device = get_device()
    assert device.type == "cpu"


def test_smoke_config_loaded() -> None:
    """Config smoke section must load and have sensible CPU-friendly values."""
    from src.config import load_config

    cfg = load_config()
    assert cfg.smoke.batch_size >= 1
    assert cfg.smoke.epochs >= 1
    assert cfg.smoke.max_pairs >= 1
    assert cfg.smoke.fixture_images >= 1
    assert cfg.smoke.fixture_audio >= 1

    # Smoke batch size should be tiny
    assert cfg.smoke.batch_size <= 8
    # Smoke epochs must be 1 (spec requirement)
    assert cfg.smoke.epochs == 1


def test_smoke_config_fits_cpu_time_budget() -> None:
    """Smoke settings must be small enough to target < 5 min on CPU.

    We verify the fixture sizes directly; actual timing is measured when
    smoke mode is run via the CLI (see README).
    """
    from src.config import load_config

    cfg = load_config()
    # Total images processed in one epoch ≤ fixture_images (no repetition)
    total_image_steps = cfg.smoke.fixture_images // cfg.smoke.batch_size
    # This should be a very small number (< 10)
    assert total_image_steps <= 10, (
        f"Smoke epoch would require {total_image_steps} batches — too many for CPU budget"
    )


def test_run_script_is_importable() -> None:
    """scripts/run.py must be importable without executing side effects."""
    # Add the repo root to sys.path for the import
    import os
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, os.path.abspath(repo_root))

    try:
        import scripts.run as run_module
        # The module must expose a main() callable
        assert callable(getattr(run_module, "main", None)), (
            "scripts/run.py must expose a main() function"
        )
    finally:
        sys.path.pop(0)


def test_run_module_has_smoke_argument() -> None:
    """scripts/run.py must register --smoke in its argument parser."""
    import os

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, os.path.abspath(repo_root))

    try:
        import scripts.run as run_module

        parser = run_module.build_parser()
        # --smoke must be a recognised argument
        args = parser.parse_args(["--smoke"])
        assert args.smoke is True
        # Without the flag, smoke must be False
        args_no_smoke = parser.parse_args([])
        assert args_no_smoke.smoke is False
    finally:
        sys.path.pop(0)
