"""GuardianEye CLI entry point.

Usage
-----
Run the project pipeline from Colab, Kaggle, or locally::

    python -m scripts.run [OPTIONS]

Options
-------
--smoke
    Run in smoke-test mode: uses tiny synthetic fixtures, 1 epoch,
    no real datasets required, targets < 5 minutes on CPU.
--data-root PATH
    Override DATA_ROOT (highest precedence, beats env var and config.yaml).
--results-root PATH
    Override RESULTS_ROOT (highest precedence).
--config PATH
    Path to a YAML config file (default: config.yaml in repo root).
--log-level LEVEL
    Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).

Phase availability
------------------
Phase 0  : configuration, logging, device detection, smoke mode  ✓
Phase 1+ : model training and evaluation — NOT YET AVAILABLE
"""

from __future__ import annotations

import argparse
import logging
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    Separated from ``main()`` so tests can import and interrogate it
    without running the full program.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all supported flags.
    """
    parser = argparse.ArgumentParser(
        prog="guardian-run",
        description="GuardianEye — multi-modal anti-poaching risk assessment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            "Run in CPU smoke-test mode: uses tiny synthetic fixtures, "
            "1 epoch, no GPU or real datasets required. "
            "Target: < 5 minutes on an ordinary CPU."
        ),
    )
    parser.add_argument(
        "--data-root",
        metavar="PATH",
        default=None,
        help="Override data root path (highest precedence).",
    )
    parser.add_argument(
        "--results-root",
        metavar="PATH",
        default=None,
        help="Override results root path (highest precedence).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def _run_smoke(cfg: GuardianEyeConfig) -> None:  # noqa: F821
    """Execute a lightweight CPU smoke test using synthetic fixtures.

    This function:
    1. Logs the active smoke configuration.
    2. Generates tiny synthetic tensors (no real data required).
    3. Confirms that the config, logging, and device detection are working.
    4. Reports elapsed time.

    It does NOT train models, download datasets, or produce real metrics.

    Parameters
    ----------
    cfg:
        Validated GuardianEye configuration object.
    """
    import torch

    from src.logging_config import get_device

    logger = logging.getLogger(__name__)
    logger.info("=== SMOKE TEST MODE ===")
    logger.info(
        "Smoke settings: batch=%d epochs=%d fixture_images=%d fixture_audio=%d",
        cfg.smoke.batch_size,
        cfg.smoke.epochs,
        cfg.smoke.fixture_images,
        cfg.smoke.fixture_audio,
    )

    device = get_device()
    logger.info("Device selected: %s", device)

    start = time.perf_counter()

    # Synthetic image-like tensors  [N, C, H, W]
    fake_images = torch.randn(
        cfg.smoke.fixture_images,
        3,
        *cfg.image.input_size,
    )
    # Synthetic audio-like mel spectrogram tensors [N, 1, Mels, T]
    time_frames = int(
        cfg.audio.duration_sec * cfg.audio.sample_rate / cfg.audio.hop_length
    )
    fake_audio = torch.randn(
        cfg.smoke.fixture_audio,
        1,
        cfg.audio.n_mels,
        time_frames,
    )

    logger.info(
        "Synthetic fixtures created -- images: %s, audio: %s",
        tuple(fake_images.shape),
        tuple(fake_audio.shape),
    )

    # Simulate one epoch of trivial processing (no model, no backprop)
    for epoch in range(cfg.smoke.epochs):
        for i in range(0, cfg.smoke.fixture_images, cfg.smoke.batch_size):
            batch = fake_images[i : i + cfg.smoke.batch_size].to(device)
            _ = batch.mean()  # stand-in for a forward pass

        logger.info("Smoke epoch %d complete", epoch)

    elapsed = time.perf_counter() - start
    budget_sec = 5 * 60  # 5 minutes

    if elapsed < budget_sec:
        logger.info(
            "Smoke test PASSED -- elapsed %.2fs (budget %ds)", elapsed, budget_sec
        )
    else:
        logger.warning(
            "Smoke test EXCEEDED budget -- elapsed %.2fs (budget %ds)",
            elapsed,
            budget_sec,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Argument list.  Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Exit code (0 = success, non-zero = failure).
    """
    from src.config import load_config
    from src.logging_config import setup_logging

    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    try:
        cfg = load_config(
            config_path=args.config,
            data_root=args.data_root,
            results_root=args.results_root,
        )
    except FileNotFoundError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    logger.info(
        "GuardianEye %s loaded -- data_root=%s results_root=%s",
        cfg.project.version,
        cfg.paths.data_root,
        cfg.paths.results_root,
    )

    if args.smoke:
        _run_smoke(cfg)
        return 0

    # -----------------------------------------------------------------------
    # Full pipeline — not yet implemented in Phase 0
    # -----------------------------------------------------------------------
    logger.info(
        "Full pipeline is NOT YET AVAILABLE in Phase 0. "
        "Use 'make train-image', 'make train-audio', etc. in later phases. "
        "Run with --smoke to verify the environment."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
