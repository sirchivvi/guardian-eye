"""
CLI script to train GuardianEye image branch models across seeds.

Usage
-----
    python scripts/train_image.py [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config import GuardianEyeConfig, load_config, resolve_paths
from src.evaluation.outputs import extract_image_outputs, save_serialized_outputs
from src.image.evaluate import evaluate_image_model
from src.image.train import train_image_model
from src.logging_config import get_device, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser for train_image."""
    parser = argparse.ArgumentParser(
        prog="train-image",
        description="Train GuardianEye 4-block CNN ImageEncoder with class-weighted cross-entropy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", metavar="PATH", default=None, help="Data root path.")
    parser.add_argument("--results-root", metavar="PATH", default=None, help="Results root path.")
    parser.add_argument("--config", metavar="PATH", default=None, help="Path to config file.")
    parser.add_argument(
        "--seeds",
        metavar="INT",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to train for (default: from config.yaml).",
    )
    parser.add_argument(
        "--epochs",
        metavar="INT",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        metavar="INT",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def write_unimodal_not_yet_run(run_dir: Path, seed: int, reason: str) -> None:
    """Write NOT YET RUN status JSON to run root and outputs folder."""
    status_data = {
        "seed": seed,
        "status": "NOT YET RUN",
        "reason": reason,
        "image_trained": False,
        "audio_trained": False,
        "cross_site_evaluated": False,
    }
    for p in [run_dir / "unimodal_status.json", run_dir / "outputs" / "unimodal_status.json"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)


def run_image_training_for_seed(
    cfg: GuardianEyeConfig,
    seed: int,
    device: torch.device,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> int:
    """Run image branch training and evaluation for a single seed."""
    run_dir = cfg.paths.results_root / "runs" / f"seed_{seed}"
    manifest_csv = run_dir / "manifest" / "manifest.csv"
    manifest_status_file = run_dir / "manifest" / "manifest_status.json"

    # Check Phase 1 manifest status
    if manifest_status_file.exists():
        try:
            with manifest_status_file.open("r", encoding="utf-8") as f:
                manifest_status = json.load(f)
            if manifest_status.get("status") == "NOT YET RUN":
                m_reason = manifest_status.get("reason", "unpopulated")
                reason = f"Phase 1 manifest status is NOT YET RUN: {m_reason}"
                logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
                write_unimodal_not_yet_run(run_dir, seed, reason)
                return 0
        except Exception as exc:
            logger.warning("Could not read %s: %s", manifest_status_file, exc)

    if not manifest_csv.exists() or manifest_csv.stat().st_size == 0:
        reason = "Manifest CSV is missing or empty."
        logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
        write_unimodal_not_yet_run(run_dir, seed, reason)
        return 0

    df = pd.read_csv(manifest_csv)
    image_df = df[df["modality"] == "image"]
    if len(image_df) == 0:
        reason = "Manifest contains 0 image records."
        logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
        write_unimodal_not_yet_run(run_dir, seed, reason)
        return 0

    train_df = image_df[image_df["split"] == "train"]
    val_df = image_df[image_df["split"] == "val"]
    test_df = image_df[image_df["split"] == "test"]

    if len(train_df) == 0:
        reason = "0 image samples assigned to train split."
        logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
        write_unimodal_not_yet_run(run_dir, seed, reason)
        return 0

    checkpoint_dir = run_dir / "checkpoints" / "image_branch"
    outputs_dir = run_dir / "outputs"
    metrics_dir = run_dir / "metrics"

    logger.info("Training ImageEncoder for seed %d (device: %s)", seed, device)
    model, _ = train_image_model(
        train_df=train_df,
        val_df=val_df,
        data_root=cfg.paths.data_root,
        cfg=cfg,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
    )

    # 1. Extract and serialize outputs for EVERY scanned image file
    logger.info("Extracting embeddings for %d image files...", len(image_df))
    image_outputs = extract_image_outputs(
        model=model,
        manifest_df=image_df,
        data_root=cfg.paths.data_root,
        device=device,
        batch_size=batch_size or cfg.image.batch_size,
    )
    save_serialized_outputs(image_outputs, outputs_dir / "image_outputs.pt")

    # 2. Evaluate model strictly on test split
    logger.info("Evaluating ImageEncoder on test split (%d samples)...", len(test_df))
    metrics = evaluate_image_model(
        model=model,
        test_df=test_df,
        data_root=cfg.paths.data_root,
        device=device,
        batch_size=batch_size or cfg.image.batch_size,
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "image_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved image metrics to %s", metrics_file)

    # Update unimodal status
    status_file = run_dir / "unimodal_status.json"
    current_status: dict[str, Any] = {}
    if status_file.exists():
        try:
            with status_file.open("r", encoding="utf-8") as f:
                current_status = json.load(f)
        except Exception:
            pass

    current_status["seed"] = seed
    current_status["status"] = "OK"
    current_status["image_trained"] = True
    for p in [status_file, outputs_dir / "unimodal_status.json"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(current_status, f, indent=2)

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for scripts/train_image.py."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    cfg = load_config(
        config_path=args.config,
        data_root=args.data_root,
        results_root=args.results_root,
    )
    resolve_paths(cfg, data_root=args.data_root, results_root=args.results_root)
    device = get_device()

    seeds = args.seeds or cfg.reproducibility.seeds
    for seed in seeds:
        code = run_image_training_for_seed(
            cfg=cfg,
            seed=seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if code != 0:
            return code

    return 0


if __name__ == "__main__":
    sys.exit(main())
