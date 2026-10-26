"""
CLI script to train GuardianEye audio branch models across seeds.

Usage
-----
    python scripts/train_audio.py [OPTIONS]
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

from src.audio.evaluate import evaluate_audio_branch
from src.audio.features import AudioProcessor
from src.audio.train import train_cross_site_audio, train_in_domain_audio
from src.config import GuardianEyeConfig, load_config, resolve_paths
from src.evaluation.outputs import extract_audio_outputs, save_serialized_outputs
from src.logging_config import get_device, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser for train_audio."""
    parser = argparse.ArgumentParser(
        prog="train-audio",
        description=(
            "Train GuardianEye AudioEncoder on in-domain pooled data and "
            "leakage-free cross-site sets."
        ),
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


def run_audio_training_for_seed(
    cfg: GuardianEyeConfig,
    seed: int,
    device: torch.device,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> int:
    """Run audio branch in-domain and cross-site training for a single seed."""
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
    audio_df = df[df["modality"] == "audio"]
    if len(audio_df) == 0:
        reason = "Manifest contains 0 audio records."
        logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
        write_unimodal_not_yet_run(run_dir, seed, reason)
        return 0

    # In-domain training pool (Belize + Vietnam, non-thunder)
    pool_train_df = audio_df[
        (audio_df["noise_tag"] != "thunder_wind")
        & (audio_df["source"].isin(["belize", "vietnam"]))
        & (audio_df["split"] == "train")
    ]

    if len(pool_train_df) == 0:
        reason = "0 in-domain audio samples assigned to train split."
        logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
        write_unimodal_not_yet_run(run_dir, seed, reason)
        return 0

    checkpoint_dir = run_dir / "checkpoints" / "audio_branch"
    cross_site_dir = run_dir / "checkpoints" / "cross_site"
    outputs_dir = run_dir / "outputs"
    metrics_dir = run_dir / "metrics"

    processor = AudioProcessor(
        sample_rate=cfg.audio.sample_rate,
        target_samples=int(cfg.audio.sample_rate * cfg.audio.duration_sec),
        n_mels=cfg.audio.n_mels,
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
    )

    # 1. Train in-domain audio model
    logger.info("Training in-domain AudioEncoder for seed %d...", seed)
    in_domain_model, _ = train_in_domain_audio(
        manifest_df=audio_df,
        data_root=cfg.paths.data_root,
        cfg=cfg,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        processor=processor,
        epochs=epochs,
        batch_size=batch_size,
    )

    # 2. Train cross-site models (leakage-free)
    cross_site_models: dict[str, torch.nn.Module] = {}
    belize_train_count = len(
        audio_df[
            (audio_df["source"] == "belize")
            & (audio_df["split"] == "train")
            & (audio_df["noise_tag"] != "thunder_wind")
        ]
    )
    vietnam_train_count = len(
        audio_df[
            (audio_df["source"] == "vietnam")
            & (audio_df["split"] == "train")
            & (audio_df["noise_tag"] != "thunder_wind")
        ]
    )

    if belize_train_count > 0:
        logger.info("Training cross-site model: Belize -> Vietnam...")
        b2v_model, _ = train_cross_site_audio(
            manifest_df=audio_df,
            source_site="belize",
            data_root=cfg.paths.data_root,
            cfg=cfg,
            seed=seed,
            device=device,
            checkpoint_dir=cross_site_dir,
            processor=processor,
            epochs=epochs,
            batch_size=batch_size,
        )
        cross_site_models["belize_to_vietnam"] = b2v_model

    if vietnam_train_count > 0:
        logger.info("Training cross-site model: Vietnam -> Belize...")
        v2b_model, _ = train_cross_site_audio(
            manifest_df=audio_df,
            source_site="vietnam",
            data_root=cfg.paths.data_root,
            cfg=cfg,
            seed=seed,
            device=device,
            checkpoint_dir=cross_site_dir,
            processor=processor,
            epochs=epochs,
            batch_size=batch_size,
        )
        cross_site_models["vietnam_to_belize"] = v2b_model

    # 3. Extract and serialize outputs for EVERY scanned audio file (including held-out thunder)
    logger.info("Extracting embeddings for %d audio files...", len(audio_df))
    audio_outputs = extract_audio_outputs(
        model=in_domain_model,
        manifest_df=audio_df,
        data_root=cfg.paths.data_root,
        device=device,
        processor=processor,
        batch_size=batch_size or cfg.audio.batch_size,
    )
    save_serialized_outputs(audio_outputs, outputs_dir / "audio_outputs.pt")

    # 4. Evaluate in-domain and cross-site models strictly on test split
    logger.info("Evaluating AudioEncoder on test splits...")
    metrics = evaluate_audio_branch(
        in_domain_model=in_domain_model,
        manifest_df=audio_df,
        data_root=cfg.paths.data_root,
        device=device,
        processor=processor,
        cross_site_models=cross_site_models,
        batch_size=batch_size or cfg.audio.batch_size,
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "audio_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved audio metrics to %s", metrics_file)

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
    current_status["audio_trained"] = True
    current_status["cross_site_evaluated"] = len(cross_site_models) > 0
    for p in [status_file, outputs_dir / "unimodal_status.json"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(current_status, f, indent=2)

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for scripts/train_audio.py."""
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
        code = run_audio_training_for_seed(
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
