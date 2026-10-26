"""
CLI script to evaluate GuardianEye unimodal models across seeds.

Usage
-----
    python scripts/evaluate_unimodal.py [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import pandas as pd
import torch

from src.audio.evaluate import evaluate_audio_branch
from src.audio.features import AudioProcessor
from src.checkpoint import load_checkpoint
from src.config import GuardianEyeConfig, load_config, resolve_paths
from src.image.evaluate import evaluate_image_model
from src.logging_config import get_device, setup_logging
from src.models import AudioEncoder, ImageEncoder

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser for evaluate_unimodal."""
    parser = argparse.ArgumentParser(
        prog="evaluate-unimodal",
        description="Evaluate GuardianEye trained unimodal models on test splits.",
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
        help="Seeds to evaluate for (default: from config.yaml).",
    )
    parser.add_argument(
        "--batch-size",
        metavar="INT",
        type=int,
        default=None,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def evaluate_unimodal_for_seed(
    cfg: GuardianEyeConfig,
    seed: int,
    device: torch.device,
    batch_size: int | None = None,
) -> int:
    """Run evaluation for a single seed."""
    run_dir = cfg.paths.results_root / "runs" / f"seed_{seed}"
    manifest_csv = run_dir / "manifest" / "manifest.csv"
    manifest_status_file = run_dir / "manifest" / "manifest_status.json"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Check manifest status
    if manifest_status_file.exists():
        try:
            with manifest_status_file.open("r", encoding="utf-8") as f:
                manifest_status = json.load(f)
            if manifest_status.get("status") == "NOT YET RUN":
                reason = manifest_status.get("reason", "unpopulated raw data")
                logger.warning("Seed %d: Manifest is NOT YET RUN (%s).", seed, reason)
                summary_data = {
                    "seed": seed,
                    "status": "NOT YET RUN",
                    "reason": reason,
                    "image": {"status": "NOT YET RUN"},
                    "audio": {"status": "NOT YET RUN"},
                }
                with (metrics_dir / "unimodal_summary.json").open("w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2)
                return 0
        except Exception as exc:
            logger.warning("Could not read %s: %s", manifest_status_file, exc)

    if not manifest_csv.exists():
        logger.warning("Seed %d: Manifest CSV not found. Skipping evaluation.", seed)
        return 0

    df = pd.read_csv(manifest_csv)
    bs = batch_size or 32

    # 1. Image evaluation
    image_metrics: dict[str, Any] = {
        "status": "NOT YET RUN",
        "reason": "Model checkpoint not found.",
    }
    image_ckpt = run_dir / "checkpoints" / "image_branch" / "best_model.pt"
    if image_ckpt.exists():
        image_model = ImageEncoder(
            num_classes=cfg.image.num_classes,
            embedding_dim=cfg.image.embedding_dim,
            use_pretrained=cfg.image.pretrained,
        ).to(device)
        state = load_checkpoint(image_ckpt)
        image_model.load_state_dict(state["model_state"])

        test_img_df = df[(df["modality"] == "image") & (df["split"] == "test")]
        image_metrics = evaluate_image_model(
            model=image_model,
            test_df=test_img_df,
            data_root=cfg.paths.data_root,
            device=device,
            batch_size=bs,
        )

    # 2. Audio evaluation
    audio_metrics: dict[str, Any] = {
        "status": "NOT YET RUN",
        "reason": "Model checkpoint not found.",
    }
    audio_ckpt = run_dir / "checkpoints" / "audio_branch" / "best_model.pt"
    if audio_ckpt.exists():
        audio_model = AudioEncoder(
            num_classes=cfg.audio.num_classes,
            embedding_dim=cfg.audio.embedding_dim,
            use_pretrained=cfg.audio.pretrained,
        ).to(device)
        state = load_checkpoint(audio_ckpt)
        audio_model.load_state_dict(state["model_state"])

        processor = AudioProcessor(
            sample_rate=cfg.audio.sample_rate,
            target_samples=int(cfg.audio.sample_rate * cfg.audio.duration_sec),
            n_mels=cfg.audio.n_mels,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
        )

        cross_site_models: dict[str, torch.nn.Module] = {}
        b2v_ckpt = run_dir / "checkpoints" / "cross_site" / "cross_site_belize_best.pt"
        if b2v_ckpt.exists():
            b2v = AudioEncoder(num_classes=2, embedding_dim=128).to(device)
            b2v.load_state_dict(load_checkpoint(b2v_ckpt)["model_state"])
            cross_site_models["belize_to_vietnam"] = b2v

        v2b_ckpt = run_dir / "checkpoints" / "cross_site" / "cross_site_vietnam_best.pt"
        if v2b_ckpt.exists():
            v2b = AudioEncoder(num_classes=2, embedding_dim=128).to(device)
            v2b.load_state_dict(load_checkpoint(v2b_ckpt)["model_state"])
            cross_site_models["vietnam_to_belize"] = v2b

        audio_metrics = evaluate_audio_branch(
            in_domain_model=audio_model,
            manifest_df=df[df["modality"] == "audio"],
            data_root=cfg.paths.data_root,
            device=device,
            processor=processor,
            cross_site_models=cross_site_models,
            batch_size=bs,
        )

    summary = {
        "seed": seed,
        "status": "OK",
        "image": image_metrics,
        "audio": audio_metrics,
    }
    summary_path = metrics_dir / "unimodal_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved unimodal summary to %s", summary_path)

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for scripts/evaluate_unimodal.py."""
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
        code = evaluate_unimodal_for_seed(
            cfg=cfg,
            seed=seed,
            device=device,
            batch_size=args.batch_size,
        )
        if code != 0:
            return code

    return 0


if __name__ == "__main__":
    sys.exit(main())
