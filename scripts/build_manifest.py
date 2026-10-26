"""GuardianEye CLI script to scan raw datasets and build per-seed manifests.

Usage
-----
    python scripts/build_manifest.py [OPTIONS]

Generates:
    results/runs/seed_<seed>/manifest/manifest.csv
    results/runs/seed_<seed>/manifest/manifest_status.json
for each configured seed (default: 27, 2701, 26009).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import load_config
from src.logging_config import setup_logging
from src.manifest import ManifestRecord, save_manifest_csv, scan_all_datasets
from src.splits import assign_splits

logger = logging.getLogger(__name__)

NOT_YET_RUN_REASON: str = (
    "No raw dataset files found in data/raw/. Datasets must be placed under "
    "data/raw/<dataset>/ as per Section 6 of GuardianEye_Build_Guide_v2.md."
)


def build_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser for build_manifest."""
    parser = argparse.ArgumentParser(
        prog="build-manifest",
        description="Scan raw datasets, assign stratified splits, and generate manifest artifacts.",
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
        help="Seeds to generate manifests for (default: from config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def build_manifest_for_seed(
    records: list[ManifestRecord],
    unmapped: list[str],
    seed: int,
    manifest_dir: Path,
    splits_cfg: Any,
) -> dict[str, Any]:
    """Assign splits and write manifest.csv and manifest_status.json for a single seed."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = manifest_dir / "manifest.csv"
    status_path = manifest_dir / "manifest_status.json"

    if not records:
        status_data: dict[str, Any] = {
            "seed": seed,
            "status": "NOT YET RUN",
            "reason": NOT_YET_RUN_REASON,
            "total_rows": 0,
            "image_count": 0,
            "audio_count": 0,
            "unmapped_count": len(unmapped),
            "warnings": ["data/raw is empty or unpopulated"],
        }
        save_manifest_csv([], csv_path)
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
        logger.warning("Seed %d: 0 records found -- wrote NOT YET RUN status", seed)
        return status_data

    # Assign splits
    split_records, sparse_classes, warnings = assign_splits(
        records=records,
        seed=seed,
        train_frac=splits_cfg.train,
        val_frac=splits_cfg.val,
        test_frac=splits_cfg.test,
    )

    # Compute actual distributions
    image_count = sum(1 for r in split_records if r.modality == "image")
    audio_count = sum(1 for r in split_records if r.modality == "audio")

    class_dist: dict[str, dict[str, int]] = {"image": {}, "audio": {}}
    for r in split_records:
        class_dist[r.modality][r.label] = class_dist[r.modality].get(r.label, 0) + 1

    split_dist: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for r in split_records:
        split_dist[r.split] = split_dist.get(r.split, 0) + 1

    source_dist: dict[str, int] = {}
    for r in split_records:
        source_dist[r.source] = source_dist.get(r.source, 0) + 1

    status_data = {
        "seed": seed,
        "status": "OK",
        "reason": None,
        "total_rows": len(split_records),
        "image_count": image_count,
        "audio_count": audio_count,
        "unmapped_count": len(unmapped),
        "class_distribution": class_dist,
        "split_distribution": split_dist,
        "source_distribution": source_dist,
        "sparse_classes": sparse_classes,
        "warnings": warnings,
    }

    save_manifest_csv(split_records, csv_path)
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    logger.info("Seed %d: generated manifest (%d rows) -> %s", seed, len(split_records), csv_path)
    return status_data


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for manifest generation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=getattr(logging, args.log_level))

    try:
        cfg = load_config(
            config_path=args.config,
            data_root=args.data_root,
            results_root=args.results_root,
        )
    except FileNotFoundError as e:
        logger.error("Configuration error: %s", e)
        return 1

    seeds = args.seeds if args.seeds else cfg.reproducibility.seeds
    logger.info("Building manifests for seeds: %s", seeds)
    logger.info("Scanning data root: %s", cfg.paths.data_root)

    records, unmapped = scan_all_datasets(cfg.paths.data_root)
    logger.info("Scanned %d valid records, %d unmapped files", len(records), len(unmapped))

    for seed in seeds:
        manifest_dir = cfg.paths.results_root / "runs" / f"seed_{seed}" / "manifest"
        build_manifest_for_seed(
            records=records,
            unmapped=unmapped,
            seed=seed,
            manifest_dir=manifest_dir,
            splits_cfg=cfg.splits,
        )

    logger.info("All manifests generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
