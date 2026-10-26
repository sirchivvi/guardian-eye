"""
GuardianEye CLI script to generate reproducible synthetic multimodal pairings.

Usage
-----
    python scripts/run_pairing.py [OPTIONS]

Generates per configured seed:
    results/runs/seed_<seed>/pairing/scenarios.csv
    results/runs/seed_<seed>/pairing/pairs.csv
    results/runs/seed_<seed>/pairing/pairing_status.json
    results/runs/seed_<seed>/pairing/pairing_self_check.json

Optional --self-check flag runs an in-memory fixture validation and writes:
    results/protocol/pairing_self_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import GuardianEyeConfig, load_config
from src.logging_config import setup_logging
from src.pairing import (
    generate_all_pairs,
    records_from_dataframe,
    run_pairing_self_check,
    save_pairs_csv,
    save_scenarios_csv,
    verify_pairs,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser for run_pairing."""
    parser = argparse.ArgumentParser(
        prog="run-pairing",
        description="Generate reproducible synthetic image-audio pairings per split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", metavar="PATH", default=None, help="Data root path.")
    parser.add_argument(
        "--results-root", metavar="PATH", default=None, help="Results root path."
    )
    parser.add_argument("--config", metavar="PATH", default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--seeds",
        metavar="INT",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to generate pairings for (default: from config.yaml).",
    )
    parser.add_argument(
        "--max-pairs-per-scenario",
        type=int,
        default=None,
        help="Maximum pairs per scenario per split (default: from config.yaml).",
    )
    parser.add_argument(
        "--max-pairs-per-split",
        type=int,
        default=None,
        help="Maximum total pairs per split (default: from config.yaml).",
    )
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=None,
        help="Minimum images required before a class is included.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run in-memory self-check and write results/protocol/pairing_self_check.json.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode with minimal pair limits for fast CPU testing.",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def write_pairing_not_yet_run(
    pairing_dir: Path,
    seed: int,
    reason: str,
) -> None:
    """
    Write NOT YET RUN status and empty schema-compliant CSVs.

    Parameters
    ----------
    pairing_dir:
        Directory results/runs/seed_<seed>/pairing/.
    seed:
        Experiment seed.
    reason:
        Explanation for NOT YET RUN status.

    """
    pairing_dir.mkdir(parents=True, exist_ok=True)
    status_file = pairing_dir / "pairing_status.json"
    status_data = {
        "seed": seed,
        "status": "NOT YET RUN",
        "reason": reason,
        "total_pairs": 0,
        "scenario_totals": {},
        "per_split": {},
    }
    with status_file.open("w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    # Write schema-compliant empty CSVs
    save_scenarios_csv([], pairing_dir / "scenarios.csv")
    save_pairs_csv([], pairing_dir / "pairs.csv")
    logger.info("Seed %d: Wrote NOT YET RUN status to %s", seed, status_file)


def run_pairing_for_seed(
    cfg: GuardianEyeConfig,
    seed: int,
    max_pairs_per_scenario: int,
    max_pairs_per_split: int,
    min_images_per_class: int,
) -> dict[str, Any]:
    """
    Run pairing for a single seed.

    Parameters
    ----------
    cfg:
        Loaded project configuration.
    seed:
        Target seed.
    max_pairs_per_scenario:
        Per-scenario pair limit.
    max_pairs_per_split:
        Per-split total pair limit.
    min_images_per_class:
        Minimum images per class threshold.

    Returns
    -------
    dict[str, Any]
        Pairing execution status and stats.

    """
    run_dir = cfg.paths.results_root / "runs" / f"seed_{seed}"
    manifest_csv = run_dir / "manifest" / "manifest.csv"
    manifest_status_file = run_dir / "manifest" / "manifest_status.json"
    pairing_dir = run_dir / "pairing"
    pairing_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check Phase 1 manifest status
    if manifest_status_file.exists():
        try:
            with manifest_status_file.open("r", encoding="utf-8") as f:
                manifest_status = json.load(f)
            if manifest_status.get("status") == "NOT YET RUN":
                m_reason = manifest_status.get("reason", "unpopulated")
                reason = f"Phase 1 manifest status is NOT YET RUN: {m_reason}"
                logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
                write_pairing_not_yet_run(pairing_dir, seed, reason)
                return {"seed": seed, "status": "NOT YET RUN", "reason": reason}
        except Exception as exc:
            logger.warning("Could not read %s: %s", manifest_status_file, exc)

    # 2. Check manifest.csv
    if not manifest_csv.exists() or manifest_csv.stat().st_size == 0:
        reason = "Manifest CSV is missing or empty."
        logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
        write_pairing_not_yet_run(pairing_dir, seed, reason)
        return {"seed": seed, "status": "NOT YET RUN", "reason": reason}

    df = pd.read_csv(manifest_csv)
    if len(df) == 0:
        reason = "Manifest contains 0 records."
        logger.warning("Seed %d: %s. Exiting cleanly.", seed, reason)
        write_pairing_not_yet_run(pairing_dir, seed, reason)
        return {"seed": seed, "status": "NOT YET RUN", "reason": reason}

    # Convert to ManifestRecord objects
    records = records_from_dataframe(df)

    # 3. Generate all pairs across train, val, test
    pairs, stats = generate_all_pairs(
        records=records,
        seed=seed,
        splits=["train", "val", "test"],
        max_pairs_per_scenario=max_pairs_per_scenario,
        max_pairs_per_split=max_pairs_per_split,
        min_images_per_class=min_images_per_class,
    )

    # 4. Save CSVs
    scenarios_csv = pairing_dir / "scenarios.csv"
    pairs_csv = pairing_dir / "pairs.csv"
    save_scenarios_csv(pairs, scenarios_csv)
    save_pairs_csv(pairs, pairs_csv)

    # 5. Audit pairs and save self-check report
    audit = verify_pairs(pairs, manifest_records=records)
    self_check_file = pairing_dir / "pairing_self_check.json"
    self_check_report = {
        "seed": seed,
        "status": audit["status"],
        "not_a_dataset_evaluation": True,
        "total_pairs": len(pairs),
        "audit": audit,
        "per_split": stats["per_split"],
        "scenario_totals": stats["scenario_totals"],
        "empty_scenarios": stats["empty_scenarios_overall"],
    }
    with self_check_file.open("w", encoding="utf-8") as f:
        json.dump(self_check_report, f, indent=2)

    # 6. Save pairing status JSON
    status_file = pairing_dir / "pairing_status.json"
    status_data = {
        "seed": seed,
        "status": "COMPLETE",
        "total_pairs": len(pairs),
        "per_split": stats["per_split"],
        "scenario_totals": stats["scenario_totals"],
        "empty_scenarios": stats["empty_scenarios_overall"],
        "scenarios_csv": str(scenarios_csv).replace("\\", "/"),
        "pairs_csv": str(pairs_csv).replace("\\", "/"),
        "audit_status": audit["status"],
    }
    with status_file.open("w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    logger.info(
        "Seed %d: Generated %d pairs across %d scenarios. Status: COMPLETE",
        seed,
        len(pairs),
        len(stats["scenario_totals"]),
    )
    return status_data


def main() -> int:
    """CLI entry point for run_pairing."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    cfg = load_config(
        config_path=args.config,
        data_root=args.data_root,
        results_root=args.results_root,
    )

    # If --self-check is requested, run in-memory self-check and exit
    if args.self_check:
        protocol_path = cfg.paths.results_root / "protocol" / "pairing_self_check.json"
        report = run_pairing_self_check(
            seed=cfg.reproducibility.seeds[0], output_path=protocol_path
        )
        logger.info("Self-check completed: status = %s", report["status"])
        return 0 if report["status"] == "PASS" else 1

    # Determine limits
    if args.smoke:
        max_scenario = min(10, cfg.smoke.max_pairs)
        max_split = cfg.smoke.max_pairs
        min_images = 0
    else:
        max_scenario = (
            args.max_pairs_per_scenario
            if args.max_pairs_per_scenario is not None
            else getattr(cfg.pairs, "max_pairs_per_scenario", 1500)
        )
        max_split = (
            args.max_pairs_per_split
            if args.max_pairs_per_split is not None
            else cfg.pairs.max_pairs_per_split
        )
        min_images = (
            args.min_images_per_class
            if args.min_images_per_class is not None
            else cfg.pairs.min_images_per_class
        )

    seeds = args.seeds if args.seeds is not None else cfg.reproducibility.seeds

    logger.info(
        "Running pairing for seeds %s (max_per_scenario=%d, max_per_split=%d, min_images=%d)",
        seeds,
        max_scenario,
        max_split,
        min_images,
    )

    for seed in seeds:
        run_pairing_for_seed(
            cfg=cfg,
            seed=seed,
            max_pairs_per_scenario=max_scenario,
            max_pairs_per_split=max_split,
            min_images_per_class=min_images,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
