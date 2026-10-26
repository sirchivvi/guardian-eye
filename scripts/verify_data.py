"""GuardianEye dataset verification and file integrity diagnostic script.

Usage
-----
    python scripts/verify_data.py [OPTIONS]

Counts files across all six datasets in data/raw/, probes up to 20 random files
per dataset for decoder validity, and outputs a clean PASS/FAIL diagnostic table.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import wave
from pathlib import Path
from typing import Any

from src.config import load_config
from src.logging_config import setup_logging
from src.manifest import AUDIO_EXTENSIONS, IGNORED_FILENAMES, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

DATASET_CONFIG = [
    ("birdsai", "image", "aerial_thermal"),
    ("cct", "image", "camera_trap"),
    ("serengeti", "image", "camera_trap"),
    ("belize", "audio", "acoustic"),
    ("vietnam", "audio", "acoustic"),
    ("esc50", "audio", "acoustic"),
]


def probe_file(file_path: Path, modality: str) -> bool:
    """Attempt to decode/validate a single file without loading full dataset."""
    if modality == "image":
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    elif modality == "audio":
        try:
            if file_path.suffix.lower() == ".wav":
                with wave.open(str(file_path), "rb") as wf:
                    return wf.getnframes() > 0
            # Fallback for flac/ogg/mp3
            return file_path.stat().st_size > 128
        except Exception:
            return False
    return False


def verify_dataset(dataset_dir: Path, modality: str, sample_size: int = 20) -> dict[str, Any]:
    """Inspect and sample a single dataset folder."""
    if not dataset_dir.is_dir():
        return {
            "found_files": 0,
            "probed_files": 0,
            "valid_files": 0,
            "corrupt_files": 0,
            "status": "MISSING",
        }

    exts = IMAGE_EXTENSIONS if modality == "image" else AUDIO_EXTENSIONS
    all_files = [
        p
        for p in dataset_dir.rglob("*")
        if p.is_file() and p.name not in IGNORED_FILENAMES and p.suffix.lower() in exts
    ]

    total_count = len(all_files)
    if total_count == 0:
        return {
            "found_files": 0,
            "probed_files": 0,
            "valid_files": 0,
            "corrupt_files": 0,
            "status": "FAIL (0 files)",
        }

    rng = random.Random(42)
    sample = rng.sample(all_files, min(sample_size, total_count))
    valid_count = 0
    corrupt_count = 0

    for f in sample:
        if probe_file(f, modality):
            valid_count += 1
        else:
            corrupt_count += 1

    status = "PASS" if corrupt_count == 0 else f"FAIL ({corrupt_count} corrupt)"
    return {
        "found_files": total_count,
        "probed_files": len(sample),
        "valid_files": valid_count,
        "corrupt_files": corrupt_count,
        "status": status,
    }


def build_parser() -> argparse.ArgumentParser:
    """Return CLI parser for verify_data."""
    parser = argparse.ArgumentParser(
        prog="verify-data",
        description="Verify raw dataset files and file integrity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", metavar="PATH", default=None, help="Data root path.")
    parser.add_argument("--config", metavar="PATH", default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--sample-size",
        metavar="N",
        type=int,
        default=20,
        help="Number of random files to probe per dataset (default: 20).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with non-zero code if any dataset is missing or fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dataset verification."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging()

    try:
        cfg = load_config(config_path=args.config, data_root=args.data_root)
    except FileNotFoundError as e:
        logger.error("Config error: %s", e)
        return 1

    raw_root = cfg.paths.data_root / cfg.paths.raw_subdir
    print(f"\nGuardianEye Data Verification Report (Data Root: {raw_root})")
    print("=" * 86)
    print(
        f"{'Dataset':<14} | {'Modality':<8} | {'Found':<8} | {'Probed':<8} | "
        f"{'Valid':<8} | {'Corrupt':<8} | {'Status':<12}"
    )
    print("-" * 86)

    all_pass = True
    for name, modality, _view in DATASET_CONFIG:
        dir_path = raw_root / name
        res = verify_dataset(dir_path, modality, sample_size=args.sample_size)
        print(
            f"{name:<14} | {modality:<8} | {res['found_files']:<8} | {res['probed_files']:<8} | "
            f"{res['valid_files']:<8} | {res['corrupt_files']:<8} | {res['status']:<12}"
        )
        if res["status"] != "PASS":
            all_pass = False

    print("=" * 86)
    if not all_pass:
        print("Note: Empty/missing datasets reflect the NOT YET RUN status prior to download.")
    else:
        print("All probed datasets verified successfully.")

    return 1 if (args.strict and not all_pass) else 0


if __name__ == "__main__":
    sys.exit(main())
