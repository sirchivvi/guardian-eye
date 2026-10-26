"""GuardianEye data packaging script.

Packages each available dataset folder separately for upload to
Google Drive or a Kaggle Dataset.

Usage
-----
From the repository root::

    python scripts/pack_data.py [OPTIONS]

Or via Makefile::

    make pack-data

Options
-------
--data-root PATH
    Root of the data directory (default: value from config.yaml, then
    DATA_ROOT env var).  The six dataset directories are expected under
    ``<data-root>/raw/``.
--output-dir PATH
    Directory where archives are written
    (default: ``<results-root>/archives/``).
--results-root PATH
    Root results directory (used only when --output-dir is not set).
--config PATH
    Path to a YAML configuration file (default: config.yaml).

Behaviour
---------
- One ``.tar.gz`` archive is created per available, non-empty dataset.
- Missing or empty dataset directories are reported as warnings; the
  script continues with the remaining datasets.
- Archives preserve the internal directory structure.
- The command is safe to rerun; existing archives are overwritten.
- Source data is never deleted or modified.

Storage notes
-------------
Archives can be large (several GB each for image datasets).  Verify
available disk space before running.  Large archives may exceed upload
limits on Google Drive (5 TB per item) or Kaggle Datasets (100 GB total).
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import time
from pathlib import Path

from src.config import load_config
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Six configured dataset directories
DATASET_NAMES = ["birdsai", "cct", "serengeti", "belize", "vietnam", "esc50"]


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for pack_data.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="pack-data",
        description="Package dataset directories into .tar.gz archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", metavar="PATH", default=None)
    parser.add_argument("--results-root", metavar="PATH", default=None)
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory for output archives (default: <results-root>/archives/).",
    )
    parser.add_argument("--config", metavar="PATH", default=None)
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


_IGNORED_FILENAMES: frozenset[str] = frozenset({".gitkeep", ".DS_Store"})


def _is_data_file(path: Path) -> bool:
    """Return True if path is a genuine data file, not a VCS or OS placeholder."""
    return (
        path.is_file()
        and path.name not in _IGNORED_FILENAMES
        and not path.name.startswith("._")
    )


def pack_dataset(
    dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
) -> Path | None:
    """Create a .tar.gz archive for one dataset directory.

    Parameters
    ----------
    dataset_dir:
        Source directory to archive (must exist and contain data files).
    output_dir:
        Destination directory for the archive.
    dataset_name:
        Name used for the archive filename.

    Returns
    -------
    Path | None
        Path to the created archive, or ``None`` if skipped.
    """
    if not dataset_dir.exists():
        logger.warning("SKIP -- directory not found: %s", dataset_dir)
        return None

    # Check non-empty (ignoring .gitkeep and OS metadata files)
    has_files = any(_is_data_file(p) for p in dataset_dir.rglob("*"))
    if not has_files:
        logger.warning("SKIP -- directory is empty: %s", dataset_dir)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{dataset_name}.tar.gz"

    logger.info("Packing %s -> %s", dataset_dir, archive_path)
    start = time.perf_counter()

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if Path(tarinfo.name).name in _IGNORED_FILENAMES:
            return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dataset_dir, arcname=dataset_name, filter=_filter)

    elapsed = time.perf_counter() - start
    size_mb = archive_path.stat().st_size / 1e6
    logger.info(
        "Created %s (%.1f MB, %.1fs)", archive_path.name, size_mb, elapsed
    )
    return archive_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for data packaging.

    Parameters
    ----------
    argv:
        Argument list.

    Returns
    -------
    int
        Exit code (0 on successful execution).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=getattr(logging, args.log_level))

    try:
        cfg = load_config(
            config_path=args.config,
            data_root=args.data_root,
            results_root=args.results_root,
        )
    except FileNotFoundError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    raw_root = cfg.paths.data_root / cfg.paths.raw_subdir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.results_root / "archives"
    )

    logger.info("Data root : %s", raw_root)
    logger.info("Output dir: %s", output_dir)

    created: list[Path] = []
    skipped: list[str] = []

    for name in DATASET_NAMES:
        result = pack_dataset(raw_root / name, output_dir, name)
        if result:
            created.append(result)
        else:
            skipped.append(name)

    logger.info("--- Summary ---")
    logger.info("Created : %d archive(s)", len(created))
    if skipped:
        logger.warning("Skipped : %s (missing or empty)", ", ".join(skipped))

    return 0


if __name__ == "__main__":
    sys.exit(main())
