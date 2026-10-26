"""Unit tests for scripts/pack_data.py.

Covers:
- pack_dataset returns None for missing directory
- pack_dataset returns None for empty directory
- pack_dataset returns None for directory containing only .gitkeep / VCS placeholders
- pack_dataset packages genuine data files into .tar.gz and preserves structure
- pack_dataset excludes placeholder files from archive
- main() safely skips missing/empty datasets and returns 0
- main() packages available fixture datasets without deleting source data
- CLI argument parsing
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from scripts.pack_data import build_parser, main, pack_dataset


def test_pack_dataset_missing_dir_returns_none(tmp_path: Path) -> None:
    """pack_dataset must return None when the dataset directory does not exist."""
    missing_dir = tmp_path / "nonexistent"
    output_dir = tmp_path / "archives"
    result = pack_dataset(missing_dir, output_dir, "nonexistent")
    assert result is None
    assert not output_dir.exists()


def test_pack_dataset_empty_dir_returns_none(tmp_path: Path) -> None:
    """pack_dataset must return None and not create an archive for an empty directory."""
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    output_dir = tmp_path / "archives"
    result = pack_dataset(empty_dir, output_dir, "empty_dataset")
    assert result is None
    assert not (output_dir / "empty_dataset.tar.gz").exists()


def test_pack_dataset_only_gitkeep_returns_none(tmp_path: Path) -> None:
    """pack_dataset must treat a directory with only .gitkeep as empty."""
    dataset_dir = tmp_path / "birdsai"
    dataset_dir.mkdir()
    (dataset_dir / ".gitkeep").touch()

    output_dir = tmp_path / "archives"
    result = pack_dataset(dataset_dir, output_dir, "birdsai")
    assert result is None
    assert not (output_dir / "birdsai.tar.gz").exists()


def test_pack_dataset_with_files_creates_valid_tar(tmp_path: Path) -> None:
    """pack_dataset must create a valid .tar.gz archive and exclude .gitkeep."""
    dataset_dir = tmp_path / "cct"
    sub_dir = dataset_dir / "images"
    sub_dir.mkdir(parents=True)

    (dataset_dir / ".gitkeep").touch()
    (sub_dir / "img001.jpg").write_bytes(b"dummy image content 1")
    (sub_dir / "img002.jpg").write_bytes(b"dummy image content 2")

    output_dir = tmp_path / "archives"
    archive_path = pack_dataset(dataset_dir, output_dir, "cct")

    assert archive_path is not None
    assert archive_path.exists()
    assert archive_path.name == "cct.tar.gz"

    # Inspect archive contents
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
        # Internal structure preserved
        assert any("img001.jpg" in n for n in names)
        assert any("img002.jpg" in n for n in names)
        # .gitkeep excluded
        assert not any(".gitkeep" in n for n in names)


def test_pack_dataset_safe_to_rerun_preserves_source(tmp_path: Path) -> None:
    """Rerunning pack_dataset must overwrite existing archive and preserve source data."""
    dataset_dir = tmp_path / "esc50"
    dataset_dir.mkdir()
    sample_file = dataset_dir / "sample.wav"
    sample_file.write_bytes(b"sample audio")

    output_dir = tmp_path / "archives"

    # First run
    path1 = pack_dataset(dataset_dir, output_dir, "esc50")
    assert path1 is not None and path1.exists()

    # Second run (safe to rerun)
    path2 = pack_dataset(dataset_dir, output_dir, "esc50")
    assert path2 == path1
    assert path2.exists()

    # Source data remains untouched
    assert sample_file.exists()
    assert sample_file.read_bytes() == b"sample audio"


def test_pack_data_main_all_empty(tmp_path: Path) -> None:
    """main() must report skipped datasets safely and return 0 when all are empty."""
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True)

    for name in ["birdsai", "cct", "serengeti", "belize", "vietnam", "esc50"]:
        d = raw_root / name
        d.mkdir()
        (d / ".gitkeep").touch()

    output_dir = tmp_path / "archives"
    ret = main(["--data-root", str(data_root), "--output-dir", str(output_dir)])
    assert ret == 0
    assert not output_dir.exists() or len(list(output_dir.glob("*.tar.gz"))) == 0


def test_pack_data_main_with_fixture(tmp_path: Path) -> None:
    """main() creates an archive for available fixture dataset and skips empty ones."""
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True)

    # Populate only birdsai
    birdsai_dir = raw_root / "birdsai"
    birdsai_dir.mkdir()
    (birdsai_dir / "sample.txt").write_text("birdsai data")

    # Leave other 5 empty with .gitkeep
    for name in ["cct", "serengeti", "belize", "vietnam", "esc50"]:
        d = raw_root / name
        d.mkdir()
        (d / ".gitkeep").touch()

    output_dir = tmp_path / "archives"
    ret = main(["--data-root", str(data_root), "--output-dir", str(output_dir)])
    assert ret == 0

    created = list(output_dir.glob("*.tar.gz"))
    assert len(created) == 1
    assert created[0].name == "birdsai.tar.gz"


def test_build_parser_options() -> None:
    """Argument parser must configure all expected options."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--data-root",
            "custom_data",
            "--results-root",
            "custom_results",
            "--output-dir",
            "custom_out",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.data_root == "custom_data"
    assert args.results_root == "custom_results"
    assert args.output_dir == "custom_out"
    assert args.log_level == "DEBUG"
