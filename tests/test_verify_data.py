"""Unit tests for scripts/verify_data.py.

Covers:
- verify_dataset returns MISSING or FAIL (0 files) for empty folders
- verify_dataset probes valid images and audio clips and returns PASS
- verify_dataset detects corrupted files and reports corruption count
- main() runs without crashing and prints table
"""

from __future__ import annotations

import wave
from pathlib import Path

from scripts.verify_data import main, verify_dataset


def test_verify_dataset_missing_folder(tmp_path: Path) -> None:
    """Missing dataset folder is marked MISSING."""
    res = verify_dataset(tmp_path / "nonexistent", "image")
    assert res["status"] == "MISSING"
    assert res["found_files"] == 0


def test_verify_dataset_empty_folder(tmp_path: Path) -> None:
    """Folder containing only .gitkeep is marked FAIL (0 files)."""
    empty_dir = tmp_path / "empty_set"
    empty_dir.mkdir()
    (empty_dir / ".gitkeep").touch()

    res = verify_dataset(empty_dir, "image")
    assert "FAIL" in res["status"]
    assert res["found_files"] == 0


def test_verify_dataset_valid_images_pass(tmp_path: Path) -> None:
    """Folder with valid images produces PASS status."""
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(5):
        im = Image.new("RGB", (64, 64), color="blue")
        im.save(img_dir / f"img_{i}.jpg")

    res = verify_dataset(img_dir, "image", sample_size=10)
    assert res["status"] == "PASS"
    assert res["found_files"] == 5
    assert res["valid_files"] == 5
    assert res["corrupt_files"] == 0


def test_verify_dataset_valid_audio_pass(tmp_path: Path) -> None:
    """Folder with valid WAV audio produces PASS status."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    for i in range(3):
        wav_path = audio_dir / f"clip_{i}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(b"\x00\x00" * 800)

    res = verify_dataset(audio_dir, "audio", sample_size=5)
    assert res["status"] == "PASS"
    assert res["found_files"] == 3
    assert res["valid_files"] == 3


def test_verify_dataset_corrupt_files(tmp_path: Path) -> None:
    """Corrupted files are detected and reported."""
    img_dir = tmp_path / "corrupt_images"
    img_dir.mkdir()

    (img_dir / "bad1.jpg").write_bytes(b"not an image header at all")

    res = verify_dataset(img_dir, "image")
    assert "FAIL" in res["status"]
    assert res["corrupt_files"] == 1


def test_verify_data_main_cli(tmp_path: Path) -> None:
    """main() prints table and exits cleanly."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    for d in ["birdsai", "cct", "serengeti", "belize", "vietnam", "esc50"]:
        (raw_dir / d).mkdir()

    ret = main(["--data-root", str(tmp_path / "data")])
    assert ret == 0
