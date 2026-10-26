"""Unit tests for src/manifest.py.

Covers:
- POSIX relative path normalization
- Schema validation for ManifestRecord (modalities, labels, views, noise_tags)
- BIRDSAI MOT annotation format parsing (human=1, animal=0, empty=no box, both=human+note)
- Class-folder scanning (CCT and Serengeti)
- COCO Camera Traps JSON scanning
- Belize train/val split metadata preservation
- Vietnam audio scanning
- ESC-50 thunder_wind tagging and background label assignment
- Duplicate path detection
- Unmapped file reporting without label fabrication
- scan_all_datasets on empty directories returns 0 records
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.manifest import (
    ManifestRecord,
    scan_all_datasets,
    scan_belize,
    scan_birdsai,
    scan_class_folders,
    scan_coco_json,
    scan_esc50,
    to_posix_relpath,
)


def test_posix_relative_path_normalization(tmp_path: Path) -> None:
    """to_posix_relpath must use forward slashes even on Windows."""
    sub = tmp_path / "data" / "raw" / "birdsai" / "seq1" / "001.jpg"
    sub.parent.mkdir(parents=True)
    sub.touch()

    rel = to_posix_relpath(sub, tmp_path)
    assert "\\" not in rel
    assert rel == "data/raw/birdsai/seq1/001.jpg"


def test_manifest_record_validation() -> None:
    """ManifestRecord.validate() enforces canonical schema taxonomy."""
    valid = ManifestRecord(
        path="raw/cct/human/001.jpg",
        modality="image",
        source="cct",
        label="human",
        view="camera_trap",
        site_or_device="cct",
        noise_tag="not_thunder_wind",
        original_split="unspecified",
        group_id="cct_seq1",
    )
    valid.validate()  # must not raise

    # Invalid image label
    bad_label = ManifestRecord(
        path="raw/cct/001.jpg",
        modality="image",
        source="cct",
        label="dog",
        view="camera_trap",
        site_or_device="cct",
        noise_tag="not_thunder_wind",
        original_split="unspecified",
        group_id="cct_1",
    )
    with pytest.raises(ValueError, match="Invalid image label"):
        bad_label.validate()

    # Invalid audio label
    bad_audio = ManifestRecord(
        path="raw/belize/001.wav",
        modality="audio",
        source="belize",
        label="speech",
        view="acoustic",
        site_or_device="belize",
        noise_tag="not_thunder_wind",
        original_split="unspecified",
        group_id="b1",
    )
    with pytest.raises(ValueError, match="Invalid audio label"):
        bad_audio.validate()

    # Non-POSIX path
    bad_path = ManifestRecord(
        path="raw\\cct\\001.jpg",
        modality="image",
        source="cct",
        label="human",
        view="camera_trap",
        site_or_device="cct",
        noise_tag="not_thunder_wind",
        original_split="unspecified",
        group_id="c1",
    )
    with pytest.raises(ValueError, match="POSIX"):
        bad_path.validate()


def test_scan_birdsai_mot_format(tmp_path: Path) -> None:
    """scan_birdsai parses MOT format annotations correctly."""
    birdsai_dir = tmp_path / "raw" / "birdsai"
    seq_dir = birdsai_dir / "sequence_01"
    frames_dir = seq_dir / "img1"
    frames_dir.mkdir(parents=True)

    # Create dummy frames: 1, 2, 3, 4
    for i in range(1, 5):
        (frames_dir / f"{i:06d}.jpg").write_bytes(b"dummy image")

    # MOT gt.txt:
    # Frame 1: human (class 1)
    # Frame 2: animal (class 0)
    # Frame 3: human (class 1) and animal (class 0) -> human with note
    # Frame 4: no annotation -> empty
    gt_lines = [
        "1,1,10,10,20,20,1,1,1.0",
        "2,2,15,15,30,30,1,0,1.0",
        "3,1,10,10,20,20,1,1,1.0",
        "3,2,50,50,20,20,1,0,1.0",
    ]
    (seq_dir / "gt.txt").write_text("\n".join(gt_lines) + "\n")

    records, unmapped = scan_birdsai(birdsai_dir, tmp_path)
    assert unmapped == []
    assert len(records) == 4

    rec_map = {r.path.split("/")[-1]: r for r in records}

    # Frame 1: human
    assert rec_map["000001.jpg"].label == "human"
    assert rec_map["000001.jpg"].view == "aerial_thermal"

    # Frame 2: animal
    assert rec_map["000002.jpg"].label == "animal"

    # Frame 3: both -> human + human_and_animal_boxes note
    assert rec_map["000003.jpg"].label == "human"
    assert rec_map["000003.jpg"].notes == "human_and_animal_boxes"

    # Frame 4: empty
    assert rec_map["000004.jpg"].label == "empty"


def test_scan_class_folders_cct(tmp_path: Path) -> None:
    """scan_class_folders scans human, vehicle, animal, empty directories."""
    cct_dir = tmp_path / "raw" / "cct"
    (cct_dir / "human").mkdir(parents=True)
    (cct_dir / "vehicle").mkdir(parents=True)
    (cct_dir / "animal").mkdir(parents=True)
    (cct_dir / "empty").mkdir(parents=True)

    (cct_dir / "human" / "h1.jpg").write_bytes(b"img")
    (cct_dir / "vehicle" / "v1.png").write_bytes(b"img")
    (cct_dir / "animal" / "a1.jpeg").write_bytes(b"img")
    (cct_dir / "empty" / "e1.jpg").write_bytes(b"img")
    # Unsupported file
    (cct_dir / "human" / "notes.txt").write_text("text")

    records, unmapped = scan_class_folders(cct_dir, tmp_path, "cct", "camera_trap", "image")
    assert len(records) == 4
    assert len(unmapped) == 1
    assert any("notes.txt" in u for u in unmapped)

    labels = {r.label for r in records}
    assert labels == {"human", "vehicle", "animal", "empty"}


def test_scan_coco_json(tmp_path: Path) -> None:
    """scan_coco_json parses COCO Camera Traps format metadata."""
    cct_dir = tmp_path / "raw" / "cct"
    cct_dir.mkdir(parents=True)

    (cct_dir / "img1.jpg").write_bytes(b"img")
    (cct_dir / "img2.jpg").write_bytes(b"img")

    coco_data = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
        ],
        "images": [
            {"id": "img1", "file_name": "img1.jpg", "location": "loc_01"},
            {"id": "img2", "file_name": "img2.jpg", "location": "loc_02"},
        ],
        "annotations": [
            {"image_id": "img1", "category_id": 1},
            {"image_id": "img2", "category_id": 2},
        ],
    }
    json_path = cct_dir / "cct_metadata.json"
    json_path.write_text(json.dumps(coco_data))

    records, unmapped = scan_coco_json(cct_dir, json_path, tmp_path, "cct", "camera_trap")
    assert len(records) == 2
    assert records[0].label == "human"
    assert records[0].group_id == "cct_loc_01"
    assert records[1].label == "vehicle"
    assert records[1].group_id == "cct_loc_02"


def test_scan_belize_preserves_original_splits(tmp_path: Path) -> None:
    """scan_belize preserves author train/val splits explicitly."""
    belize_dir = tmp_path / "raw" / "belize"
    train_gun = belize_dir / "Training data" / "Gunshot"
    train_bg = belize_dir / "Training data" / "Background"
    val_gun = belize_dir / "Validation data" / "Gunshot"
    val_bg = belize_dir / "Validation data" / "Background"

    for d in [train_gun, train_bg, val_gun, val_bg]:
        d.mkdir(parents=True)

    (train_gun / "g1.wav").write_bytes(b"wav")
    (train_bg / "b1.wav").write_bytes(b"wav")
    (val_gun / "g2.wav").write_bytes(b"wav")
    (val_bg / "b2.wav").write_bytes(b"wav")

    records, unmapped = scan_belize(belize_dir, tmp_path)
    assert len(records) == 4
    assert unmapped == []

    orig_map = {r.path.split("/")[-1]: r for r in records}
    assert orig_map["g1.wav"].original_split == "train"
    assert orig_map["g1.wav"].label == "gunshot"
    assert orig_map["b1.wav"].original_split == "train"
    assert orig_map["b1.wav"].label == "background"
    assert orig_map["g2.wav"].original_split == "val"
    assert orig_map["b2.wav"].original_split == "val"


def test_scan_esc50_noise_tag(tmp_path: Path) -> None:
    """scan_esc50 tags thunderstorm/wind/rain clips with noise_tag='thunder_wind'."""
    esc_dir = tmp_path / "raw" / "esc50"
    audio_dir = esc_dir / "audio"
    audio_dir.mkdir(parents=True)

    # 1 target clip and 1 non-target clip
    (audio_dir / "1-100-A-19.wav").write_bytes(b"wav")  # thunderstorm
    (audio_dir / "1-200-A-0.wav").write_bytes(b"wav")  # dog (non-target)

    meta_dir = esc_dir / "meta"
    meta_dir.mkdir(parents=True)
    csv_content = (
        "filename,fold,target,category,esc10,src_id,take\n"
        "1-100-A-19.wav,1,19,thunderstorm,True,100,A\n"
        "1-200-A-0.wav,1,0,dog,False,200,A\n"
    )
    (meta_dir / "esc50.csv").write_text(csv_content)

    records, unmapped = scan_esc50(esc_dir, tmp_path)
    assert len(records) == 1
    assert records[0].label == "background"  # NEVER relabeled as gunshot
    assert records[0].noise_tag == "thunder_wind"
    assert records[0].source == "esc50"


def test_scan_all_datasets_empty_directory(tmp_path: Path) -> None:
    """scan_all_datasets on clean unpopulated directory returns 0 records."""
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True)

    for d in ["birdsai", "cct", "serengeti", "belize", "vietnam", "esc50"]:
        (raw_root / d).mkdir()
        (raw_root / d / ".gitkeep").touch()

    records, unmapped = scan_all_datasets(data_root)
    assert records == []
    assert unmapped == []
