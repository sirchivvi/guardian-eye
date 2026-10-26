"""
GuardianEye dataset scanner and manifest builder.

Scans raw dataset directories, validates schema and modality-specific labels,
normalizes paths to POSIX format, detects duplicates, preserves original splits,
and generates standardized manifest tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical schema constants
VALID_MODALITIES: frozenset[str] = frozenset({"image", "audio"})
VALID_IMAGE_LABELS: frozenset[str] = frozenset({"human", "vehicle", "animal", "empty"})
VALID_AUDIO_LABELS: frozenset[str] = frozenset({"gunshot", "background"})
VALID_VIEWS: frozenset[str] = frozenset({"aerial_thermal", "camera_trap", "acoustic"})
VALID_NOISE_TAGS: frozenset[str] = frozenset({"thunder_wind", "not_thunder_wind"})
VALID_ORIGINAL_SPLITS: frozenset[str] = frozenset({"train", "val", "test", "unspecified"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg", ".mp3"})
IGNORED_FILENAMES: frozenset[str] = frozenset({".gitkeep", ".DS_Store", "Thumbs.db"})

MANIFEST_COLUMNS: list[str] = [
    "path",
    "modality",
    "source",
    "label",
    "view",
    "site_or_device",
    "noise_tag",
    "original_split",
    "group_id",
    "split",
    "notes",
]


@dataclass(frozen=True)
class ManifestRecord:
    """A single normalized manifest entry representing an image or audio file."""

    path: str  # POSIX relative path from data_root
    modality: str  # "image" or "audio"
    source: str  # dataset slug: birdsai, cct, serengeti, belize, vietnam, esc50
    label: str  # canonical class label
    view: str  # "aerial_thermal", "camera_trap", or "acoustic"
    site_or_device: str  # site/device identifier or "unknown"
    noise_tag: str  # "thunder_wind" or "not_thunder_wind"
    original_split: str  # "train", "val", "test", or "unspecified"
    group_id: str  # group identifier for leakage-free splitting
    split: str = "unassigned"  # assigned experimental split
    notes: str = ""  # metadata or warning notes

    def validate(self) -> None:
        """Validate record against canonical taxonomy."""
        if self.modality not in VALID_MODALITIES:
            raise ValueError(
                f"Invalid modality {self.modality!r}; must be one of {VALID_MODALITIES}"
            )

        if self.modality == "image" and self.label not in VALID_IMAGE_LABELS:
            raise ValueError(
                f"Invalid image label {self.label!r}; must be one of {VALID_IMAGE_LABELS}"
            )

        if self.modality == "audio" and self.label not in VALID_AUDIO_LABELS:
            raise ValueError(
                f"Invalid audio label {self.label!r}; must be one of {VALID_AUDIO_LABELS}"
            )

        if self.view not in VALID_VIEWS:
            raise ValueError(f"Invalid view {self.view!r}; must be one of {VALID_VIEWS}")

        if self.noise_tag not in VALID_NOISE_TAGS:
            raise ValueError(
                f"Invalid noise_tag {self.noise_tag!r}; must be one of {VALID_NOISE_TAGS}"
            )

        if self.original_split not in VALID_ORIGINAL_SPLITS:
            raise ValueError(
                f"Invalid original_split {self.original_split!r}; "
                f"must be one of {VALID_ORIGINAL_SPLITS}"
            )

        if "\\" in self.path:
            raise ValueError(f"Path must be POSIX normalized (use '/'), got {self.path!r}")


def to_posix_relpath(path: Path, base_dir: Path) -> str:
    """Convert path to a POSIX-compliant relative string from base_dir."""
    return str(path.resolve().relative_to(base_dir.resolve())).replace("\\", "/")


def _file_hash_prefix(path: Path) -> str:
    """Compute short MD5 prefix of file header for duplicate detection."""
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
            return hashlib.md5(chunk).hexdigest()[:12]
    except OSError:
        return "unreadable"


# ---------------------------------------------------------------------------
# Dataset Scanners
# ---------------------------------------------------------------------------


def scan_birdsai(dataset_dir: Path, data_root: Path) -> tuple[list[ManifestRecord], list[str]]:
    """
    Scan BIRDSAI thermal aerial frames and annotations.

    Supports:
    1. MOT-style annotation files (gt.txt, annotations.txt) with format:
       <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class_id>,...
       where class_id 1 = human, 0 = animal. Frames with no box = empty.
       Frames with both = human with note 'human_and_animal_boxes'.
    2. Class folders (human, vehicle, animal, empty) as fallback.
    """
    records: list[ManifestRecord] = []
    unmapped: list[str] = []

    if not dataset_dir.is_dir():
        return records, unmapped

    # Check for MOT annotation files
    annot_files = list(dataset_dir.rglob("gt.txt")) + list(dataset_dir.rglob("*annotation*.txt"))
    if annot_files:
        for annot_file in annot_files:
            seq_dir = annot_file.parent
            frames_dir = seq_dir / "img1" if (seq_dir / "img1").is_dir() else seq_dir / "frames"
            if not frames_dir.is_dir():
                frames_dir = seq_dir

            # Parse MOT annotations per frame
            frame_classes: dict[int, set[int]] = {}
            try:
                with annot_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 8:
                            try:
                                frame_idx = int(parts[0])
                                class_id = int(parts[7])
                                frame_classes.setdefault(frame_idx, set()).add(class_id)
                            except ValueError:
                                continue
            except OSError as err:
                logger.warning("Error reading BIRDSAI annotation %s: %s", annot_file, err)
                continue

            # Ingest all image frames
            for img_path in sorted(frames_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Deduce frame number from filename stem (e.g. 000001.jpg -> 1)
                try:
                    num_str = "".join(c for c in img_path.stem if c.isdigit())
                    f_idx = int(num_str) if num_str else -1
                except ValueError:
                    f_idx = -1

                classes = frame_classes.get(f_idx, set())
                notes = ""
                if 1 in classes and 0 in classes:
                    label = "human"
                    notes = "human_and_animal_boxes"
                elif 1 in classes:
                    label = "human"
                elif 0 in classes:
                    label = "animal"
                else:
                    label = "empty"

                group_id = f"birdsai_{seq_dir.name}"
                records.append(
                    ManifestRecord(
                        path=to_posix_relpath(img_path, data_root),
                        modality="image",
                        source="birdsai",
                        label=label,
                        view="aerial_thermal",
                        site_or_device=seq_dir.name,
                        noise_tag="not_thunder_wind",
                        original_split="unspecified",
                        group_id=group_id,
                        notes=notes,
                    )
                )
        return records, unmapped

    # Class-folder fallback
    return scan_class_folders(
        dataset_dir=dataset_dir,
        data_root=data_root,
        source="birdsai",
        view="aerial_thermal",
        modality="image",
    )


def scan_class_folders(
    dataset_dir: Path,
    data_root: Path,
    source: str,
    view: str,
    modality: str,
) -> tuple[list[ManifestRecord], list[str]]:
    """Scan dataset structured in class sub-folders (human, vehicle, animal, empty)."""
    records: list[ManifestRecord] = []
    unmapped: list[str] = []

    if not dataset_dir.is_dir():
        return records, unmapped

    # Check for COCO camera traps json annotation file
    coco_jsons = [
        p for p in dataset_dir.glob("*.json")
        if "cct" in p.name.lower() or "coco" in p.name.lower()
    ]
    if coco_jsons:
        return scan_coco_json(dataset_dir, coco_jsons[0], data_root, source, view)

    valid_labels = VALID_IMAGE_LABELS if modality == "image" else VALID_AUDIO_LABELS
    exts = IMAGE_EXTENSIONS if modality == "image" else AUDIO_EXTENSIONS

    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and child.name.lower() in valid_labels:
            label = child.name.lower()
            for file_path in sorted(child.rglob("*")):
                if file_path.is_file():
                    if file_path.name in IGNORED_FILENAMES:
                        continue
                    if file_path.suffix.lower() in exts:
                        # Extract deployment/camera group identifier if directory nested
                        group_id = f"{source}_{file_path.parent.name}_{file_path.stem}"
                        records.append(
                            ManifestRecord(
                                path=to_posix_relpath(file_path, data_root),
                                modality=modality,
                                source=source,
                                label=label,
                                view=view,
                                site_or_device=source,
                                noise_tag="not_thunder_wind",
                                original_split="unspecified",
                                group_id=group_id,
                            )
                        )
                    else:
                        unmapped.append(to_posix_relpath(file_path, data_root))

    return records, unmapped


def scan_coco_json(
    dataset_dir: Path,
    json_path: Path,
    data_root: Path,
    source: str,
    view: str,
) -> tuple[list[ManifestRecord], list[str]]:
    """Scan COCO Camera Traps format JSON metadata."""
    records: list[ManifestRecord] = []
    unmapped: list[str] = []

    try:
        with json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse COCO JSON %s: %s", json_path, e)
        return records, unmapped

    # Map categories to canonical labels: human, vehicle, animal, empty
    cat_map: dict[int, str] = {}
    for cat in coco.get("categories", []):
        cname = cat.get("name", "").lower()
        cid = cat.get("id")
        if "person" in cname or "human" in cname:
            cat_map[cid] = "human"
        elif "vehicle" in cname or "car" in cname or "truck" in cname:
            cat_map[cid] = "vehicle"
        elif "empty" in cname or "blank" in cname:
            cat_map[cid] = "empty"
        else:
            cat_map[cid] = "animal"

    # Annotations by image_id
    img_annots: dict[Any, list[int]] = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if img_id is not None and cat_id is not None:
            img_annots.setdefault(img_id, []).append(cat_id)

    images_dir = dataset_dir / "images" if (dataset_dir / "images").is_dir() else dataset_dir

    for img in coco.get("images", []):
        img_id = img.get("id")
        file_name = img.get("file_name", "")
        img_file = images_dir / file_name
        if not img_file.is_file():
            img_file = dataset_dir / file_name

        if not img_file.is_file():
            continue

        cat_ids = img_annots.get(img_id, [])
        assigned_labels = [cat_map[cid] for cid in cat_ids if cid in cat_map]
        notes = ""
        if "human" in assigned_labels and "animal" in assigned_labels:
            label = "human"
            notes = "human_and_animal_boxes"
        elif "human" in assigned_labels:
            label = "human"
        elif "vehicle" in assigned_labels:
            label = "vehicle"
        elif "animal" in assigned_labels:
            label = "animal"
        else:
            label = "empty"

        location = str(img.get("location", img.get("site", source)))
        group_id = f"{source}_{location}"
        records.append(
            ManifestRecord(
                path=to_posix_relpath(img_file, data_root),
                modality="image",
                source=source,
                label=label,
                view=view,
                site_or_device=location,
                noise_tag="not_thunder_wind",
                original_split="unspecified",
                group_id=group_id,
                notes=notes,
            )
        )

    return records, unmapped


def scan_belize(dataset_dir: Path, data_root: Path) -> tuple[list[ManifestRecord], list[str]]:
    """
    Scan Belize gunshot dataset, strictly preserving author train/validation split.

    Structure expected:
    - Training data/
        - Gunshot/
        - Background/
    - Validation data/
        - Gunshot/
        - Background/
    """
    records: list[ManifestRecord] = []
    unmapped: list[str] = []

    if not dataset_dir.is_dir():
        return records, unmapped

    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue

        orig_split = "unspecified"
        lower_name = child.name.lower()
        if "train" in lower_name:
            orig_split = "train"
        elif "val" in lower_name:
            orig_split = "val"

        for label_dir in sorted(child.iterdir()):
            if not label_dir.is_dir():
                continue

            l_name = label_dir.name.lower()
            if "gunshot" in l_name:
                label = "gunshot"
            elif "background" in l_name:
                label = "background"
            else:
                continue

            for file_path in sorted(label_dir.rglob("*")):
                if file_path.is_file():
                    if file_path.name in IGNORED_FILENAMES:
                        continue
                    if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                        # Extract session/deployment group ID from filename prefix if present
                        group_id = f"belize_{file_path.stem.split('_')[0]}"
                        records.append(
                            ManifestRecord(
                                path=to_posix_relpath(file_path, data_root),
                                modality="audio",
                                source="belize",
                                label=label,
                                view="acoustic",
                                site_or_device="belize_tropical_forest",
                                noise_tag="not_thunder_wind",
                                original_split=orig_split,
                                group_id=group_id,
                            )
                        )
                    else:
                        unmapped.append(to_posix_relpath(file_path, data_root))

    return records, unmapped


def scan_vietnam(dataset_dir: Path, data_root: Path) -> tuple[list[ManifestRecord], list[str]]:
    """Scan Vietnam Chu Mom Ray audio recordings."""
    records: list[ManifestRecord] = []
    unmapped: list[str] = []

    if not dataset_dir.is_dir():
        return records, unmapped

    for file_path in sorted(dataset_dir.rglob("*")):
        if file_path.is_file():
            if file_path.name in IGNORED_FILENAMES:
                continue
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                parent_name = file_path.parent.name.lower()
                stem_name = file_path.stem.lower()

                if "gunshot" in parent_name or "gunshot" in stem_name:
                    label = "gunshot"
                else:
                    label = "background"

                group_id = f"vietnam_{file_path.stem.split('_')[0]}"
                records.append(
                    ManifestRecord(
                        path=to_posix_relpath(file_path, data_root),
                        modality="audio",
                        source="vietnam",
                        label=label,
                        view="acoustic",
                        site_or_device="vietnam_chu_mom_ray",
                        noise_tag="not_thunder_wind",
                        original_split="unspecified",
                        group_id=group_id,
                    )
                )
            elif (
                file_path.suffix.lower() in IMAGE_EXTENSIONS
                and "spectrogram" in str(file_path).lower()
            ):
                # Ignore standalone spectrogram images per Build Guide v2
                pass
            else:
                unmapped.append(to_posix_relpath(file_path, data_root))

    return records, unmapped


def scan_esc50(dataset_dir: Path, data_root: Path) -> tuple[list[ManifestRecord], list[str]]:
    """Scan ESC-50 dataset for thunder, wind, and rain clips (noise_tag=thunder_wind)."""
    records: list[ManifestRecord] = []
    unmapped: list[str] = []

    if not dataset_dir.is_dir():
        return records, unmapped

    meta_file = dataset_dir / "meta" / "esc50.csv"
    if not meta_file.is_file():
        # Fallback to search any esc50.csv
        metas = list(dataset_dir.rglob("esc50.csv"))
        if metas:
            meta_file = metas[0]

    target_categories = {"thunderstorm", "wind", "rain"}
    target_filenames: dict[str, str] = {}

    if meta_file.is_file():
        try:
            with meta_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cat = row.get("category", "").lower()
                    fname = row.get("filename", "")
                    if cat in target_categories:
                        target_filenames[fname] = cat
        except OSError as e:
            logger.warning("Error reading ESC-50 meta.csv %s: %s", meta_file, e)

    audio_dir = dataset_dir / "audio" if (dataset_dir / "audio").is_dir() else dataset_dir

    for file_path in sorted(audio_dir.rglob("*")):
        if file_path.is_file():
            if file_path.name in IGNORED_FILENAMES:
                continue
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                fname = file_path.name
                parent_name = file_path.parent.name.lower()
                is_target = False
                matched_cat = "thunder_wind"

                if target_filenames:
                    if fname in target_filenames:
                        is_target = True
                        matched_cat = target_filenames[fname]
                else:
                    if any(t in fname.lower() or t in parent_name for t in target_categories):
                        is_target = True

                if is_target:
                    # Parse fold from filename (e.g. 1-100032-A-0.wav -> fold 1)
                    fold = fname.split("-")[0] if "-" in fname else "1"
                    group_id = f"esc50_fold_{fold}"
                    records.append(
                        ManifestRecord(
                            path=to_posix_relpath(file_path, data_root),
                            modality="audio",
                            source="esc50",
                            label="background",  # NEVER relabeled as gunshot
                            view="acoustic",
                            site_or_device="esc50_environmental",
                            noise_tag="thunder_wind",  # Strictly tagged
                            original_split="unspecified",
                            group_id=group_id,
                            notes=f"esc50_category_{matched_cat}",
                        )
                    )
            else:
                unmapped.append(to_posix_relpath(file_path, data_root))

    return records, unmapped


# ---------------------------------------------------------------------------
# Master Scanner
# ---------------------------------------------------------------------------


def scan_all_datasets(data_root: Path) -> tuple[list[ManifestRecord], list[str]]:
    """
    Scan all raw datasets under data_root/raw and return records and unmapped files.

    Detects and eliminates duplicate paths, verifies taxonomy schema, and sorts deterministically.
    """
    raw_root = data_root / "raw" if (data_root / "raw").is_dir() else data_root
    records: list[ManifestRecord] = []
    unmapped: list[str] = []

    # 1. BIRDSAI
    r, u = scan_birdsai(raw_root / "birdsai", data_root)
    records.extend(r)
    unmapped.extend(u)

    # 2. CCT
    r, u = scan_class_folders(raw_root / "cct", data_root, "cct", "camera_trap", "image")
    records.extend(r)
    unmapped.extend(u)

    # 3. Serengeti
    r, u = scan_class_folders(
        raw_root / "serengeti", data_root, "serengeti", "camera_trap", "image"
    )
    records.extend(r)
    unmapped.extend(u)

    # 4. Belize
    r, u = scan_belize(raw_root / "belize", data_root)
    records.extend(r)
    unmapped.extend(u)

    # 5. Vietnam
    r, u = scan_vietnam(raw_root / "vietnam", data_root)
    records.extend(r)
    unmapped.extend(u)

    # 6. ESC-50
    r, u = scan_esc50(raw_root / "esc50", data_root)
    records.extend(r)
    unmapped.extend(u)

    # Validate, deduplicate, and sort deterministically
    seen_paths: set[str] = set()
    deduped_records: list[ManifestRecord] = []
    for rec in records:
        rec.validate()
        if rec.path in seen_paths:
            logger.warning("Duplicate path detected and skipped: %s", rec.path)
            continue
        seen_paths.add(rec.path)
        deduped_records.append(rec)

    # Deterministic sorting: modality, source, path
    deduped_records.sort(key=lambda r: (r.modality, r.source, r.path))
    return deduped_records, unmapped


def save_manifest_csv(records: list[ManifestRecord], output_path: Path) -> None:
    """Save manifest records to CSV with standard schema columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    logger.info("Saved manifest (%d rows) -> %s", len(records), output_path)
