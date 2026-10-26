"""Unit tests for src/splits.py.

Covers:
- Group leakage prevention: all samples with the same group_id stay in one split
- Mutual exclusivity across train, val, and test splits
- Determinism and reproducibility with fixed seeds (27, 2701, 26009)
- Preservation of original_split metadata
- Honest handling of sparse classes (fewer groups than splits triggers warnings)
"""

from __future__ import annotations

from src.manifest import ManifestRecord
from src.splits import assign_splits


def _make_dummy_records(
    n_groups: int = 10, items_per_group: int = 3, label: str = "animal"
) -> list[ManifestRecord]:
    """Helper to generate dummy records with explicit group IDs."""
    records: list[ManifestRecord] = []
    for g in range(n_groups):
        group_id = f"group_{label}_{g:03d}"
        for i in range(items_per_group):
            records.append(
                ManifestRecord(
                    path=f"raw/cct/{label}/g{g}_{i}.jpg",
                    modality="image",
                    source="cct",
                    label=label,
                    view="camera_trap",
                    site_or_device=f"loc_{g}",
                    noise_tag="not_thunder_wind",
                    original_split="unspecified",
                    group_id=group_id,
                )
            )
    return records


def test_no_group_leakage() -> None:
    """Records sharing the same group_id must strictly belong to the same split."""
    records = _make_dummy_records(n_groups=20, items_per_group=4)
    split_records, _, _ = assign_splits(records, seed=27)

    # Map group_id to splits containing it
    group_splits: dict[str, set[str]] = {}
    for r in split_records:
        group_splits.setdefault(r.group_id, set()).add(r.split)

    # Every single group must belong to exactly one split
    for gid, splits in group_splits.items():
        assert len(splits) == 1, f"Group {gid} leaked across multiple splits: {splits}"


def test_splits_are_mutually_exclusive() -> None:
    """Train, val, and test paths must have zero intersection."""
    records = _make_dummy_records(n_groups=30, items_per_group=2)
    split_records, _, _ = assign_splits(records, seed=27)

    train_paths = {r.path for r in split_records if r.split == "train"}
    val_paths = {r.path for r in split_records if r.split == "val"}
    test_paths = {r.path for r in split_records if r.split == "test"}

    assert len(train_paths & val_paths) == 0, "Train and val intersect!"
    assert len(train_paths & test_paths) == 0, "Train and test intersect!"
    assert len(val_paths & test_paths) == 0, "Val and test intersect!"
    assert len(train_paths | val_paths | test_paths) == len(records)


def test_splits_deterministic_with_seed() -> None:
    """Same seed must produce identical split assignments; different seeds must vary."""
    records = _make_dummy_records(n_groups=20, items_per_group=3)

    run1, _, _ = assign_splits(records, seed=2701)
    run2, _, _ = assign_splits(records, seed=2701)
    run_diff, _, _ = assign_splits(records, seed=26009)

    splits1 = [r.split for r in run1]
    splits2 = [r.split for r in run2]
    splits_diff = [r.split for r in run_diff]

    assert splits1 == splits2, "Identical seed produced different splits!"
    assert splits1 != splits_diff, "Different seeds produced identical splits on large set!"


def test_original_split_preserved() -> None:
    """original_split column must remain untouched when split is assigned."""
    records = [
        ManifestRecord(
            path=f"raw/belize/g{i}.wav",
            modality="audio",
            source="belize",
            label="gunshot",
            view="acoustic",
            site_or_device="belize",
            noise_tag="not_thunder_wind",
            original_split="train" if i % 2 == 0 else "val",
            group_id=f"bg_{i}",
        )
        for i in range(12)
    ]

    split_records, _, _ = assign_splits(records, seed=27)

    for orig, updated in zip(records, split_records, strict=True):
        assert updated.original_split == orig.original_split
        assert updated.split in {"train", "val", "test"}


def test_honest_sparse_class_handling() -> None:
    """Sparse classes with < 3 groups trigger warnings and don't leak or fabricate."""
    # Vehicle has only 2 groups
    sparse_vehicles = _make_dummy_records(n_groups=2, items_per_group=2, label="vehicle")
    # Animals has 15 groups
    dense_animals = _make_dummy_records(n_groups=15, items_per_group=2, label="animal")

    combined = sparse_vehicles + dense_animals
    split_records, sparse_dict, warnings = assign_splits(combined, seed=27)

    assert len(warnings) > 0
    assert any("vehicle" in w for w in warnings)
    assert any("cct:vehicle:not_thunder_wind" in k for k in sparse_dict)

    # Verify no vehicle group leaked
    veh_splits = {r.split for r in split_records if r.label == "vehicle"}
    # Since only 2 groups, vehicle is present in at most 2 splits
    assert len(veh_splits) <= 2
