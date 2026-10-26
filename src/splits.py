"""
GuardianEye group-aware stratified splitting module.

Partitions manifest records into train, val, and test splits (0.70 / 0.15 / 0.15)
while strictly preventing group leakage (video sequences, sensor deployments,
and recording sessions are never split across subsets).
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import Any

from src.manifest import ManifestRecord

logger = logging.getLogger(__name__)


def assign_splits(
    records: list[ManifestRecord],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[list[ManifestRecord], dict[str, Any], list[str]]:
    """
    Assign records to train/val/test splits without group leakage.

    Parameters
    ----------
    records:
        List of unassigned or previously assigned ManifestRecord objects.
    seed:
        Integer random seed for deterministic partitioning.
    train_frac:
        Target fraction for training split.
    val_frac:
        Target fraction for validation split.
    test_frac:
        Target fraction for test split.

    Returns
    -------
    tuple[list[ManifestRecord], dict[str, Any], list[str]]
        - Updated ManifestRecord list with split set to 'train', 'val', or 'test'.
        - Sparse class summary dict.
        - List of warnings (e.g. sparse classes with fewer groups than splits).

    """
    total_frac = round(train_frac + val_frac + test_frac, 6)
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_frac}")

    if not records:
        return [], {}, ["No records to split"]

    # 1. Group records by group_id
    groups: dict[str, list[ManifestRecord]] = {}
    for r in records:
        groups.setdefault(r.group_id, []).append(r)

    # 2. Determine strata key for each group based on majority label & source
    group_strata: dict[str, tuple[str, str, str, str]] = {}
    for gid, recs in groups.items():
        # Represent stratum by (modality, source, majority_label, noise_tag)
        labels = [r.label for r in recs]
        maj_label = max(set(labels), key=labels.count)
        first = recs[0]
        group_strata[gid] = (first.modality, first.source, maj_label, first.noise_tag)

    # 3. Aggregate groups by stratum
    strata_groups: dict[tuple[str, str, str, str], list[str]] = {}
    for gid, stratum in group_strata.items():
        strata_groups.setdefault(stratum, []).append(gid)

    # Deterministic sorting before shuffle
    rng = random.Random(seed)
    group_assignments: dict[str, str] = {}
    sparse_classes: dict[str, Any] = {}
    warnings: list[str] = []

    # Track class occurrences across assigned splits
    class_splits: dict[str, set[str]] = {}

    for stratum, gids in sorted(strata_groups.items(), key=lambda t: t[0]):
        modality, source, label, noise_tag = stratum
        sorted_gids = sorted(gids)
        rng.shuffle(sorted_gids)
        n_groups = len(sorted_gids)

        if n_groups < 3:
            # Sparse class warning — impossible to guarantee all 3 splits without group leakage
            warn_msg = (
                f"Sparse stratum {stratum}: only {n_groups} independent group(s). "
                f"Cannot represent all 3 splits without leakage."
            )
            warnings.append(warn_msg)
            logger.warning(warn_msg)

            # Assign available groups deterministically
            if n_groups == 1:
                assigned = [("train", sorted_gids[0])]
            elif n_groups == 2:
                assigned = [("train", sorted_gids[0]), ("val", sorted_gids[1])]
            else:
                assigned = []

            for split_name, gid in assigned:
                group_assignments[gid] = split_name
                class_splits.setdefault(label, set()).add(split_name)

            sparse_classes[f"{source}:{label}:{noise_tag}"] = {
                "num_groups": n_groups,
                "splits_present": sorted(class_splits.get(label, set())),
            }
            continue

        # Standard partitioning for strata with >= 3 groups
        n_val = max(1, round(n_groups * val_frac))
        n_test = max(1, round(n_groups * test_frac))
        # Ensure at least 1 group remains for train
        if n_val + n_test >= n_groups:
            n_val = 1
            n_test = 1
        n_train = n_groups - n_val - n_test

        train_gids = sorted_gids[:n_train]
        val_gids = sorted_gids[n_train : n_train + n_val]
        test_gids = sorted_gids[n_train + n_val :]

        for gid in train_gids:
            group_assignments[gid] = "train"
            class_splits.setdefault(label, set()).add("train")
        for gid in val_gids:
            group_assignments[gid] = "val"
            class_splits.setdefault(label, set()).add("val")
        for gid in test_gids:
            group_assignments[gid] = "test"
            class_splits.setdefault(label, set()).add("test")

    # 4. Update records with assigned splits while preserving original_split
    updated_records: list[ManifestRecord] = []
    for r in records:
        assigned_split = group_assignments.get(r.group_id, "train")
        updated_records.append(replace(r, split=assigned_split))

    # Deterministic sorting
    updated_records.sort(key=lambda r: (r.modality, r.source, r.path))
    return updated_records, sparse_classes, warnings
