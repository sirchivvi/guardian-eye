"""
GuardianEye synthetic pairing module.

Generates reproducible, leakage-free multimodal (RGB image + audio) pairings
strictly partitioned within dataset splits (train, validation, test).

Key guarantees:
1. Split Isolation: Pairs are generated strictly within a single split (train × train,
   val × val, test × test). Cross-split pairings are strictly forbidden.
2. Deduplication: The exact (image_path, audio_path, scenario) triple is never repeated.
3. Class Honesty: If candidate pools for a class are empty (such as vehicle images),
   zero pairs are built. No synthetic vehicle images or false audio labels are ever invented.
4. Determinism: Cross-platform deterministic pairing seeded via sha256 integer hashes.
5. Scenario Semantics: Maps pairs to canonical scenarios and evaluates heuristic alert
   (suspicious) and risk tier (HIGH, MEDIUM, LOW) via src.decisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.decisions import risk, suspicious
from src.manifest import ManifestRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models and Specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """Specification of a multimodal synthetic scenario."""

    name: str
    image_label: str
    audio_label: str
    audio_noise_tag: str | None = None  # None matches any noise_tag


CANONICAL_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("human_gunshot", "human", "gunshot", "not_thunder_wind"),
    ScenarioSpec("vehicle_gunshot", "vehicle", "gunshot", "not_thunder_wind"),
    ScenarioSpec("animal_gunshot", "animal", "gunshot", "not_thunder_wind"),
    ScenarioSpec("human_background", "human", "background", "not_thunder_wind"),
    ScenarioSpec("vehicle_background", "vehicle", "background", "not_thunder_wind"),
    ScenarioSpec("animal_background", "animal", "background", "not_thunder_wind"),
    ScenarioSpec("empty_background", "empty", "background", "not_thunder_wind"),
    ScenarioSpec("empty_thunder", "empty", "background", "thunder_wind"),
)

PAIRS_CSV_COLUMNS: list[str] = [
    "pair_id",
    "split",
    "scenario",
    "image_path",
    "audio_path",
    "image_source",
    "audio_source",
    "image_label",
    "audio_label",
    "audio_noise_tag",
    "image_view",
    "image_group_id",
    "audio_group_id",
    "suspicious",
    "risk",
]

SCENARIOS_CSV_COLUMNS: list[str] = [
    "split",
    "image_label",
    "audio_label",
    "scenario",
    "suspicious",
    "risk",
]


@dataclass(frozen=True)
class SyntheticPair:
    """A synthetic multimodal pair with provenance, labels, and heuristic risk."""

    pair_id: str
    split: str
    scenario: str
    image_path: str
    audio_path: str
    image_source: str
    audio_source: str
    image_label: str
    audio_label: str
    audio_noise_tag: str
    image_view: str
    image_group_id: str
    audio_group_id: str
    suspicious: int
    risk: str


# ---------------------------------------------------------------------------
# Filtering and Pairing Logic
# ---------------------------------------------------------------------------


def filter_candidates(
    records: list[ManifestRecord],
    split: str,
    modality: str,
    label: str,
    noise_tag: str | None = None,
) -> list[ManifestRecord]:
    """
    Filter records by split, modality, label, and optional noise tag.

    Parameters
    ----------
    records:
        List of ManifestRecord objects.
    split:
        Target split ("train", "val", "test").
    modality:
        "image" or "audio".
    label:
        Target class label.
    noise_tag:
        Optional noise tag to filter by ("thunder_wind", "not_thunder_wind").

    Returns
    -------
    list[ManifestRecord]
        Filtered candidate records, sorted deterministically by (source, path).

    """
    candidates: list[ManifestRecord] = []
    for r in records:
        if r.split != split:
            continue
        if r.modality != modality:
            continue
        if r.label != label:
            continue
        if noise_tag is not None and r.noise_tag != noise_tag:
            continue
        candidates.append(r)

    # Sort deterministically to ensure reproducibility regardless of input order
    candidates.sort(key=lambda r: (r.source, r.path))
    return candidates


def _derive_scenario_seed(master_seed: int, split: str, scenario_name: str) -> int:
    """
    Derive a deterministic integer seed for a scenario within a split.

    Uses sha256 to remain independent of process-specific hash randomization.
    """
    token = f"{master_seed}_{split}_{scenario_name}".encode()
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def pair_scenario(
    image_records: list[ManifestRecord],
    audio_records: list[ManifestRecord],
    scenario: ScenarioSpec,
    split: str,
    seed: int,
    max_pairs: int,
) -> list[SyntheticPair]:
    """
    Generate synthetic pairs for a single scenario within one split.

    Guarantees:
    - Never fabricates pairs if image_records or audio_records is empty.
    - No duplicate (image_path, audio_path, scenario) triples.
    - Fair and balanced reuse across candidates.
    - Seed-deterministic ordering.

    Parameters
    ----------
    image_records:
        Image candidate records for this scenario.
    audio_records:
        Audio candidate records for this scenario.
    scenario:
        Scenario specification.
    split:
        Split name ("train", "val", "test").
    seed:
        Master integer seed.
    max_pairs:
        Maximum number of pairs to produce for this scenario.

    Returns
    -------
    list[SyntheticPair]
        Generated synthetic pairs.

    """
    if not image_records or not audio_records or max_pairs <= 0:
        return []

    # Ensure deterministic base order
    images = sorted(image_records, key=lambda r: (r.source, r.path))
    audios = sorted(audio_records, key=lambda r: (r.source, r.path))

    n_images = len(images)
    n_audio = len(audios)
    max_possible = n_images * n_audio
    target_count = min(max_pairs, max_possible)

    # Shuffle both pools with the scenario-derived seed
    scenario_seed = _derive_scenario_seed(seed, split, scenario.name)
    import random

    rng = random.Random(scenario_seed)
    shuffled_images = list(images)
    shuffled_audios = list(audios)
    rng.shuffle(shuffled_images)
    rng.shuffle(shuffled_audios)

    seen_pairs: set[tuple[str, str]] = set()
    pairs: list[SyntheticPair] = []
    pair_counter = 0

    # Systematically generate unique pairs with balanced candidate reuse
    for offset in range(n_audio):
        for i in range(n_images):
            img = shuffled_images[i]
            aud = shuffled_audios[(i + offset) % n_audio]
            pair_key = (img.path, aud.path)

            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            susp = suspicious(img.label, aud.label)
            r_level = risk(img.label, aud.label)

            p_id = f"pair_{split}_{scenario.name}_{pair_counter:06d}"
            pairs.append(
                SyntheticPair(
                    pair_id=p_id,
                    split=split,
                    scenario=scenario.name,
                    image_path=img.path,
                    audio_path=aud.path,
                    image_source=img.source,
                    audio_source=aud.source,
                    image_label=img.label,
                    audio_label=aud.label,
                    audio_noise_tag=aud.noise_tag,
                    image_view=img.view,
                    image_group_id=img.group_id,
                    audio_group_id=aud.group_id,
                    suspicious=susp,
                    risk=r_level,
                )
            )
            pair_counter += 1
            if len(pairs) >= target_count:
                break
        if len(pairs) >= target_count:
            break

    return pairs


def generate_split_pairs(
    records: list[ManifestRecord],
    split: str,
    seed: int,
    max_pairs_per_scenario: int = 1500,
    max_pairs_per_split: int = 10000,
    min_images_per_class: int = 0,
    scenarios: Sequence[ScenarioSpec] = CANONICAL_SCENARIOS,
) -> tuple[list[SyntheticPair], dict[str, int]]:
    """
    Generate synthetic pairings strictly within a single split.

    Parameters
    ----------
    records:
        Manifest records containing samples for this and potentially other splits.
    split:
        Target split ("train", "val", "test").
    seed:
        Master integer seed.
    max_pairs_per_scenario:
        Maximum pairs to build for each scenario.
    max_pairs_per_split:
        Hard ceiling on total pairs across all scenarios in this split.
    min_images_per_class:
        Minimum candidate images required before a class is included.
    scenarios:
        List of scenario specifications to build.

    Returns
    -------
    tuple[list[SyntheticPair], dict[str, int]]
        List of created SyntheticPair objects and per-scenario count dict.

    """
    # Strict split isolation: drop anything not belonging to this split
    split_records = [r for r in records if r.split == split]
    if not split_records:
        logger.info("Split %s has no records.", split)
        return [], {s.name: 0 for s in scenarios}

    # Count image candidates per class to enforce min_images_per_class threshold
    image_class_counts: dict[str, int] = {}
    for r in split_records:
        if r.modality == "image":
            image_class_counts[r.label] = image_class_counts.get(r.label, 0) + 1

    all_pairs: list[SyntheticPair] = []
    scenario_counts: dict[str, int] = {}

    for sc in scenarios:
        remaining_budget = max_pairs_per_split - len(all_pairs)
        if remaining_budget <= 0:
            logger.info("Split %s reached max_pairs_per_split (%d).", split, max_pairs_per_split)
            scenario_counts[sc.name] = 0
            continue

        # Check min_images_per_class threshold
        n_class_images = image_class_counts.get(sc.image_label, 0)
        if min_images_per_class > 0 and n_class_images < min_images_per_class:
            logger.warning(
                "Split %s, scenario %s: class %r has only %d images (< min %d). "
                "Writing 0 pairs honestly.",
                split,
                sc.name,
                sc.image_label,
                n_class_images,
                min_images_per_class,
            )
            scenario_counts[sc.name] = 0
            continue

        img_candidates = filter_candidates(
            split_records, split=split, modality="image", label=sc.image_label
        )
        aud_candidates = filter_candidates(
            split_records,
            split=split,
            modality="audio",
            label=sc.audio_label,
            noise_tag=sc.audio_noise_tag,
        )

        if not img_candidates:
            logger.info("Split %s: pool empty for image class %r.", split, sc.image_label)
        if not aud_candidates:
            logger.info(
                "Split %s: pool empty for audio label %r (noise_tag=%r).",
                split,
                sc.audio_label,
                sc.audio_noise_tag,
            )

        cap = min(max_pairs_per_scenario, remaining_budget)
        sc_pairs = pair_scenario(
            image_records=img_candidates,
            audio_records=aud_candidates,
            scenario=sc,
            split=split,
            seed=seed,
            max_pairs=cap,
        )

        scenario_counts[sc.name] = len(sc_pairs)
        all_pairs.extend(sc_pairs)

    return all_pairs, scenario_counts


def generate_all_pairs(
    records: list[ManifestRecord],
    seed: int,
    splits: Sequence[str] = ("train", "val", "test"),
    max_pairs_per_scenario: int = 1500,
    max_pairs_per_split: int = 10000,
    min_images_per_class: int = 0,
    scenarios: Sequence[ScenarioSpec] = CANONICAL_SCENARIOS,
) -> tuple[list[SyntheticPair], dict[str, Any]]:
    """
    Generate synthetic pairings across all splits with full statistics.

    Parameters
    ----------
    records:
        Manifest records.
    seed:
        Random seed.
    splits:
        Splits to process.
    max_pairs_per_scenario:
        Maximum pairs per scenario per split.
    max_pairs_per_split:
        Maximum total pairs per split.
    min_images_per_class:
        Minimum images per class threshold.
    scenarios:
        Sequence of scenarios to build.

    Returns
    -------
    tuple[list[SyntheticPair], dict[str, Any]]
        List of all pairs and a comprehensive statistics dictionary.

    """
    all_pairs: list[SyntheticPair] = []
    per_split_stats: dict[str, Any] = {}
    scenario_totals: dict[str, int] = {s.name: 0 for s in scenarios}

    for sp in splits:
        sp_pairs, sp_counts = generate_split_pairs(
            records=records,
            split=sp,
            seed=seed,
            max_pairs_per_scenario=max_pairs_per_scenario,
            max_pairs_per_split=max_pairs_per_split,
            min_images_per_class=min_images_per_class,
            scenarios=scenarios,
        )
        all_pairs.extend(sp_pairs)
        per_split_stats[sp] = {
            "total_pairs": len(sp_pairs),
            "scenario_counts": sp_counts,
            "empty_scenarios": [s for s, count in sp_counts.items() if count == 0],
        }
        for s_name, count in sp_counts.items():
            scenario_totals[s_name] += count

    stats: dict[str, Any] = {
        "seed": seed,
        "total_pairs": len(all_pairs),
        "per_split": per_split_stats,
        "scenario_totals": scenario_totals,
        "empty_scenarios_overall": [s for s, count in scenario_totals.items() if count == 0],
    }

    return all_pairs, stats


# ---------------------------------------------------------------------------
# CSV Persistence Functions
# ---------------------------------------------------------------------------


def save_pairs_csv(pairs: list[SyntheticPair], output_path: Path | str) -> Path:
    """
    Save detailed synthetic pairs with full provenance to CSV.

    Parameters
    ----------
    pairs:
        List of SyntheticPair instances.
    output_path:
        Target file path.

    Returns
    -------
    Path
        Resolved target file path.

    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIRS_CSV_COLUMNS)
        writer.writeheader()
        for p in pairs:
            writer.writerow(asdict(p))

    logger.info("Saved %d pairs to %s", len(pairs), out)
    return out


def save_scenarios_csv(pairs: list[SyntheticPair], output_path: Path | str) -> Path:
    """
    Save pairs to the BDA-compatible scenarios CSV schema.

    Schema: split,image_label,audio_label,scenario,suspicious,risk

    Parameters
    ----------
    pairs:
        List of SyntheticPair instances.
    output_path:
        Target file path (results/runs/seed_<n>/pairing/scenarios.csv).

    Returns
    -------
    Path
        Resolved target file path.

    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCENARIOS_CSV_COLUMNS)
        writer.writeheader()
        for p in pairs:
            writer.writerow(
                {
                    "split": p.split,
                    "image_label": p.image_label,
                    "audio_label": p.audio_label,
                    "scenario": p.scenario,
                    "suspicious": p.suspicious,
                    "risk": p.risk,
                }
            )

    logger.info("Saved %d scenario rows to %s", len(pairs), out)
    return out


def records_from_dataframe(df: pd.DataFrame) -> list[ManifestRecord]:
    """
    Convert a manifest DataFrame into a list of validated ManifestRecord objects.

    Parameters
    ----------
    df:
        DataFrame containing manifest columns.

    Returns
    -------
    list[ManifestRecord]
        List of ManifestRecord objects.

    """
    records: list[ManifestRecord] = []
    for _, row in df.iterrows():
        rec = ManifestRecord(
            path=str(row["path"]),
            modality=str(row["modality"]),
            source=str(row["source"]),
            label=str(row["label"]),
            view=str(row["view"]),
            site_or_device=str(row.get("site_or_device", "unknown")),
            noise_tag=str(row.get("noise_tag", "not_thunder_wind")),
            original_split=str(row.get("original_split", "unspecified")),
            group_id=str(row.get("group_id", "unknown")),
            split=str(row.get("split", "unassigned")),
            notes=str(row.get("notes", "")) if pd.notna(row.get("notes")) else "",
        )
        rec.validate()
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Verification and Self-Check
# ---------------------------------------------------------------------------


def verify_pairs(
    pairs: list[SyntheticPair],
    manifest_records: list[ManifestRecord] | None = None,
) -> dict[str, Any]:
    """
    Audit pairs for split leakage, duplicate triples, and decision rule violations.

    Parameters
    ----------
    pairs:
        List of SyntheticPair instances to audit.
    manifest_records:
        Optional original manifest records to strictly verify split provenance.

    Returns
    -------
    dict[str, Any]
        Audit report containing status, violation counts, and details.

    """
    manifest_map: dict[str, ManifestRecord] = {}
    if manifest_records:
        for r in manifest_records:
            manifest_map[r.path] = r

    scenario_map: dict[str, ScenarioSpec] = {s.name: s for s in CANONICAL_SCENARIOS}

    seen_triples: set[tuple[str, str, str]] = set()
    cross_split_violations = 0
    duplicate_triple_violations = 0
    decision_violations = 0
    scenario_label_violations = 0
    manifest_membership_violations = 0
    errors: list[str] = []

    for p in pairs:
        # 1. Duplicate triple check: (image_path, audio_path, scenario)
        triple = (p.image_path, p.audio_path, p.scenario)
        if triple in seen_triples:
            duplicate_triple_violations += 1
            errors.append(f"Duplicate triple: {triple}")
        seen_triples.add(triple)

        # 2. Scenario label semantics check
        if p.scenario not in scenario_map:
            scenario_label_violations += 1
            errors.append(f"Unknown scenario {p.scenario!r} in pair {p.pair_id}")
        else:
            spec = scenario_map[p.scenario]
            if p.image_label != spec.image_label:
                scenario_label_violations += 1
                errors.append(
                    f"Scenario {p.scenario} expects image_label {spec.image_label!r}, "
                    f"got {p.image_label!r} in pair {p.pair_id}"
                )
            if p.audio_label != spec.audio_label:
                scenario_label_violations += 1
                errors.append(
                    f"Scenario {p.scenario} expects audio_label {spec.audio_label!r}, "
                    f"got {p.audio_label!r} in pair {p.pair_id}"
                )
            if spec.audio_noise_tag is not None and p.audio_noise_tag != spec.audio_noise_tag:
                scenario_label_violations += 1
                errors.append(
                    f"Scenario {p.scenario} expects audio_noise_tag {spec.audio_noise_tag!r}, "
                    f"got {p.audio_noise_tag!r} in pair {p.pair_id}"
                )

        # 3. Decision rule check
        expected_susp = suspicious(p.image_label, p.audio_label)
        expected_risk = risk(p.image_label, p.audio_label)
        if p.suspicious != expected_susp or p.risk != expected_risk:
            decision_violations += 1
            errors.append(
                f"Decision mismatch in pair {p.pair_id}: "
                f"got suspicious={p.suspicious}, risk={p.risk}; "
                f"expected suspicious={expected_susp}, risk={expected_risk}"
            )

        # 4. Manifest membership and split provenance
        if manifest_map:
            img_rec = manifest_map.get(p.image_path)
            aud_rec = manifest_map.get(p.audio_path)

            if img_rec is None:
                manifest_membership_violations += 1
                errors.append(
                    f"Pair {p.pair_id}: image_path {p.image_path!r} not found in manifest"
                )
            else:
                if img_rec.modality != "image":
                    manifest_membership_violations += 1
                    errors.append(
                        f"Pair {p.pair_id}: image_path {p.image_path!r} is not modality 'image'"
                    )
                if img_rec.label != p.image_label:
                    manifest_membership_violations += 1
                    errors.append(
                        f"Pair {p.pair_id}: image_path label {img_rec.label!r} != "
                        f"pair image_label {p.image_label!r}"
                    )
                if img_rec.split != p.split:
                    cross_split_violations += 1
                    errors.append(
                        f"Pair {p.pair_id}: image split {img_rec.split!r} != pair.split {p.split!r}"
                    )

            if aud_rec is None:
                manifest_membership_violations += 1
                errors.append(
                    f"Pair {p.pair_id}: audio_path {p.audio_path!r} not found in manifest"
                )
            else:
                if aud_rec.modality != "audio":
                    manifest_membership_violations += 1
                    errors.append(
                        f"Pair {p.pair_id}: audio_path {p.audio_path!r} is not modality 'audio'"
                    )
                if aud_rec.label != p.audio_label:
                    manifest_membership_violations += 1
                    errors.append(
                        f"Pair {p.pair_id}: audio_path label {aud_rec.label!r} != "
                        f"pair audio_label {p.audio_label!r}"
                    )
                if aud_rec.noise_tag != p.audio_noise_tag:
                    manifest_membership_violations += 1
                    errors.append(
                        f"Pair {p.pair_id}: audio noise_tag {aud_rec.noise_tag!r} != "
                        f"pair audio_noise_tag {p.audio_noise_tag!r}"
                    )
                if aud_rec.split != p.split:
                    cross_split_violations += 1
                    errors.append(
                        f"Pair {p.pair_id}: audio split {aud_rec.split!r} != pair.split {p.split!r}"
                    )

    passed = (
        cross_split_violations == 0
        and duplicate_triple_violations == 0
        and decision_violations == 0
        and scenario_label_violations == 0
        and manifest_membership_violations == 0
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "total_pairs": len(pairs),
        "unique_triples": len(seen_triples),
        "cross_split_violations": cross_split_violations,
        "duplicate_triple_violations": duplicate_triple_violations,
        "decision_violations": decision_violations,
        "scenario_label_violations": scenario_label_violations,
        "manifest_membership_violations": manifest_membership_violations,
        "errors": errors[:50],  # Cap error list to prevent huge outputs
    }


def run_pairing_self_check(
    seed: int = 27,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run an in-memory self-check using synthetic fixtures.

    Tests split isolation, seed reproducibility, empty pools, duplicate
    prevention, and decision rules on synthetic fixtures. Writes results
    to results/protocol/pairing_self_check.json by default.

    Parameters
    ----------
    seed:
        Seed for deterministic generation.
    output_path:
        Optional output JSON path.

    Returns
    -------
    dict[str, Any]
        Self-check summary dictionary.

    """
    logger.info("Running pairing self-check (seed=%d)...", seed)

    # 1. Build synthetic fixture records
    # Deliberately omit vehicle images to test empty pool handling
    fixture_records: list[ManifestRecord] = []
    splits = ["train", "val", "test"]

    for sp in splits:
        # Images: 2 human, 2 animal, 2 empty (NO vehicle images)
        for i in range(2):
            fixture_records.append(
                ManifestRecord(
                    path=f"raw/birdsai/{sp}/human_{i}.jpg",
                    modality="image",
                    source="birdsai",
                    label="human",
                    view="aerial_thermal",
                    site_or_device="cam1",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"seq_{sp}_human",
                    split=sp,
                )
            )
            fixture_records.append(
                ManifestRecord(
                    path=f"raw/serengeti/{sp}/animal_{i}.jpg",
                    modality="image",
                    source="serengeti",
                    label="animal",
                    view="camera_trap",
                    site_or_device="trap1",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"seq_{sp}_animal",
                    split=sp,
                )
            )
            fixture_records.append(
                ManifestRecord(
                    path=f"raw/cct/{sp}/empty_{i}.jpg",
                    modality="image",
                    source="cct",
                    label="empty",
                    view="camera_trap",
                    site_or_device="trap2",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"seq_{sp}_empty",
                    split=sp,
                )
            )

        # Audio: 2 gunshot, 2 background normal, 2 background thunder_wind
        for j in range(2):
            fixture_records.append(
                ManifestRecord(
                    path=f"raw/esc50/{sp}/gunshot_{j}.wav",
                    modality="audio",
                    source="esc50",
                    label="gunshot",
                    view="acoustic",
                    site_or_device="mic1",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"rec_{sp}_shot",
                    split=sp,
                )
            )
            fixture_records.append(
                ManifestRecord(
                    path=f"raw/esc50/{sp}/background_ambient_{j}.wav",
                    modality="audio",
                    source="esc50",
                    label="background",
                    view="acoustic",
                    site_or_device="mic2",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"rec_{sp}_ambient",
                    split=sp,
                )
            )
            fixture_records.append(
                ManifestRecord(
                    path=f"raw/esc50/{sp}/background_thunder_{j}.wav",
                    modality="audio",
                    source="esc50",
                    label="background",
                    view="acoustic",
                    site_or_device="mic3",
                    noise_tag="thunder_wind",
                    original_split=sp,
                    group_id=f"rec_{sp}_thunder",
                    split=sp,
                )
            )

    # Validate all fixture records
    for r in fixture_records:
        r.validate()

    # 2. Run pairing (run 1)
    pairs_run1, stats_run1 = generate_all_pairs(
        records=fixture_records,
        seed=seed,
        splits=splits,
        max_pairs_per_scenario=10,
        max_pairs_per_split=100,
        min_images_per_class=0,
    )

    # 3. Run pairing (run 2) with identical seed to test reproducibility
    pairs_run2, _ = generate_all_pairs(
        records=fixture_records,
        seed=seed,
        splits=splits,
        max_pairs_per_scenario=10,
        max_pairs_per_split=100,
        min_images_per_class=0,
    )

    seed_reproducible = pairs_run1 == pairs_run2

    # 4. Audit pairs
    audit = verify_pairs(pairs_run1, manifest_records=fixture_records)

    # 5. Verify vehicle pools produced 0 pairs honestly
    vehicle_gunshot_total = stats_run1["scenario_totals"].get("vehicle_gunshot", -1)
    vehicle_background_total = stats_run1["scenario_totals"].get("vehicle_background", -1)
    empty_pools_handled = (vehicle_gunshot_total == 0) and (vehicle_background_total == 0)

    all_passed = (
        audit["status"] == "PASS"
        and seed_reproducible
        and empty_pools_handled
        and audit["cross_split_violations"] == 0
        and audit["duplicate_triple_violations"] == 0
        and audit["decision_violations"] == 0
        and audit["scenario_label_violations"] == 0
        and audit["manifest_membership_violations"] == 0
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "PASS" if all_passed else "FAIL",
        "not_a_dataset_evaluation": True,
        "seed": seed,
        "cross_split_pairs": audit["cross_split_violations"],
        "duplicate_triples": audit["duplicate_triple_violations"],
        "seed_reproducible": seed_reproducible,
        "empty_pools_handled": empty_pools_handled,
        "vehicle_gunshot_count": vehicle_gunshot_total,
        "vehicle_background_count": vehicle_background_total,
        "total_pairs_generated": len(pairs_run1),
        "per_split": stats_run1["per_split"],
        "scenario_totals": stats_run1["scenario_totals"],
        "audit_details": audit,
    }

    if output_path is not None:
        target_path = Path(output_path)
    else:
        target_path = Path("results/protocol/pairing_self_check.json")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Pairing self-check saved to %s (status: %s)", target_path, report["status"])
    return report
