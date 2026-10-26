"""
Unit tests for GuardianEye synthetic multimodal pairing (Phase 4 Stage 1).

Tests:
- Split isolation (train/val/test never cross)
- Duplicate prevention (exact (image_path, audio_path, scenario) never repeated)
- Honest empty class handling (vehicle or other missing class produces 0 pairs)
- Seed determinism (same seed produces identical pairs, different seeds vary)
- Scenario semantics & decision rules (matches src.decisions for all scenarios)
- BDA compatibility (scenarios.csv passes ScenarioMapper without errors)
- NOT YET RUN status handling on unpopulated manifests
- In-memory pairing self-check protocol
- CLI invocation
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_pairing import build_parser, main, run_pairing_for_seed
from src.bda.mapper_scenario import ScenarioMapper
from src.bda.stream import run_scenario_counts
from src.config import load_config
from src.decisions import risk, suspicious
from src.manifest import ManifestRecord
from src.pairing import (
    CANONICAL_SCENARIOS,
    PAIRS_CSV_COLUMNS,
    SCENARIOS_CSV_COLUMNS,
    SyntheticPair,
    generate_all_pairs,
    generate_split_pairs,
    pair_scenario,
    records_from_dataframe,
    run_pairing_self_check,
    save_scenarios_csv,
    verify_pairs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_manifest_records() -> list[ManifestRecord]:
    """Provide a minimal balanced set of ManifestRecords across splits."""
    records: list[ManifestRecord] = []
    splits = ["train", "val", "test"]

    for sp in splits:
        # Images: human, vehicle, animal, empty (2 of each)
        for i in range(2):
            records.append(
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
            records.append(
                ManifestRecord(
                    path=f"raw/belize/{sp}/vehicle_{i}.jpg",
                    modality="image",
                    source="belize",
                    label="vehicle",
                    view="camera_trap",
                    site_or_device="trap_v",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"seq_{sp}_vehicle",
                    split=sp,
                )
            )
            records.append(
                ManifestRecord(
                    path=f"raw/serengeti/{sp}/animal_{i}.jpg",
                    modality="image",
                    source="serengeti",
                    label="animal",
                    view="camera_trap",
                    site_or_device="trap_a",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"seq_{sp}_animal",
                    split=sp,
                )
            )
            records.append(
                ManifestRecord(
                    path=f"raw/cct/{sp}/empty_{i}.jpg",
                    modality="image",
                    source="cct",
                    label="empty",
                    view="camera_trap",
                    site_or_device="trap_e",
                    noise_tag="not_thunder_wind",
                    original_split=sp,
                    group_id=f"seq_{sp}_empty",
                    split=sp,
                )
            )

        # Audio: gunshot, background normal, background thunder_wind (2 of each)
        for j in range(2):
            records.append(
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
            records.append(
                ManifestRecord(
                    path=f"raw/esc50/{sp}/background_{j}.wav",
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
            records.append(
                ManifestRecord(
                    path=f"raw/esc50/{sp}/thunder_{j}.wav",
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

    for r in records:
        r.validate()

    return records


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestSplitIsolation:
    """Verify that pairing never mixes samples across splits."""

    def test_strict_split_isolation(self, sample_manifest_records: list[ManifestRecord]):
        pairs, _ = generate_all_pairs(
            sample_manifest_records,
            seed=27,
            splits=["train", "val", "test"],
            max_pairs_per_scenario=5,
        )

        assert len(pairs) > 0
        path_to_split = {r.path: r.split for r in sample_manifest_records}

        for p in pairs:
            img_split = path_to_split[p.image_path]
            aud_split = path_to_split[p.audio_path]
            assert img_split == p.split, f"Image split mismatch: {img_split} != {p.split}"
            assert aud_split == p.split, f"Audio split mismatch: {aud_split} != {p.split}"

        # Verification audit should also pass
        audit = verify_pairs(pairs, manifest_records=sample_manifest_records)
        assert audit["status"] == "PASS"
        assert audit["cross_split_violations"] == 0

    def test_audit_detects_artificially_leaked_pairs(
        self, sample_manifest_records: list[ManifestRecord]
    ):
        pairs, _ = generate_all_pairs(sample_manifest_records, seed=27, max_pairs_per_scenario=2)

        # Artificially alter one pair's declared split
        bad_pair = SyntheticPair(
            pair_id="corrupt_pair_001",
            split="train",  # Declared as train, but audio is test
            scenario="human_gunshot",
            image_path="raw/birdsai/train/human_0.jpg",
            audio_path="raw/esc50/test/gunshot_0.wav",  # Leaked from test!
            image_source="birdsai",
            audio_source="esc50",
            image_label="human",
            audio_label="gunshot",
            audio_noise_tag="not_thunder_wind",
            image_view="aerial_thermal",
            image_group_id="g1",
            audio_group_id="g2",
            suspicious=1,
            risk="HIGH",
        )
        corrupted_pairs = list(pairs) + [bad_pair]

        audit = verify_pairs(corrupted_pairs, manifest_records=sample_manifest_records)
        assert audit["status"] == "FAIL"
        assert audit["cross_split_violations"] == 1


class TestDuplicatePrevention:
    """Verify duplicate (image_path, audio_path, scenario) triples are prevented."""

    def test_no_duplicate_triples_when_budget_exceeds_pool(self):
        # 2 images and 2 audios = at most 4 unique pairs
        images = [
            ManifestRecord(
                path=f"raw/img_{i}.jpg",
                modality="image",
                source="birdsai",
                label="human",
                view="aerial_thermal",
                site_or_device="cam",
                noise_tag="not_thunder_wind",
                original_split="train",
                group_id=f"g_img_{i}",
                split="train",
            )
            for i in range(2)
        ]
        audios = [
            ManifestRecord(
                path=f"raw/aud_{j}.wav",
                modality="audio",
                source="esc50",
                label="gunshot",
                view="acoustic",
                site_or_device="mic",
                noise_tag="not_thunder_wind",
                original_split="train",
                group_id=f"g_aud_{j}",
                split="train",
            )
            for j in range(2)
        ]

        sc = CANONICAL_SCENARIOS[0]  # human_gunshot
        # Request 100 pairs, but only 4 exist
        pairs = pair_scenario(images, audios, sc, split="train", seed=42, max_pairs=100)

        assert len(pairs) == 4
        seen_triples = {(p.image_path, p.audio_path, p.scenario) for p in pairs}
        assert len(seen_triples) == 4

        audit = verify_pairs(pairs)
        assert audit["status"] == "PASS"
        assert audit["duplicate_triple_violations"] == 0

    def test_audit_flags_duplicate_triples(self):
        pair = SyntheticPair(
            pair_id="p1",
            split="train",
            scenario="animal_gunshot",
            image_path="img1.jpg",
            audio_path="aud1.wav",
            image_source="serengeti",
            audio_source="esc50",
            image_label="animal",
            audio_label="gunshot",
            audio_noise_tag="not_thunder_wind",
            image_view="camera_trap",
            image_group_id="g1",
            audio_group_id="g2",
            suspicious=0,
            risk="MEDIUM",
        )
        audit = verify_pairs([pair, pair])  # duplicate!
        assert audit["status"] == "FAIL"
        assert audit["duplicate_triple_violations"] == 1


class TestEmptyClasses:
    """Verify missing candidate pools produce zero-count scenarios honestly."""

    def test_empty_vehicle_pool_produces_zero_pairs(
        self, sample_manifest_records: list[ManifestRecord]
    ):
        # Filter out all vehicles
        no_vehicles = [r for r in sample_manifest_records if r.label != "vehicle"]

        pairs, stats = generate_all_pairs(
            no_vehicles,
            seed=27,
            splits=["train", "val", "test"],
            max_pairs_per_scenario=10,
        )

        assert stats["scenario_totals"]["vehicle_gunshot"] == 0
        assert stats["scenario_totals"]["vehicle_background"] == 0
        assert "vehicle_gunshot" in stats["empty_scenarios_overall"]
        assert "vehicle_background" in stats["empty_scenarios_overall"]

        # Ensure no pair contains vehicle
        for p in pairs:
            assert p.scenario not in {"vehicle_gunshot", "vehicle_background"}
            assert p.image_label != "vehicle"

    def test_min_images_per_class_threshold(self, sample_manifest_records: list[ManifestRecord]):
        # Our sample fixture has only 2 images per class
        # Enforcing min_images_per_class = 5 should write 0 pairs for all scenarios
        pairs, stats = generate_all_pairs(
            sample_manifest_records,
            seed=27,
            min_images_per_class=5,
        )
        assert len(pairs) == 0
        assert all(count == 0 for count in stats["scenario_totals"].values())


class TestDeterministicSeeds:
    """Verify that pairings are exactly reproducible given the same seed."""

    def test_seed_reproducibility(self, sample_manifest_records: list[ManifestRecord]):
        pairs_seed_a1, _ = generate_all_pairs(sample_manifest_records, seed=2701)
        pairs_seed_a2, _ = generate_all_pairs(sample_manifest_records, seed=2701)
        pairs_seed_b, _ = generate_all_pairs(sample_manifest_records, seed=26009)

        assert pairs_seed_a1 == pairs_seed_a2
        # Different seeds should yield different ordering of pairs
        a_paths = [(p.image_path, p.audio_path) for p in pairs_seed_a1]
        b_paths = [(p.image_path, p.audio_path) for p in pairs_seed_b]
        assert a_paths != b_paths


class TestScenarioSemantics:
    """Verify that all scenarios match canonical labels, suspicious flags, and risk levels."""

    def test_all_canonical_scenarios_match_decisions(
        self, sample_manifest_records: list[ManifestRecord]
    ):
        pairs, _ = generate_all_pairs(sample_manifest_records, seed=27, max_pairs_per_scenario=2)
        scenarios_seen = {p.scenario for p in pairs}
        assert len(scenarios_seen) == 8  # all 8 canonical scenarios represented

        for p in pairs:
            assert p.suspicious == suspicious(p.image_label, p.audio_label)
            assert p.risk == risk(p.image_label, p.audio_label)

            if p.scenario == "human_gunshot":
                assert p.image_label == "human" and p.audio_label == "gunshot"
                assert p.suspicious == 1 and p.risk == "HIGH"
            elif p.scenario == "vehicle_gunshot":
                assert p.image_label == "vehicle" and p.audio_label == "gunshot"
                assert p.suspicious == 1 and p.risk == "HIGH"
            elif p.scenario == "animal_gunshot":
                assert p.image_label == "animal" and p.audio_label == "gunshot"
                assert p.suspicious == 0 and p.risk == "MEDIUM"
            elif p.scenario == "human_background":
                assert p.image_label == "human" and p.audio_label == "background"
                assert p.suspicious == 0 and p.risk == "MEDIUM"
            elif p.scenario == "vehicle_background":
                assert p.image_label == "vehicle" and p.audio_label == "background"
                assert p.suspicious == 0 and p.risk == "MEDIUM"
            elif p.scenario == "animal_background":
                assert p.image_label == "animal" and p.audio_label == "background"
                assert p.suspicious == 0 and p.risk == "LOW"
            elif p.scenario == "empty_background":
                assert p.image_label == "empty" and p.audio_label == "background"
                assert p.audio_noise_tag == "not_thunder_wind"
                assert p.suspicious == 0 and p.risk == "LOW"
            elif p.scenario == "empty_thunder":
                assert p.image_label == "empty" and p.audio_label == "background"
                assert p.audio_noise_tag == "thunder_wind"
                assert p.suspicious == 0 and p.risk == "LOW"


class TestBDACompatibility:
    """Verify that scenarios.csv works seamlessly with Phase 3 BDA jobs."""

    def test_scenarios_csv_schema_and_mapper(
        self, sample_manifest_records: list[ManifestRecord], tmp_path: Path
    ):
        pairs, _ = generate_all_pairs(sample_manifest_records, seed=27, max_pairs_per_scenario=3)
        scenarios_csv = tmp_path / "scenarios.csv"
        save_scenarios_csv(pairs, scenarios_csv)

        assert scenarios_csv.exists()

        # Parse with Phase 3 ScenarioMapper
        mapper = ScenarioMapper()
        with scenarios_csv.open("r", encoding="utf-8") as fh:
            header = fh.readline()
            mapper.set_header(header)
            all_emitted = []
            for line in fh:
                all_emitted.extend(mapper.map(line))

        assert mapper.rows_processed == len(pairs)
        assert mapper.rows_skipped == 0
        assert len(all_emitted) == len(pairs)

    def test_scenarios_csv_with_bda_stream(
        self, sample_manifest_records: list[ManifestRecord], tmp_path: Path
    ):
        pairs, _ = generate_all_pairs(sample_manifest_records, seed=27, max_pairs_per_scenario=2)
        scenarios_csv = tmp_path / "scenarios.csv"
        counts_csv = tmp_path / "scenario_counts.csv"
        save_scenarios_csv(pairs, scenarios_csv)

        # Run Phase 3 streaming job
        res = run_scenario_counts(scenarios_csv, counts_csv)
        assert res["status"] == "OK"
        assert counts_csv.exists()

        counts_df = pd.read_csv(counts_csv)
        assert set(counts_df.columns) == {"split", "scenario", "suspicious", "risk", "count"}
        assert counts_df["count"].sum() == len(pairs)


class TestNotYetRunHandling:
    """Verify clean exit and status reporting when manifests are missing or NOT YET RUN."""

    def test_unpopulated_manifest_status_handling(self, tmp_path: Path):
        cfg = load_config(results_root=tmp_path / "results")
        run_dir = cfg.paths.results_root / "runs" / "seed_27"
        manifest_dir = run_dir / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        status_file = manifest_dir / "manifest_status.json"
        with status_file.open("w", encoding="utf-8") as f:
            json.dump({"seed": 27, "status": "NOT YET RUN", "reason": "No raw datasets"}, f)

        res = run_pairing_for_seed(
            cfg=cfg,
            seed=27,
            max_pairs_per_scenario=10,
            max_pairs_per_split=100,
            min_images_per_class=0,
        )

        assert res["status"] == "NOT YET RUN"
        pairing_status = run_dir / "pairing" / "pairing_status.json"
        assert pairing_status.exists()

        with pairing_status.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "NOT YET RUN"
        assert "NOT YET RUN" in data["reason"]

        # Empty schema CSVs must be present
        scenarios_csv = run_dir / "pairing" / "scenarios.csv"
        pairs_csv = run_dir / "pairing" / "pairs.csv"
        assert scenarios_csv.exists()
        assert pairs_csv.exists()

        sc_df = pd.read_csv(scenarios_csv)
        assert list(sc_df.columns) == SCENARIOS_CSV_COLUMNS
        assert len(sc_df) == 0

        p_df = pd.read_csv(pairs_csv)
        assert list(p_df.columns) == PAIRS_CSV_COLUMNS
        assert len(p_df) == 0


class TestSelfCheckProtocol:
    """Verify the in-memory pairing self-check routine."""

    def test_run_pairing_self_check(self, tmp_path: Path):
        out_json = tmp_path / "protocol" / "pairing_self_check.json"
        report = run_pairing_self_check(seed=27, output_path=out_json)

        assert report["status"] == "PASS"
        assert report["not_a_dataset_evaluation"] is True
        assert report["cross_split_pairs"] == 0
        assert report["duplicate_triples"] == 0
        assert report["seed_reproducible"] is True
        assert report["empty_pools_handled"] is True
        assert report["vehicle_gunshot_count"] == 0
        assert report["vehicle_background_count"] == 0
        assert out_json.exists()


class TestCliAndUtilities:
    """Verify CLI parser and helper functions."""

    def test_cli_parser(self):
        parser = build_parser()
        args = parser.parse_args(["--smoke", "--seeds", "27", "2701", "--self-check"])
        assert args.smoke is True
        assert args.self_check is True
        assert args.seeds == [27, 2701]

    def test_records_from_dataframe(self, sample_manifest_records: list[ManifestRecord]):
        data = [
            {
                "path": r.path,
                "modality": r.modality,
                "source": r.source,
                "label": r.label,
                "view": r.view,
                "site_or_device": r.site_or_device,
                "noise_tag": r.noise_tag,
                "original_split": r.original_split,
                "group_id": r.group_id,
                "split": r.split,
                "notes": r.notes,
            }
            for r in sample_manifest_records[:5]
        ]
        df = pd.DataFrame(data)
        recs = records_from_dataframe(df)
        assert len(recs) == 5
        assert recs[0].path == sample_manifest_records[0].path

    def test_cli_self_check_exits_without_per_seed_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """CLI --self-check executes the protocol check and returns without per-seed runs."""
        results_dir = tmp_path / "results"
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_pairing.py",
                "--self-check",
                "--results-root",
                str(results_dir),
            ],
        )
        ret = main()
        assert ret == 0
        assert (results_dir / "protocol" / "pairing_self_check.json").exists()
        # Verify that normal runs directory was NOT created
        assert (results_dir / "runs").exists() is False


class TestBudgetEnforcement:
    """Verify enforcement of max_pairs_per_scenario and max_pairs_per_split."""

    def test_max_pairs_per_scenario_enforced(
        self, sample_manifest_records: list[ManifestRecord]
    ):
        pairs, counts = generate_split_pairs(
            sample_manifest_records,
            split="train",
            seed=27,
            max_pairs_per_scenario=1,
            max_pairs_per_split=1000,
        )
        for sc_name, count in counts.items():
            assert count <= 1, f"Scenario {sc_name} exceeded max_pairs_per_scenario=1"

    def test_max_pairs_per_split_enforced(
        self, sample_manifest_records: list[ManifestRecord]
    ):
        pairs, counts = generate_split_pairs(
            sample_manifest_records,
            split="train",
            seed=27,
            max_pairs_per_scenario=10,
            max_pairs_per_split=5,
        )
        assert len(pairs) <= 5
        assert sum(counts.values()) <= 5

    def test_deterministic_across_multiple_runs(
        self, sample_manifest_records: list[ManifestRecord]
    ):
        baseline, _ = generate_all_pairs(sample_manifest_records, seed=2701)
        for _ in range(5):
            rerun, _ = generate_all_pairs(sample_manifest_records, seed=2701)
            assert rerun == baseline


class TestVerificationAudit:
    """Verify that verify_pairs audits scenario labels, decisions, and manifest membership."""

    def test_audit_flags_scenario_label_mismatch(self):
        # Pair claims human_gunshot, but image_label is animal
        pair = SyntheticPair(
            pair_id="p1",
            split="train",
            scenario="human_gunshot",
            image_path="img.jpg",
            audio_path="aud.wav",
            image_source="birdsai",
            audio_source="esc50",
            image_label="animal",  # mismatch with human_gunshot!
            audio_label="gunshot",
            audio_noise_tag="not_thunder_wind",
            image_view="aerial_thermal",
            image_group_id="g1",
            audio_group_id="g2",
            suspicious=0,
            risk="MEDIUM",
        )
        audit = verify_pairs([pair])
        assert audit["status"] == "FAIL"
        assert audit["scenario_label_violations"] == 1

    def test_audit_flags_unknown_scenario(self):
        pair = SyntheticPair(
            pair_id="p1",
            split="train",
            scenario="alien_laser",  # unknown scenario
            image_path="img.jpg",
            audio_path="aud.wav",
            image_source="birdsai",
            audio_source="esc50",
            image_label="human",
            audio_label="gunshot",
            audio_noise_tag="not_thunder_wind",
            image_view="aerial_thermal",
            image_group_id="g1",
            audio_group_id="g2",
            suspicious=1,
            risk="HIGH",
        )
        audit = verify_pairs([pair])
        assert audit["status"] == "FAIL"
        assert audit["scenario_label_violations"] == 1

    def test_audit_flags_missing_manifest_record(self):
        pair = SyntheticPair(
            pair_id="p1",
            split="train",
            scenario="human_gunshot",
            image_path="missing_img.jpg",
            audio_path="aud.wav",
            image_source="birdsai",
            audio_source="esc50",
            image_label="human",
            audio_label="gunshot",
            audio_noise_tag="not_thunder_wind",
            image_view="aerial_thermal",
            image_group_id="g1",
            audio_group_id="g2",
            suspicious=1,
            risk="HIGH",
        )
        manifest_rec = ManifestRecord(
            path="aud.wav",
            modality="audio",
            source="esc50",
            label="gunshot",
            view="acoustic",
            site_or_device="mic",
            noise_tag="not_thunder_wind",
            original_split="train",
            group_id="g2",
            split="train",
        )
        audit = verify_pairs([pair], manifest_records=[manifest_rec])
        assert audit["status"] == "FAIL"
        assert audit["manifest_membership_violations"] >= 1

    def test_audit_flags_manifest_modality_mismatch(self):
        pair = SyntheticPair(
            pair_id="p1",
            split="train",
            scenario="human_gunshot",
            image_path="img.jpg",
            audio_path="aud.wav",
            image_source="birdsai",
            audio_source="esc50",
            image_label="human",
            audio_label="gunshot",
            audio_noise_tag="not_thunder_wind",
            image_view="aerial_thermal",
            image_group_id="g1",
            audio_group_id="g2",
            suspicious=1,
            risk="HIGH",
        )
        # Put img.jpg into manifest as modality audio instead of image
        bad_img_rec = ManifestRecord(
            path="img.jpg",
            modality="audio",  # Wrong modality
            source="birdsai",
            label="human",
            view="acoustic",
            site_or_device="cam",
            noise_tag="not_thunder_wind",
            original_split="train",
            group_id="g1",
            split="train",
        )
        aud_rec = ManifestRecord(
            path="aud.wav",
            modality="audio",
            source="esc50",
            label="gunshot",
            view="acoustic",
            site_or_device="mic",
            noise_tag="not_thunder_wind",
            original_split="train",
            group_id="g2",
            split="train",
        )
        audit = verify_pairs([pair], manifest_records=[bad_img_rec, aud_rec])
        assert audit["status"] == "FAIL"
        assert audit["manifest_membership_violations"] >= 1

    def test_audit_flags_decision_mismatch(self):
        pair = SyntheticPair(
            pair_id="p1",
            split="train",
            scenario="human_gunshot",
            image_path="img.jpg",
            audio_path="aud.wav",
            image_source="birdsai",
            audio_source="esc50",
            image_label="human",
            audio_label="gunshot",
            audio_noise_tag="not_thunder_wind",
            image_view="aerial_thermal",
            image_group_id="g1",
            audio_group_id="g2",
            suspicious=0,  # Wrong! Should be 1
            risk="LOW",    # Wrong! Should be HIGH
        )
        audit = verify_pairs([pair])
        assert audit["status"] == "FAIL"
        assert audit["decision_violations"] == 1

