from pathlib import Path

from src.bda.mapper_class import ClassCountMapper
from src.bda.mapper_metric import MetricMapper
from src.bda.mapper_scenario import ScenarioMapper

_FIXTURES = Path(__file__).parent / "fixtures" / "bda"


# ---- ClassCountMapper ----


class TestClassCountMapper:
    """Tests for the class-count mapper."""

    def test_basic_emit(self):
        mapper = ClassCountMapper()
        header = (
            "path,modality,source,label,view,site_or_device,noise_tag,"
            "original_split,group_id,split,notes"
        )
        mapper.set_header(header)

        line = (
            "raw/birdsai/frames/seq1_00001.jpg,image,birdsai,human,"
            "aerial_thermal,site_alpha,not_thunder_wind,train,birdsai_seq1,train,"
        )
        results = list(mapper.map(line))
        assert len(results) == 1
        key, value = results[0]
        assert key == "image\tbirdsai\thuman"
        assert value == "1"

    def test_all_fixture_rows(self):
        mapper = ClassCountMapper()
        fixture_path = _FIXTURES / "fixture_manifest.csv"
        with fixture_path.open() as fh:
            header = fh.readline()
            mapper.set_header(header)
            all_pairs = []
            for line in fh:
                all_pairs.extend(mapper.map(line))

        assert mapper.rows_processed == 12
        assert mapper.rows_skipped == 0
        assert len(all_pairs) == 12

    def test_empty_line_skipped(self):
        mapper = ClassCountMapper()
        header = "path,modality,source,label"
        mapper.set_header(header)

        results = list(mapper.map(""))
        assert results == []
        assert mapper.rows_skipped == 0  # empty lines are silently ignored

    def test_malformed_row_skipped(self):
        mapper = ClassCountMapper()
        header = "path,modality,source,label"
        mapper.set_header(header)

        # Too few fields
        results = list(mapper.map("only_one_field"))
        assert results == []
        assert mapper.rows_skipped == 1

    def test_empty_field_skipped(self):
        mapper = ClassCountMapper()
        header = "path,modality,source,label"
        mapper.set_header(header)

        results = list(mapper.map("some_path,,belize,gunshot"))
        assert results == []
        assert mapper.rows_skipped == 1

    def test_missing_or_unannotated_labels_not_converted_to_background(self):
        """Missing or unannotated labels are never converted to background/empty."""
        mapper = ClassCountMapper()
        header = (
            "path,modality,source,label,view,site_or_device,noise_tag,"
            "original_split,group_id,split,notes"
        )
        mapper.set_header(header)

        # Missing/empty label row should be skipped, NEVER mapped to 'background'
        empty_label_row = (
            "raw/birdsai/f1.jpg,image,birdsai,,aerial,s1,not_thunder_wind,train,g1,train,"
        )
        results = list(mapper.map(empty_label_row))
        assert results == []
        assert mapper.rows_skipped == 1

        # 'unannotated' label row must emit its exact label, never coerced to background
        unannotated_row = (
            "raw/birdsai/f2.jpg,image,birdsai,unannotated,aerial,s1,"
            "not_thunder_wind,train,g1,train,"
        )
        results = list(mapper.map(unannotated_row))
        assert len(results) == 1
        assert results[0][0] == "image\tbirdsai\tunannotated"
        assert "background" not in results[0][0]
        assert "empty" not in results[0][0]


# ---- ScenarioMapper ----


class TestScenarioMapper:
    """Tests for the scenario mapper."""

    def test_basic_emit(self):
        mapper = ScenarioMapper()
        header = "split,image_label,audio_label,scenario,suspicious,risk"
        mapper.set_header(header)

        line = "train,human,gunshot,human_gunshot,1,HIGH"
        results = list(mapper.map(line))
        assert len(results) == 1
        key, value = results[0]
        assert key == "train\thuman_gunshot\t1\tHIGH"
        assert value == "1"

    def test_all_fixture_rows(self):
        mapper = ScenarioMapper()
        fixture_path = _FIXTURES / "fixture_scenarios.csv"
        with fixture_path.open() as fh:
            header = fh.readline()
            mapper.set_header(header)
            all_pairs = []
            for line in fh:
                all_pairs.extend(mapper.map(line))

        assert mapper.rows_processed == 8
        assert mapper.rows_skipped == 0

    def test_empty_line_ignored(self):
        mapper = ScenarioMapper()
        header = "split,scenario,suspicious,risk"
        mapper.set_header(header)

        results = list(mapper.map("   "))
        assert results == []


# ---- MetricMapper ----


class TestMetricMapper:
    """Tests for the metric mapper."""

    def test_basic_emit(self):
        mapper = MetricMapper()
        line = (
            '{"method": "image_only", "condition": "clean", '
            '"metric": "f1", "value": 0.72, "seed": 27}'
        )
        results = list(mapper.map(line))
        assert len(results) == 1
        key, value = results[0]
        assert key == "image_only\tclean\tf1"
        assert value == "0.72"

    def test_null_value_skipped(self):
        mapper = MetricMapper()
        line = '{"method": "image_only", "condition": "clean", "metric": "f1", "value": null}'
        results = list(mapper.map(line))
        assert results == []
        assert mapper.rows_skipped == 1

    def test_missing_keys_skipped(self):
        mapper = MetricMapper()
        line = '{"method": "image_only", "metric": "f1"}'
        results = list(mapper.map(line))
        assert results == []
        assert mapper.rows_skipped == 1

    def test_invalid_json_skipped(self):
        mapper = MetricMapper()
        results = list(mapper.map("not valid json"))
        assert results == []
        assert mapper.rows_skipped == 1

    def test_all_fixture_entries(self):
        import json

        mapper = MetricMapper()
        fixture_path = _FIXTURES / "fixture_metrics.json"
        with fixture_path.open() as fh:
            records = json.load(fh)

        all_pairs = []
        for record in records:
            all_pairs.extend(mapper.map(json.dumps(record)))

        # All 6 entries have non-null values
        assert len(all_pairs) == 6
        assert mapper.rows_processed == 6
