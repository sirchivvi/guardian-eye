"""Integration tests for the BDA streaming engine."""

from pathlib import Path

import pytest

from src.bda import stream

_FIXTURES = Path(__file__).parent / "fixtures" / "bda"


class TestLocalStreaming:
    """End-to-end local streaming tests against fixture data."""

    def test_class_counts(self, tmp_path):
        """Class-count job matches expected golden output."""
        output_path = tmp_path / "class_counts.csv"
        result = stream.run_class_counts(
            _FIXTURES / "fixture_manifest.csv",
            output_path,
        )
        assert result["status"] == "OK"
        assert result["engine"] == "python_streaming_local"
        assert output_path.exists()

        expected_path = _FIXTURES / "expected_class_counts.csv"
        assert output_path.read_text().strip() == expected_path.read_text().strip()

    def test_scenario_counts(self, tmp_path):
        """Scenario-count job matches expected golden output."""
        output_path = tmp_path / "scenario_counts.csv"
        result = stream.run_scenario_counts(
            _FIXTURES / "fixture_scenarios.csv",
            output_path,
        )
        assert result["status"] == "OK"
        assert result["engine"] == "python_streaming_local"
        assert output_path.exists()

        expected_path = _FIXTURES / "expected_scenario_counts.csv"
        assert output_path.read_text().strip() == expected_path.read_text().strip()

    def test_metric_summary(self, tmp_path):
        """Metric-summary job matches expected golden output."""
        output_path = tmp_path / "metric_summary.csv"
        result = stream.run_metric_summary(
            _FIXTURES / "fixture_metrics.json",
            output_path,
        )
        assert result["status"] == "OK"
        assert result["engine"] == "python_streaming_local"
        assert output_path.exists()

        expected_path = _FIXTURES / "expected_metric_summary.csv"
        actual_lines = output_path.read_text().strip().split("\n")
        expected_lines = expected_path.read_text().strip().split("\n")

        # Compare headers
        assert actual_lines[0] == expected_lines[0]

        # Compare data rows with float tolerance
        assert len(actual_lines) == len(expected_lines)
        for actual, expected in zip(actual_lines[1:], expected_lines[1:], strict=True):
            actual_parts = actual.split(",")
            expected_parts = expected.split(",")
            # First 3 fields are strings, rest are numeric
            assert actual_parts[:3] == expected_parts[:3]
            assert actual_parts[3] == expected_parts[3]  # n
            assert float(actual_parts[4]) == pytest.approx(
                float(expected_parts[4]), abs=1e-6
            )
            # sample_std might be empty
            if expected_parts[5]:
                assert float(actual_parts[5]) == pytest.approx(
                    float(expected_parts[5]), abs=1e-6
                )
            else:
                assert actual_parts[5] == ""

    def test_missing_manifest(self, tmp_path):
        """Missing manifest produces NOT YET RUN status."""
        result = stream.run_class_counts(
            tmp_path / "nonexistent.csv",
            tmp_path / "output.csv",
        )
        assert result["status"] == "NOT YET RUN"
        assert "not found" in result["reason"].lower()

    def test_missing_scenarios(self, tmp_path):
        """Missing scenarios file produces NOT YET RUN status."""
        result = stream.run_scenario_counts(
            tmp_path / "nonexistent.csv",
            tmp_path / "output.csv",
        )
        assert result["status"] == "NOT YET RUN"

    def test_missing_metrics(self, tmp_path):
        """Missing metrics file produces NOT YET RUN status."""
        result = stream.run_metric_summary(
            tmp_path / "nonexistent.json",
            tmp_path / "output.csv",
        )
        assert result["status"] == "NOT YET RUN"

    def test_engine_name(self):
        """Engine name is reported correctly (should be local in test env)."""
        engine = stream.get_engine_name()
        assert engine in ("hadoop_streaming", "python_streaming_local")
        # In the test environment, we don't expect Hadoop
        assert engine == "python_streaming_local"

    def test_unexecuted_hadoop_never_reported_as_successful(self):
        """When Hadoop is not installed, engine must never report hadoop_streaming."""
        has_hadoop, _ = stream._detect_hadoop()
        engine = stream.get_engine_name()
        if not has_hadoop:
            assert engine != "hadoop_streaming"
            assert engine == "python_streaming_local"


class TestEmptyInputs:
    """Tests for empty input files."""

    def test_empty_manifest(self, tmp_path):
        """Manifest with only a header produces empty output."""
        manifest = tmp_path / "empty_manifest.csv"
        manifest.write_text(
            "path,modality,source,label,view,site_or_device,noise_tag,"
            "original_split,group_id,split,notes\n"
        )
        output = tmp_path / "counts.csv"
        result = stream.run_class_counts(manifest, output)
        assert result["status"] == "OK"
        assert output.exists()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1  # header only
