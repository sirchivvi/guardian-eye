"""Tests for BDA Spark jobs (optional dependency)."""

import pytest

from src.bda import spark_jobs


class TestSparkAvailability:
    """Tests for PySpark availability detection."""

    def test_get_spark_status(self):
        """get_spark_status returns a valid status dict."""
        status = spark_jobs.get_spark_status()
        assert "status" in status
        if spark_jobs.PYSPARK_AVAILABLE:
            assert status["status"] == "OK"
            assert status["pyspark_available"] is True
        else:
            assert status["status"] == "NOT YET RUN"
            assert "pyspark" in status["reason"].lower()

    def test_unexecuted_spark_never_reported_as_successful(self):
        """When PySpark is absent, status must NOT be reported as OK or executed."""
        if not spark_jobs.PYSPARK_AVAILABLE:
            status = spark_jobs.get_spark_status()
            assert status["status"] == "NOT YET RUN"
            assert status.get("pyspark_available") is not True


class TestSparkNotInstalled:
    """Tests that verify graceful handling when PySpark is absent."""

    def test_manifest_to_parquet_without_spark(self, tmp_path):
        """manifest_to_parquet returns NOT YET RUN when pyspark is absent."""
        if spark_jobs.PYSPARK_AVAILABLE:
            pytest.skip("PySpark is available; this test is for absent-pyspark scenario")

        result = spark_jobs.manifest_to_parquet(
            tmp_path / "manifest.csv",
            tmp_path / "output.parquet",
        )
        assert result["status"] == "NOT YET RUN"
        assert "pyspark" in result["reason"].lower()

    def test_compute_class_weights_without_spark(self, tmp_path):
        """compute_class_weights returns NOT YET RUN when pyspark is absent."""
        if spark_jobs.PYSPARK_AVAILABLE:
            pytest.skip("PySpark is available; this test is for absent-pyspark scenario")

        result = spark_jobs.compute_class_weights(
            tmp_path / "manifest.csv",
            tmp_path / "output.parquet",
        )
        assert result["status"] == "NOT YET RUN"

    def test_compute_audio_features_without_spark(self, tmp_path):
        """compute_audio_features returns NOT YET RUN when pyspark is absent."""
        if spark_jobs.PYSPARK_AVAILABLE:
            pytest.skip("PySpark is available; this test is for absent-pyspark scenario")

        result = spark_jobs.compute_audio_features(
            tmp_path / "manifest.csv",
            tmp_path / "output.parquet",
        )
        assert result["status"] == "NOT YET RUN"

    def test_run_all_spark_jobs_without_spark(self, tmp_path):
        """run_all_spark_jobs returns NOT YET RUN when pyspark is absent."""
        if spark_jobs.PYSPARK_AVAILABLE:
            pytest.skip("PySpark is available; this test is for absent-pyspark scenario")

        result = spark_jobs.run_all_spark_jobs(
            tmp_path / "manifest.csv",
            tmp_path / "spark_output",
        )
        assert result["status"] == "NOT YET RUN"
        assert result["spark_available"] is False


class TestSparkInstalled:
    """Tests that run when PySpark IS available."""

    @pytest.fixture(autouse=True)
    def _require_pyspark(self):
        if not spark_jobs.PYSPARK_AVAILABLE:
            pytest.skip("PySpark not installed")

    def test_manifest_to_parquet(self, tmp_path):
        """manifest_to_parquet writes a valid Parquet file."""
        from pathlib import Path

        fixture = Path(__file__).parent / "fixtures" / "bda" / "fixture_manifest.csv"
        output = tmp_path / "manifest.parquet"
        result = spark_jobs.manifest_to_parquet(fixture, output)
        assert result["status"] == "OK"
        assert result["row_count"] == 12

    def test_compute_class_weights(self, tmp_path):
        """compute_class_weights produces valid weight records."""
        from pathlib import Path

        fixture = Path(__file__).parent / "fixtures" / "bda" / "fixture_manifest.csv"
        output = tmp_path / "class_weights.parquet"
        result = spark_jobs.compute_class_weights(fixture, output)
        assert result["status"] == "OK"
        assert len(result["weights"]) > 0

        # Verify weight formula: w_c = N / (k * n_c)
        for w in result["weights"]:
            expected = w["N"] / (w["k"] * w["n_c"])
            assert abs(w["weight"] - round(expected, 6)) < 1e-4

    def test_missing_manifest_spark(self, tmp_path):
        """Spark jobs handle missing manifest gracefully."""
        result = spark_jobs.manifest_to_parquet(
            tmp_path / "nonexistent.csv",
            tmp_path / "output.parquet",
        )
        assert result["status"] == "NOT YET RUN"
