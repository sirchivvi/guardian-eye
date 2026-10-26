"""Unit tests for BDA reducer modules."""

import math

import pytest

from src.bda.reducer_metric import MetricReducer
from src.bda.reducer_sum import SumReducer

# ---- SumReducer ----


class TestSumReducer:
    """Tests for the summation reducer."""

    def test_basic_sum(self):
        reducer = SumReducer()
        results = list(reducer.reduce("image\tbirdsai\thuman", iter(["1", "1", "1"])))
        assert len(results) == 1
        assert results[0] == "image\tbirdsai\thuman\t3"

    def test_single_value(self):
        reducer = SumReducer()
        results = list(reducer.reduce("audio\tbelize\tgunshot", iter(["1"])))
        assert results[0] == "audio\tbelize\tgunshot\t1"

    def test_non_integer_skipped(self):
        reducer = SumReducer()
        results = list(reducer.reduce("test_key", iter(["1", "bad", "2"])))
        assert results[0] == "test_key\t3"

    def test_empty_values(self):
        reducer = SumReducer()
        results = list(reducer.reduce("test_key", iter([])))
        assert results[0] == "test_key\t0"


# ---- MetricReducer ----


class TestMetricReducer:
    """Tests for the metric reducer."""

    def test_two_values(self):
        reducer = MetricReducer()
        results = list(
            reducer.reduce("image_only\tclean\tf1", iter(["0.72", "0.74"]))
        )
        assert len(results) == 1
        parts = results[0].split("\t")
        assert parts[0] == "image_only"
        assert parts[1] == "clean"
        assert parts[2] == "f1"
        assert parts[3] == "2"  # n
        assert float(parts[4]) == pytest.approx(0.73, abs=1e-10)  # mean
        # sample_std = sqrt(sum((x - mean)^2) / (n-1))
        # = sqrt(((0.72-0.73)^2 + (0.74-0.73)^2) / 1) = sqrt(0.0002) ≈ 0.01414
        expected_std = math.sqrt(((0.72 - 0.73) ** 2 + (0.74 - 0.73) ** 2) / 1)
        assert float(parts[5]) == pytest.approx(expected_std, abs=1e-10)

    def test_single_value_empty_std(self):
        """When n < 2, sample_std should be empty string."""
        reducer = MetricReducer()
        results = list(
            reducer.reduce("late_fusion\tclean\tf1", iter(["0.80"]))
        )
        assert len(results) == 1
        parts = results[0].split("\t")
        assert parts[3] == "1"  # n
        assert float(parts[4]) == pytest.approx(0.80, abs=1e-10)
        assert parts[5] == ""  # empty std

    def test_empty_values_no_output(self):
        """When no valid values are provided, no output is emitted."""
        reducer = MetricReducer()
        results = list(reducer.reduce("test_key", iter([])))
        assert results == []

    def test_non_numeric_skipped(self):
        reducer = MetricReducer()
        results = list(
            reducer.reduce("test_key", iter(["0.5", "bad", "0.7"]))
        )
        # Output format: "key\tn\tmean\tsample_std"
        # But key here is "test_key" (no tabs), so output is "test_key\t2\t0.6\t..."
        output = results[0]
        # Split on tabs — key may itself contain tabs when used in practice
        # For this test, key = "test_key" so we get: [test_key, 2, 0.6, std]
        parts = output.split("\t")
        # test_key is 1 field, then n, mean, std
        assert parts[1] == "2"  # n = 2 (the non-numeric is skipped)
        assert float(parts[2]) == pytest.approx(0.6, abs=1e-10)

    def test_three_values(self):
        reducer = MetricReducer()
        values = ["0.70", "0.80", "0.75"]
        results = list(reducer.reduce("method\tcondition\tmetric", iter(values)))
        parts = results[0].split("\t")
        assert parts[3] == "3"
        mean = (0.70 + 0.80 + 0.75) / 3
        assert float(parts[4]) == pytest.approx(mean, abs=1e-10)
        variance = sum((x - mean) ** 2 for x in [0.70, 0.80, 0.75]) / 2
        expected_std = math.sqrt(variance)
        assert float(parts[5]) == pytest.approx(expected_std, abs=1e-10)
