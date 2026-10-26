"""CLI smoke tests for the BDA module."""

import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestBdaCli:
    """Tests for the run_bda.py CLI script."""

    def test_self_check(self, tmp_path):
        """--self-check runs on fixture data and validates outputs."""
        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_bda.py"),
                "--self-check",
                "--results-root",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASSED" in result.stdout.upper()

        # Verify status JSON was written
        status_path = tmp_path / "bda" / "bda_status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["validation_passed"] is True
        assert status["engine"] in ("hadoop_streaming", "python_streaming_local")

    def test_self_check_creates_output_files(self, tmp_path):
        """--self-check creates the expected output CSVs."""
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_bda.py"),
                "--self-check",
                "--results-root",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=60,
        )
        self_check_dir = tmp_path / "bda" / "self_check"
        assert (self_check_dir / "class_counts_check.csv").exists()
        assert (self_check_dir / "scenario_counts_check.csv").exists()
        assert (self_check_dir / "metric_summary_check.csv").exists()

    def test_class_counts_mode_missing_data(self, tmp_path):
        """Running class_counts on missing data exits cleanly."""
        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_bda.py"),
                "--mode",
                "class_counts",
                "--results-root",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=60,
        )
        # Should not crash even with missing data
        assert result.returncode == 0

        # Verify status JSON
        status_path = tmp_path / "bda" / "bda_status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["status"] in ("NOT YET RUN", "PARTIAL", "OK")

    def test_help_flag(self):
        """--help exits cleanly."""
        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_bda.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=30,
        )
        assert result.returncode == 0
        assert "BDA" in result.stdout or "bda" in result.stdout.lower()
