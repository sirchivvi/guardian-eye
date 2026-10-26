"""
CLI entry point for the GuardianEye BDA module.

Usage
-----
Run class counts::

    python scripts/run_bda.py --mode class_counts

Run all streaming jobs::

    python scripts/run_bda.py --mode all

Run self-check on fixtures::

    python scripts/run_bda.py --self-check

Run Spark jobs (requires pyspark)::

    python scripts/run_bda.py --mode spark
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bda import spark_jobs, stream  # noqa: E402
from src.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_SELF_CHECK_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "bda"


def _write_status(output_dir, status):
    """Write bda_status.json atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "bda_status.json"
    tmp_path = status_path.with_suffix(".tmp")
    with tmp_path.open("w") as fh:
        json.dump(status, fh, indent=2, default=str)
    tmp_path.replace(status_path)
    logger.info("BDA status written to %s", status_path)


def run_class_counts(manifest_path, output_dir):
    """Run class-count streaming job."""
    output_path = output_dir / "class_counts.csv"
    if manifest_path.exists():
        with manifest_path.open() as fh:
            lines = [line.strip() for line in fh if line.strip()]
            if len(lines) <= 1:
                return {
                    "status": "NOT YET RUN",
                    "reason": f"Manifest contains 0 data rows: {manifest_path}",
                    "engine": stream.get_engine_name(),
                }
    return stream.run_class_counts(manifest_path, output_path)


def run_scenario_counts(scenarios_path, output_dir):
    """Run scenario-count streaming job."""
    output_path = output_dir / "scenario_counts.csv"
    return stream.run_scenario_counts(scenarios_path, output_path)


def run_metric_summary(metrics_path, output_dir):
    """Run metric-summary streaming job."""
    output_path = output_dir / "metric_summary.csv"
    return stream.run_metric_summary(metrics_path, output_path)


def run_spark(manifest_path, output_dir, data_root=None):
    """Run all Spark jobs."""
    spark_dir = output_dir / "spark"
    return spark_jobs.run_all_spark_jobs(manifest_path, spark_dir, data_root)


def run_self_check(output_dir):
    """Run all streaming jobs on fixture data and validate against expected outputs."""
    fixture_dir = _SELF_CHECK_FIXTURE_DIR
    self_check_dir = output_dir / "self_check"
    self_check_dir.mkdir(parents=True, exist_ok=True)

    results = {"self_check": True}

    # Class counts
    cc_result = stream.run_class_counts(
        fixture_dir / "fixture_manifest.csv",
        self_check_dir / "class_counts_check.csv",
    )
    results["class_counts"] = cc_result

    # Scenario counts
    sc_result = stream.run_scenario_counts(
        fixture_dir / "fixture_scenarios.csv",
        self_check_dir / "scenario_counts_check.csv",
    )
    results["scenario_counts"] = sc_result

    # Metric summary
    ms_result = stream.run_metric_summary(
        fixture_dir / "fixture_metrics.json",
        self_check_dir / "metric_summary_check.csv",
    )
    results["metric_summary"] = ms_result

    # Validate against expected outputs
    validation_passed = True
    for name in ["class_counts", "scenario_counts", "metric_summary"]:
        actual_path = self_check_dir / f"{name}_check.csv"
        expected_path = fixture_dir / f"expected_{name}.csv"

        if not actual_path.exists():
            results[f"{name}_validation"] = "FAIL: output file not created"
            validation_passed = False
            continue

        if not expected_path.exists():
            results[f"{name}_validation"] = "SKIP: expected file not found"
            continue

        with actual_path.open() as af, expected_path.open() as ef:
            actual_lines = af.read().strip().split("\n")
            expected_lines = ef.read().strip().split("\n")

        if len(actual_lines) != len(expected_lines):
            results[f"{name}_validation"] = (
                f"FAIL: line count differs ({len(actual_lines)} vs {len(expected_lines)})"
            )
            validation_passed = False
            continue

        if name == "metric_summary":
            # Float-tolerant comparison for numeric fields
            match = _compare_metric_summary(actual_lines, expected_lines)
        else:
            match = (actual_lines == expected_lines)

        if match:
            results[f"{name}_validation"] = "PASS"
        else:
            results[f"{name}_validation"] = "FAIL: output differs from expected"
            validation_passed = False
            logger.warning(
                "Self-check mismatch for %s:\nExpected:\n%s\nActual:\n%s",
                name,
                "\n".join(expected_lines[:10]),
                "\n".join(actual_lines[:10]),
            )

    results["validation_passed"] = validation_passed
    results["engine"] = stream.get_engine_name()
    return results


def _compare_metric_summary(actual_lines, expected_lines):
    """Compare metric summary lines with float tolerance."""
    if actual_lines[0] != expected_lines[0]:  # headers must match exactly
        return False
    for actual, expected in zip(actual_lines[1:], expected_lines[1:], strict=True):
        ap = actual.split(",")
        ep = expected.split(",")
        if len(ap) != len(ep):
            return False
        # First 3 fields are strings (method, condition, metric)
        if ap[:3] != ep[:3]:
            return False
        # Field 3 is n (integer)
        if ap[3] != ep[3]:
            return False
        # Field 4 is mean (float)
        try:
            if abs(float(ap[4]) - float(ep[4])) > 1e-6:
                return False
        except ValueError:
            return False
        # Field 5 is sample_std (float or empty)
        if ep[5] == "" and ap[5] == "":
            continue
        if ep[5] == "" or ap[5] == "":
            return False
        try:
            if abs(float(ap[5]) - float(ep[5])) > 1e-6:
                return False
        except ValueError:
            return False
    return True


def _find_manifest(cfg):
    """Locate the manifest CSV for the first seed."""
    seeds = cfg.reproducibility.seeds
    for seed in seeds:
        path = cfg.paths.results_root / "runs" / f"seed_{seed}" / "manifest" / "manifest.csv"
        if path.exists():
            return path
    # Default path even if it doesn't exist
    return cfg.paths.results_root / "runs" / f"seed_{seeds[0]}" / "manifest" / "manifest.csv"


def _find_scenarios(cfg):
    """Locate the scenarios CSV (from Phase 3 pairing)."""
    seeds = cfg.reproducibility.seeds
    for seed in seeds:
        path = cfg.paths.results_root / "runs" / f"seed_{seed}" / "pairing" / "scenarios.csv"
        if path.exists():
            return path
    return cfg.paths.results_root / "runs" / f"seed_{seeds[0]}" / "pairing" / "scenarios.csv"


def _find_metrics(cfg):
    """Locate the metrics JSON."""
    seeds = cfg.reproducibility.seeds
    for seed in seeds:
        path = cfg.paths.results_root / "runs" / f"seed_{seed}" / "metrics" / "metrics.json"
        if path.exists():
            return path
    return cfg.paths.results_root / "runs" / f"seed_{seeds[0]}" / "metrics" / "metrics.json"


def main():
    parser = argparse.ArgumentParser(
        description="GuardianEye BDA — Big Data Analytics module"
    )
    parser.add_argument(
        "--mode",
        choices=["class_counts", "scenarios", "metrics", "spark", "all"],
        default="all",
        help="Which BDA job(s) to run (default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Override results root directory",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Explicit path to the manifest CSV file",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run self-check on fixture data and validate outputs",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(data_root=args.data_root, results_root=args.results_root)
    output_dir = cfg.paths.results_root / "bda"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Self-check mode
    if args.self_check:
        status = run_self_check(output_dir)
        _write_status(output_dir, status)
        if status.get("validation_passed"):
            print("Self-check PASSED")
        else:
            print("Self-check FAILED — see bda_status.json for details")
            sys.exit(1)
        return

    # Resolve input paths
    manifest_path = Path(args.manifest_path) if args.manifest_path else _find_manifest(cfg)

    status = {"engine": stream.get_engine_name()}

    if args.mode in ("class_counts", "all"):
        status["class_counts"] = run_class_counts(manifest_path, output_dir)

    if args.mode in ("scenarios", "all"):
        scenarios_path = _find_scenarios(cfg)
        status["scenario_counts"] = run_scenario_counts(scenarios_path, output_dir)

    if args.mode in ("metrics", "all"):
        metrics_path = _find_metrics(cfg)
        status["metric_summary"] = run_metric_summary(metrics_path, output_dir)

    if args.mode in ("spark", "all"):
        status["spark"] = run_spark(
            manifest_path, output_dir, data_root=cfg.paths.data_root
        )

    # Overall status
    sub_statuses = [
        v.get("status") for v in status.values() if isinstance(v, dict)
    ]
    if all(s == "OK" for s in sub_statuses):
        status["status"] = "OK"
    elif any(s == "OK" for s in sub_statuses):
        status["status"] = "PARTIAL"
    else:
        status["status"] = "NOT YET RUN"
        status["reason"] = "No input files found. Run Phases 1-2 first."

    _write_status(output_dir, status)
    print(f"BDA complete — status: {status['status']}, engine: {status['engine']}")


if __name__ == "__main__":
    main()
