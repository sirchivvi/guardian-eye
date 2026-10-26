"""
Streaming engine for Hadoop-streaming-style BDA jobs.

If ``HADOOP_HOME`` is set and the ``hadoop`` binary is on PATH, runs
mapper/reducer through the Hadoop Streaming jar.  Otherwise sorts mapper
output locally and pipes through the same reducer.

The engine name is logged for every run:
- ``"hadoop_streaming"`` when using an actual Hadoop installation.
- ``"python_streaming_local"`` when running the pure-Python fallback.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from src.bda.mapper_class import ClassCountMapper
from src.bda.mapper_metric import MetricMapper
from src.bda.mapper_scenario import ScenarioMapper
from src.bda.reducer_metric import MetricReducer
from src.bda.reducer_sum import SumReducer

logger = logging.getLogger(__name__)

# Type aliases
MapperType = ClassCountMapper | ScenarioMapper | MetricMapper
ReducerType = SumReducer | MetricReducer


def _detect_hadoop() -> tuple[bool, str | None]:
    """
    Detect whether Hadoop is available.

    Returns
    -------
    tuple[bool, str | None]
        ``(is_available, streaming_jar_path)``.

    """
    hadoop_home = os.environ.get("HADOOP_HOME")
    if not hadoop_home:
        return False, None

    hadoop_bin = shutil.which("hadoop")
    if not hadoop_bin:
        logger.info("HADOOP_HOME set but 'hadoop' not on PATH; using local streaming")
        return False, None

    # Try to locate the streaming jar
    tools_lib = Path(hadoop_home) / "share" / "hadoop" / "tools" / "lib"
    jar_candidates = sorted(tools_lib.glob("hadoop-streaming-*.jar"))
    if not jar_candidates:
        logger.info("Hadoop Streaming jar not found in %s; using local streaming", tools_lib)
        return False, None

    jar_path = str(jar_candidates[-1])  # latest version
    logger.info("Detected Hadoop Streaming jar: %s", jar_path)
    return True, jar_path


def _run_hadoop_streaming(
    jar_path: str,
    mapper_module: str,
    reducer_module: str,
    input_path: str,
    output_path: str,
) -> None:
    """
    Run a MapReduce job through Hadoop Streaming.

    Parameters
    ----------
    jar_path:
        Path to the Hadoop Streaming JAR file.
    mapper_module:
        Python module path for the mapper (e.g. ``src.bda.mapper_class``).
    reducer_module:
        Python module path for the reducer (e.g. ``src.bda.reducer_sum``).
    input_path:
        HDFS or local input path.
    output_path:
        HDFS or local output path.

    """
    cmd = [
        "hadoop", "jar", jar_path,
        "-mapper", f"python -m {mapper_module}",
        "-reducer", f"python -m {reducer_module}",
        "-input", input_path,
        "-output", output_path,
    ]
    logger.info("Running Hadoop Streaming: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603


def _run_local_streaming_csv(
    mapper: ClassCountMapper | ScenarioMapper,
    reducer: SumReducer,
    input_path: Path,
    output_path: Path,
    header_fields: list[str],
) -> None:
    """
    Run a CSV-based streaming job locally.

    Parameters
    ----------
    mapper:
        An initialized mapper instance.
    reducer:
        An initialized reducer instance.
    input_path:
        Path to the input CSV file.
    output_path:
        Path to write the output CSV file.
    header_fields:
        Column names for the output CSV header (key components + "count").

    """
    # Map phase
    kv_pairs: list[tuple[str, str]] = []

    with input_path.open() as fh:
        header_line = fh.readline()
        mapper.set_header(header_line)
        for line in fh:
            for key, value in mapper.map(line):
                kv_pairs.append((key, value))

    logger.info(
        "Mapper processed %d rows, skipped %d",
        mapper.rows_processed,
        mapper.rows_skipped,
    )

    # Sort phase (simulates Hadoop shuffle/sort)
    kv_pairs.sort(key=lambda x: x[0])

    # Reduce phase
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        # Write header
        fh.write(",".join(header_fields) + "\n")

        if not kv_pairs:
            return

        current_key = kv_pairs[0][0]
        current_values: list[str] = []

        for key, value in kv_pairs:
            if key != current_key:
                for output_line in reducer.reduce(current_key, iter(current_values)):
                    parts = output_line.split("\t")
                    fh.write(",".join(parts) + "\n")
                current_values = []
                current_key = key
            current_values.append(value)

        # Flush last group
        if current_values:
            for output_line in reducer.reduce(current_key, iter(current_values)):
                parts = output_line.split("\t")
                fh.write(",".join(parts) + "\n")


def _run_local_streaming_metrics(
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Run metric streaming job locally from a JSON array file.

    Parameters
    ----------
    input_path:
        Path to the JSON metrics file (array of metric objects).
    output_path:
        Path to write the output CSV file.

    """
    mapper = MetricMapper()
    reducer = MetricReducer()

    # Map phase: read JSON array and emit key-value pairs
    with input_path.open() as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        records = [records]

    kv_pairs: list[tuple[str, str]] = []
    for record in records:
        line = json.dumps(record)
        for key, value in mapper.map(line):
            kv_pairs.append((key, value))

    # Sort phase
    kv_pairs.sort(key=lambda x: x[0])

    # Reduce phase
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        fh.write("method,condition,metric,n,mean,sample_std\n")

        if not kv_pairs:
            return

        current_key = kv_pairs[0][0]
        current_values: list[str] = []

        for key, value in kv_pairs:
            if key != current_key:
                for output_line in reducer.reduce(current_key, iter(current_values)):
                    parts = output_line.split("\t")
                    fh.write(",".join(parts) + "\n")
                current_values = []
                current_key = key
            current_values.append(value)

        if current_values:
            for output_line in reducer.reduce(current_key, iter(current_values)):
                parts = output_line.split("\t")
                fh.write(",".join(parts) + "\n")


def get_engine_name() -> str:
    """
    Return the name of the streaming engine that will be used.

    Returns
    -------
    str
        ``"hadoop_streaming"`` or ``"python_streaming_local"``.

    """
    has_hadoop, _ = _detect_hadoop()
    return "hadoop_streaming" if has_hadoop else "python_streaming_local"


def run_class_counts(
    manifest_path: Path,
    output_path: Path,
) -> dict[str, str | int]:
    """
    Run the class-count streaming job.

    Parameters
    ----------
    manifest_path:
        Path to the manifest CSV file.
    output_path:
        Path to write the output CSV file.

    Returns
    -------
    dict[str, str | int]
        Status information including engine name and row count.

    """
    engine = get_engine_name()
    logger.info("Running class-counts job with engine: %s", engine)

    if not manifest_path.exists():
        logger.warning("Manifest not found: %s", manifest_path)
        return {
            "status": "NOT YET RUN",
            "reason": f"Manifest file not found: {manifest_path}",
            "engine": engine,
        }

    if engine == "hadoop_streaming":
        _, jar_path = _detect_hadoop()
        assert jar_path is not None
        _run_hadoop_streaming(
            jar_path,
            "src.bda.mapper_class",
            "src.bda.reducer_sum",
            str(manifest_path),
            str(output_path),
        )
    else:
        mapper = ClassCountMapper()
        reducer = SumReducer()
        _run_local_streaming_csv(
            mapper,
            reducer,
            manifest_path,
            output_path,
            header_fields=["modality", "source", "label", "count"],
        )

    return {"status": "OK", "engine": engine, "output": str(output_path)}


def run_scenario_counts(
    scenarios_path: Path,
    output_path: Path,
) -> dict[str, str | int]:
    """
    Run the scenario-count streaming job.

    Parameters
    ----------
    scenarios_path:
        Path to the scenario CSV file.
    output_path:
        Path to write the output CSV file.

    Returns
    -------
    dict[str, str | int]
        Status information including engine name.

    """
    engine = get_engine_name()
    logger.info("Running scenario-counts job with engine: %s", engine)

    if not scenarios_path.exists():
        logger.warning("Scenarios file not found: %s", scenarios_path)
        return {
            "status": "NOT YET RUN",
            "reason": f"Scenarios file not found: {scenarios_path}",
            "engine": engine,
        }

    if engine == "hadoop_streaming":
        _, jar_path = _detect_hadoop()
        assert jar_path is not None
        _run_hadoop_streaming(
            jar_path,
            "src.bda.mapper_scenario",
            "src.bda.reducer_sum",
            str(scenarios_path),
            str(output_path),
        )
    else:
        mapper = ScenarioMapper()
        reducer = SumReducer()
        _run_local_streaming_csv(
            mapper,
            reducer,
            scenarios_path,
            output_path,
            header_fields=["split", "scenario", "suspicious", "risk", "count"],
        )

    return {"status": "OK", "engine": engine, "output": str(output_path)}


def run_metric_summary(
    metrics_path: Path,
    output_path: Path,
) -> dict[str, str | int]:
    """
    Run the metric summary streaming job.

    Parameters
    ----------
    metrics_path:
        Path to the metrics JSON file.
    output_path:
        Path to write the output CSV file.

    Returns
    -------
    dict[str, str | int]
        Status information including engine name.

    """
    engine = get_engine_name()
    logger.info("Running metric-summary job with engine: %s", engine)

    if not metrics_path.exists():
        logger.warning("Metrics file not found: %s", metrics_path)
        return {
            "status": "NOT YET RUN",
            "reason": f"Metrics file not found: {metrics_path}",
            "engine": engine,
        }

    if engine == "hadoop_streaming":
        _, jar_path = _detect_hadoop()
        assert jar_path is not None
        _run_hadoop_streaming(
            jar_path,
            "src.bda.mapper_metric",
            "src.bda.reducer_metric",
            str(metrics_path),
            str(output_path),
        )
    else:
        _run_local_streaming_metrics(metrics_path, output_path)

    return {"status": "OK", "engine": engine, "output": str(output_path)}
