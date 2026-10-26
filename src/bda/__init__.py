"""
Big Data Analytics (BDA) module — Phase 3.

Provides Hadoop-streaming-style MapReduce jobs, an optional Spark path,
and a local Python streaming fallback for counting and summarizing
manifest data, scenario counts, and evaluation metrics.

Public API
----------
Mappers:
    ClassCountMapper  — emits (modality, source, label) → 1
    ScenarioMapper    — emits (split, scenario, suspicious, risk) → 1
    MetricMapper      — emits (method, condition, metric) → value

Reducers:
    SumReducer        — sums integer values per key
    MetricReducer     — computes n, mean, sample_std per key

Engine:
    stream.run_class_counts      — run class-count pipeline
    stream.run_scenario_counts   — run scenario-count pipeline
    stream.run_metric_summary    — run metric-summary pipeline
    stream.get_engine_name       — report Hadoop vs local engine

Spark (optional):
    spark_jobs.get_spark_status        — check PySpark availability
    spark_jobs.manifest_to_parquet     — ingest manifest to Parquet
    spark_jobs.compute_class_weights   — compute w_c = N / (k * n_c)
    spark_jobs.compute_audio_features  — audio log-mel feature manifest
"""

from src.bda.mapper_class import ClassCountMapper
from src.bda.mapper_metric import MetricMapper
from src.bda.mapper_scenario import ScenarioMapper
from src.bda.reducer_metric import MetricReducer
from src.bda.reducer_sum import SumReducer

__all__ = [
    "ClassCountMapper",
    "MetricMapper",
    "ScenarioMapper",
    "MetricReducer",
    "SumReducer",
]
