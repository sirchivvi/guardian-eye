# Big Data Analytics (BDA) Module — GuardianEye

> **Design and implementation documentation for Phase 3 Big Data Analytics.**

The GuardianEye BDA module (`src/bda/`) provides scalable data-aggregation pipelines designed for large-scale wildlife monitoring and sensor networks. It implements Hadoop-Streaming-compatible MapReduce jobs, an automated local streaming engine for local and CI execution, optional PySpark pipelines, Hive DDL schemas, and verification self-checks.

---

## 1. Architecture Overview

In an anti-poaching sensor deployment, acoustic sensors and camera traps generate high-volume distributed logs and manifests. The BDA architecture aggregates raw metadata, scenario combinations, and evaluation metrics across seeds and folds without requiring manual spreadsheet manipulation or single-threaded loops.

```
                    ┌───────────────────────────────┐
                    │ Raw Manifest / Scenarios /    │
                    │ Evaluation Metrics            │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                      Hadoop Streaming / Local Sort
                    ┌───────────────────────────────┐
                    │ Mapper: emits key-value pairs │
                    │ Sort: groups keys (composite) │
                    │ Reducer: aggregation / stats  │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────┐
│ CSV Aggregations:     │                       │ Optional Spark /      │
│ - class_counts.csv    │                       │ Parquet Ingestion     │
│ - scenario_counts.csv │                       │ - manifest.parquet    │
│ - metric_summary.csv  │                       │ - class_weights.pq    │
│ - bda_status.json     │                       │ - audio_features.pq   │
└───────────────────────┘                       └───────────────────────┘
```

---

## 2. Hadoop Streaming MapReduce Jobs

Each mapper and reducer is executable as a standalone module via stdin/stdout, making it directly consumable by the standard `hadoop jar hadoop-streaming-*.jar`:

### 2.1 Class Counts Pipeline
- **Mapper** (`src/bda/mapper_class.py`):
  Reads manifest CSV lines (`path,modality,source,label,...`) and emits:
  ```
  modality\tsource\tlabel \t 1
  ```
  Missing or empty label fields are skipped (never converted into negative/background examples).
- **Reducer** (`src/bda/reducer_sum.py`):
  Groups pre-sorted keys and sums the integer counts, producing:
  ```
  modality\tsource\tlabel\t<count>
  ```

### 2.2 Scenario Counts Pipeline
- **Mapper** (`src/bda/mapper_scenario.py`):
  Reads pairing CSV lines (`split,scenario,suspicious,risk,...`) and emits:
  ```
  split\tscenario\tsuspicious\trisk \t 1
  ```
- **Reducer** (`src/bda/reducer_sum.py`):
  Aggregates counts per scenario tuple, producing:
  ```
  split\tscenario\tsuspicious\trisk\t<count>
  ```

### 2.3 Evaluation Metric Summary Pipeline
- **Mapper** (`src/bda/mapper_metric.py`):
  Reads JSON metric records (`{"method": ..., "condition": ..., "metric": ..., "value": ...}`) and emits:
  ```
  method\tcondition\tmetric \t <value>
  ```
  Undefined metrics (`value: null`, zero-denominator situations) are omitted.
- **Reducer** (`src/bda/reducer_metric.py`):
  Computes sample statistics for each key:
  - $n$: sample count
  - $\text{mean} = \frac{1}{n} \sum x_i$
  - $\text{sample\_std} = \sqrt{\frac{1}{n-1} \sum (x_i - \text{mean})^2}$ when $n \ge 2$; left empty when $n < 2$.

---

## 3. Streaming Engine & Execution Detection

`src/bda/stream.py` provides an execution engine that automatically chooses between Hadoop Streaming and a local Python fallback:

1. **Hadoop Detection**: Checks if `HADOOP_HOME` is set, `hadoop` binary exists on PATH, and `hadoop-streaming-*.jar` is located.
2. **Local Fallback (`python_streaming_local`)**: Simulates the MapReduce pipeline by executing the mapper, sorting key-value pairs in memory, and piping them into the reducer.
3. **Engine Logging**: The engine name (`hadoop_streaming` or `python_streaming_local`) is recorded in logs and in `bda_status.json`.

---

## 4. Optional PySpark Jobs

`src/bda/spark_jobs.py` provides DataFrame transformations for larger scale datasets:

1. **`manifest_to_parquet`**: Ingests manifest CSV into compressed Parquet.
2. **`compute_class_weights`**: Calculates balanced class weights per modality:
   $$w_c = \frac{N}{k \cdot n_c}$$
   Guarded against division by zero if $k=0$ or $n_c=0$.
3. **`compute_audio_features`**: Uses a `@pandas_udf` to compute 64-band log-mel energy summaries across worker nodes using `torchaudio`.

**Honesty Policy on PySpark**:
If `pyspark` is not installed, all Spark routines gracefully return `{"status": "NOT YET RUN", "reason": "pyspark not installed"}`. The system never fakes Spark completion.

---

## 5. Hive DDL Schemas

`src/bda/schema.hql` provides Hive external table definitions for warehouse analytics:
- `manifest`: external table pointing to raw manifest CSV.
- `class_counts`: table for class distribution summaries.
- `scenario_counts`: table for synthetic pair scenario distributions.
- `metric_summary`: table for multi-seed performance statistics.
- `manifest_parquet`: Parquet representation of the manifest.
- `class_weights`: Parquet table of computed inverse-frequency weights.
- `audio_features`: Parquet table with 64-band log-mel feature arrays.

*Note: `schema.hql` serves as schema documentation and is not executed in local pipelines.*

---

## 6. CLI and Makefile Integration

### Run BDA Self-Check (on built-in fixtures)
```bash
make bda-check
# or
python scripts/run_bda.py --self-check
```
Validates mapper and reducer pipelines against golden fixtures (`tests/fixtures/bda/`) with tolerance-based numeric comparison.

### Run BDA on Project Manifests
```bash
make bda
# or
python scripts/run_bda.py --mode all
```
Modes available:
- `--mode class_counts`: Run class counts job.
- `--mode scenarios`: Run scenario pairing distribution job.
- `--mode metrics`: Run evaluation metrics summary job.
- `--mode spark`: Run PySpark jobs (if installed).
- `--mode all`: Run all streaming and Spark jobs.

### Status Tracking
All executions write atomic status output to `results/bda/bda_status.json`. If inputs are unpopulated (e.g. before downloading real raw data), the pipeline exits cleanly with `NOT YET RUN`.
