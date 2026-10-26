"""
Optional Spark jobs for GuardianEye BDA module.

All PySpark imports are wrapped in try/except.  If ``pyspark`` is not
installed, every public function returns a status dict with
``{"status": "NOT YET RUN", "reason": "pyspark not installed"}``.

When PySpark is available, three jobs are provided:

1. **manifest_to_parquet** — read manifest CSV, write Parquet.
2. **compute_class_weights** — compute ``w_c = N / (k * n_c)`` per modality.
3. **compute_audio_features** — pandas UDF producing log-mel spectrogram features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NOT_YET_RUN: dict[str, str] = {
    "status": "NOT YET RUN",
    "reason": "pyspark not installed. Install with: pip install 'guardian-eye[spark]'",
}

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F  # noqa: N812

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


def _get_spark_session(app_name: str = "GuardianEye-BDA") -> Any:
    """
    Create or get a local Spark session.

    Parameters
    ----------
    app_name:
        Application name for the Spark session.

    Returns
    -------
    SparkSession
        A local Spark session.

    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def manifest_to_parquet(
    manifest_csv: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Read manifest CSV and write it as Parquet.

    Parameters
    ----------
    manifest_csv:
        Path to the manifest CSV file.
    output_path:
        Path to write the Parquet file.

    Returns
    -------
    dict[str, Any]
        Status dict with row count and output path.

    """
    if not PYSPARK_AVAILABLE:
        return dict(_NOT_YET_RUN)

    if not manifest_csv.exists():
        return {
            "status": "NOT YET RUN",
            "reason": f"Manifest CSV not found: {manifest_csv}",
        }

    spark = _get_spark_session()
    try:
        df = spark.read.csv(str(manifest_csv), header=True, inferSchema=True)
        row_count = df.count()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write.mode("overwrite").parquet(str(output_path))
        logger.info("Manifest written to Parquet: %s (%d rows)", output_path, row_count)
        return {
            "status": "OK",
            "output": str(output_path),
            "row_count": row_count,
        }
    finally:
        spark.stop()


def compute_class_weights(
    manifest_csv: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Compute class weights ``w_c = N / (k * n_c)`` per modality.

    Parameters
    ----------
    manifest_csv:
        Path to the manifest CSV file.
    output_path:
        Path to write the class-weights Parquet file.

    Returns
    -------
    dict[str, Any]
        Status dict with the computed weights.

    """
    if not PYSPARK_AVAILABLE:
        return dict(_NOT_YET_RUN)

    if not manifest_csv.exists():
        return {
            "status": "NOT YET RUN",
            "reason": f"Manifest CSV not found: {manifest_csv}",
        }

    spark = _get_spark_session()
    try:
        df = spark.read.csv(str(manifest_csv), header=True, inferSchema=True)
        total_rows = df.count()
        if total_rows == 0:
            return {
                "status": "NOT YET RUN",
                "reason": f"Manifest CSV contains 0 data rows: {manifest_csv}",
            }

        # Count per (modality, label)
        counts = df.groupBy("modality", "label").count()

        # Total N per modality and number of distinct classes k per modality
        totals = df.groupBy("modality").agg(
            F.count("*").alias("N"),
            F.countDistinct("label").alias("k"),
        )

        # Join and compute w_c = N / (k * n_c), guarding against zero division
        weights_df = (
            counts
            .join(totals, on="modality")
            .withColumn(
                "weight",
                F.when(
                    (F.col("k") > 0) & (F.col("count") > 0),
                    F.col("N") / (F.col("k") * F.col("count")),
                ).otherwise(0.0),
            )
            .select("modality", "label", "count", "N", "k", "weight")
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        weights_df.write.mode("overwrite").parquet(str(output_path))

        # Collect for the status dict
        rows = weights_df.collect()
        weight_records = [
            {
                "modality": r["modality"],
                "label": r["label"],
                "n_c": r["count"],
                "N": r["N"],
                "k": r["k"],
                "weight": round(float(r["weight"]), 6),
            }
            for r in rows
        ]

        logger.info("Class weights computed: %d entries", len(weight_records))
        return {
            "status": "OK",
            "output": str(output_path),
            "weights": weight_records,
        }
    finally:
        spark.stop()


def compute_audio_features(
    manifest_csv: Path,
    output_path: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    Compute audio log-mel features using a pandas UDF.

    Parameters
    ----------
    manifest_csv:
        Path to the manifest CSV file.
    output_path:
        Path to write the features Parquet file.
    data_root:
        Root directory for resolving audio file paths.

    Returns
    -------
    dict[str, Any]
        Status dict.

    """
    if not PYSPARK_AVAILABLE:
        return dict(_NOT_YET_RUN)

    if not manifest_csv.exists():
        return {
            "status": "NOT YET RUN",
            "reason": f"Manifest CSV not found: {manifest_csv}",
        }

    spark = _get_spark_session()
    try:
        import pandas as pd
        from pyspark.sql.functions import pandas_udf

        df = spark.read.csv(str(manifest_csv), header=True, inferSchema=True)
        audio_df = df.filter(F.col("modality") == "audio")

        row_count = audio_df.count()
        if row_count == 0:
            return {
                "status": "NOT YET RUN",
                "reason": "No audio rows in manifest.",
            }

        resolved_root_str = str(data_root.resolve()) if data_root else ""

        @pandas_udf("array<float>")
        def extract_logmel_bands_udf(paths: pd.Series) -> pd.Series:
            """Compute 64-band log-mel energy means via pandas UDF."""
            try:
                import torchaudio
                has_torchaudio = True
            except ImportError:
                has_torchaudio = False

            results = []
            for rel_path in paths:
                if not has_torchaudio or not resolved_root_str:
                    results.append([0.0] * 64)
                    continue
                full_path = Path(resolved_root_str) / str(rel_path)
                if not full_path.exists():
                    results.append([0.0] * 64)
                    continue
                try:
                    waveform, sr = torchaudio.load(str(full_path))
                    if sr != 8000:
                        waveform = torchaudio.functional.resample(waveform, sr, 8000)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    mel_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=8000, n_fft=512, hop_length=128, n_mels=64
                    )(waveform)
                    log_mel = (mel_spec + 1e-6).log()
                    band_means = log_mel.mean(dim=2).squeeze(0).tolist()
                    results.append([float(x) for x in band_means])
                except Exception:
                    results.append([0.0] * 64)
            return pd.Series(results)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_features_df = (
            audio_df
            .withColumn("logmel_features", extract_logmel_bands_udf(F.col("path")))
            .select("path", "source", "label", "split", "logmel_features")
        )
        audio_features_df.write.mode("overwrite").parquet(str(output_path))

        logger.info("Audio features manifest written: %s (%d rows)", output_path, row_count)
        return {
            "status": "OK",
            "output": str(output_path),
            "row_count": row_count,
            "feature_dim": 64,
        }
    finally:
        spark.stop()


def get_spark_status() -> dict[str, Any]:
    """
    Return the availability status of PySpark.

    Returns
    -------
    dict[str, Any]
        Status dict indicating whether PySpark is available.

    """
    if PYSPARK_AVAILABLE:
        return {"status": "OK", "pyspark_available": True}
    return dict(_NOT_YET_RUN)


def run_all_spark_jobs(
    manifest_csv: Path,
    output_dir: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    Run all Spark jobs and return a combined status.

    Parameters
    ----------
    manifest_csv:
        Path to the manifest CSV file.
    output_dir:
        Directory to write Spark outputs.
    data_root:
        Root directory for resolving file paths.

    Returns
    -------
    dict[str, Any]
        Combined status for all Spark jobs.

    """
    results: dict[str, Any] = {"spark_available": PYSPARK_AVAILABLE}

    if not PYSPARK_AVAILABLE:
        results["status"] = "NOT YET RUN"
        results["reason"] = _NOT_YET_RUN["reason"]
        return results

    output_dir.mkdir(parents=True, exist_ok=True)

    results["manifest_parquet"] = manifest_to_parquet(
        manifest_csv, output_dir / "manifest.parquet"
    )
    results["class_weights"] = compute_class_weights(
        manifest_csv, output_dir / "class_weights.parquet"
    )
    results["audio_features"] = compute_audio_features(
        manifest_csv, output_dir / "audio_features.parquet", data_root
    )

    # Overall status
    statuses = [
        results["manifest_parquet"].get("status"),
        results["class_weights"].get("status"),
        results["audio_features"].get("status"),
    ]
    if all(s == "OK" for s in statuses):
        results["status"] = "OK"
    else:
        results["status"] = "PARTIAL"
        not_run = [
            k for k, v in results.items()
            if isinstance(v, dict) and v.get("status") != "OK"
        ]
        results["not_run_jobs"] = not_run

    return results
