-- =============================================================================
-- GuardianEye — Hive DDL schema for BDA outputs
-- =============================================================================
-- This file defines EXTERNAL TABLES that point at the CSV/Parquet outputs
-- produced by the BDA streaming and Spark pipelines.
--
-- IMPORTANT: This file is NEVER EXECUTED by the GuardianEye pipeline.
-- It serves as documentation for how a Hive environment would consume
-- the BDA outputs.  Adjust LOCATION paths to match your HDFS layout.
-- =============================================================================

CREATE DATABASE IF NOT EXISTS guardian_eye
  COMMENT 'GuardianEye anti-poaching analytics';

USE guardian_eye;

-- ---------------------------------------------------------------------------
-- Manifest table (from Phase 1 manifest CSV)
-- ---------------------------------------------------------------------------
CREATE EXTERNAL TABLE IF NOT EXISTS manifest (
    path            STRING      COMMENT 'POSIX relative path from data root',
    modality        STRING      COMMENT 'image or audio',
    source          STRING      COMMENT 'Dataset slug: birdsai, cct, serengeti, belize, vietnam, esc50',
    label           STRING      COMMENT 'Canonical class label',
    view            STRING      COMMENT 'aerial_thermal, camera_trap, or acoustic',
    site_or_device  STRING      COMMENT 'Site or device identifier',
    noise_tag       STRING      COMMENT 'thunder_wind or not_thunder_wind',
    original_split  STRING      COMMENT 'Original dataset split: train, val, test, or unspecified',
    group_id        STRING      COMMENT 'Group identifier for leakage-free splitting',
    split           STRING      COMMENT 'Assigned experimental split: train, val, or test',
    notes           STRING      COMMENT 'Additional notes'
)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/data/guardian_eye/manifest/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- ---------------------------------------------------------------------------
-- Class counts (from BDA class-count streaming job)
-- ---------------------------------------------------------------------------
CREATE EXTERNAL TABLE IF NOT EXISTS class_counts (
    modality    STRING      COMMENT 'image or audio',
    source      STRING      COMMENT 'Dataset slug',
    label       STRING      COMMENT 'Class label',
    count       BIGINT      COMMENT 'Number of files'
)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/data/guardian_eye/bda/class_counts/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- ---------------------------------------------------------------------------
-- Scenario counts (from BDA scenario streaming job)
-- ---------------------------------------------------------------------------
CREATE EXTERNAL TABLE IF NOT EXISTS scenario_counts (
    split       STRING      COMMENT 'Experimental split: train, val, or test',
    scenario    STRING      COMMENT 'Scenario name (e.g. human_gunshot)',
    suspicious  INT         COMMENT '1 if suspicious, 0 otherwise',
    risk        STRING      COMMENT 'Risk level: HIGH, MEDIUM, or LOW',
    count       BIGINT      COMMENT 'Number of pairs'
)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/data/guardian_eye/bda/scenario_counts/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- ---------------------------------------------------------------------------
-- Metric summary (from BDA metric streaming job)
-- ---------------------------------------------------------------------------
CREATE EXTERNAL TABLE IF NOT EXISTS metric_summary (
    method      STRING      COMMENT 'Evaluation method (e.g. late_fusion, image_only)',
    condition   STRING      COMMENT 'Evaluation condition (e.g. clean, audio_dropped)',
    metric      STRING      COMMENT 'Metric name (e.g. f1, accuracy, fpr)',
    n           INT         COMMENT 'Number of seed runs contributing to the statistic',
    mean        DOUBLE      COMMENT 'Arithmetic mean across seeds',
    sample_std  DOUBLE      COMMENT 'Sample standard deviation (n-1 divisor); NULL if n < 2'
)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/data/guardian_eye/bda/metric_summary/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- ---------------------------------------------------------------------------
-- Manifest Parquet table (from Spark ingest job)
-- ---------------------------------------------------------------------------
CREATE EXTERNAL TABLE IF NOT EXISTS manifest_parquet (
    path            STRING,
    modality        STRING,
    source          STRING,
    label           STRING,
    view            STRING,
    site_or_device  STRING,
    noise_tag       STRING,
    original_split  STRING,
    group_id        STRING,
    split           STRING,
    notes           STRING
)
STORED AS PARQUET
LOCATION '/data/guardian_eye/bda/manifest_parquet/';

-- ---------------------------------------------------------------------------
-- Class weights Parquet table (from Spark class-weight job)
-- ---------------------------------------------------------------------------
CREATE EXTERNAL TABLE IF NOT EXISTS class_weights (
    modality    STRING,
    label       STRING,
    count       BIGINT      COMMENT 'n_c: count of class c',
    N           BIGINT      COMMENT 'Total samples in modality',
    k           INT         COMMENT 'Number of distinct classes',
    weight      DOUBLE      COMMENT 'w_c = N / (k * n_c)'
)
STORED AS PARQUET
LOCATION '/data/guardian_eye/bda/class_weights/';

-- ---------------------------------------------------------------------------
-- Audio features Parquet table (from Spark audio log-mel UDF job)
-- ---------------------------------------------------------------------------
CREATE EXTERNAL TABLE IF NOT EXISTS audio_features (
    path            STRING      COMMENT 'Audio file relative path',
    source          STRING      COMMENT 'Audio dataset source',
    label           STRING      COMMENT 'Class label: gunshot or background',
    split           STRING      COMMENT 'Experimental split: train, val, or test',
    logmel_features ARRAY<FLOAT> COMMENT '64-band log-mel energy features'
)
STORED AS PARQUET
LOCATION '/data/guardian_eye/bda/audio_features/';
