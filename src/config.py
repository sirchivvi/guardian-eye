"""
GuardianEye configuration loader.

Reads ``config.yaml``, validates all values with Pydantic v2, and exposes a
``load_config()`` function that respects the following precedence order
(highest first):

1. Explicit CLI arguments passed to ``resolve_paths()``
2. ``DATA_ROOT`` / ``RESULTS_ROOT`` environment variables
3. Values in ``config.yaml``

Usage
-----
>>> from src.config import load_config
>>> cfg = load_config()
>>> cfg.paths.data_root
PosixPath('data')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """Top-level project metadata."""

    name: str = "guardian-eye"
    version: str = "0.1.0"


class PathsConfig(BaseModel):
    """
    Filesystem path configuration.

    ``data_root`` and ``results_root`` are overridable via the ``DATA_ROOT``
    and ``RESULTS_ROOT`` environment variables.
    """

    data_root: Path = Path("data")
    results_root: Path = Path("results")
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"

    @property
    def raw(self) -> Path:
        """Return the full path to the raw data directory."""
        return self.data_root / self.raw_subdir

    @property
    def processed(self) -> Path:
        """Return the full path to the processed data directory."""
        return self.data_root / self.processed_subdir


class ReproducibilityConfig(BaseModel):
    """Random seeds for experiment replication."""

    seeds: list[int] = [27, 2701, 26009]


class SplitsConfig(BaseModel):
    """Dataset split fractions; must sum to 1.0."""

    train: float = Field(0.70, gt=0, lt=1)
    val: float = Field(0.15, gt=0, lt=1)
    test: float = Field(0.15, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_sum(self) -> SplitsConfig:
        """Validate that split fractions sum to 1.0."""
        total = round(self.train + self.val + self.test, 6)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"splits must sum to 1.0, got {total}"
            )
        return self


class DatasetsConfig(BaseModel):
    """Names of dataset sub-directories within data/raw/."""

    image: list[str] = ["birdsai", "cct", "serengeti", "belize", "vietnam"]
    audio: list[str] = ["esc50"]


class ImageConfig(BaseModel):
    """Image-branch hyperparameters."""

    input_size: tuple[int, int] = (224, 224)
    backbone: str = "cnn4"
    pretrained: bool = False
    embedding_dim: int = Field(128, ge=1)
    num_classes: int = Field(4, ge=1)
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(30, ge=1)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    augment: bool = True


class AudioConfig(BaseModel):
    """Audio-branch hyperparameters."""

    sample_rate: int = Field(8000, gt=0)
    n_mels: int = Field(64, ge=1)
    hop_length: int = Field(128, ge=1)
    n_fft: int = Field(512, ge=1)
    duration_sec: float = Field(4.0, gt=0)
    backbone: str = "cnn4"
    pretrained: bool = False
    embedding_dim: int = Field(128, ge=1)
    num_classes: int = Field(2, ge=1)
    batch_size: int = Field(64, ge=1)
    epochs: int = Field(30, ge=1)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    augment: bool = True


class PairsConfig(BaseModel):
    """Image–audio synthetic pairing limits."""

    max_pairs_per_split: int = Field(10_000, ge=1)
    min_images_per_class: int = Field(50, ge=1)
    cross_split_allowed: bool = False


class FusionConfig(BaseModel):
    """Late-fusion settings."""

    method: str = "late_fusion"
    image_weight: float = Field(0.6, ge=0, le=1)
    audio_weight: float = Field(0.4, ge=0, le=1)

    @field_validator("method")
    @classmethod
    def _check_method(cls, v: str) -> str:
        """Validate fusion method name."""
        allowed = {"late_fusion", "learned_weighted"}
        if v not in allowed:
            raise ValueError(f"fusion.method must be one of {allowed}, got {v!r}")
        return v


class RobustnessConfig(BaseModel):
    """Robustness evaluation parameters."""

    noise_types: list[str] = ["gaussian", "salt_pepper", "occlusion"]
    noise_levels: list[float] = [0.05, 0.10, 0.20]


class EvaluationConfig(BaseModel):
    """Evaluation metrics and bootstrap settings."""

    metrics: list[str] = ["accuracy", "f1_macro", "roc_auc", "ap"]
    bootstrap_resamples: int = Field(1000, ge=1)
    confidence_level: float = Field(0.95, gt=0, lt=1)


class LatencyConfig(BaseModel):
    """Per-inference latency targets in milliseconds."""

    image_ms: float = Field(100.0, gt=0)
    audio_ms: float = Field(100.0, gt=0)
    fusion_ms: float = Field(20.0, gt=0)
    total_ms: float = Field(250.0, gt=0)


class SmokeConfig(BaseModel):
    """Settings for CPU-only smoke tests."""

    batch_size: int = Field(2, ge=1)
    epochs: int = Field(1, ge=1)
    max_pairs: int = Field(10, ge=1)
    fixture_images: int = Field(8, ge=1)
    fixture_audio: int = Field(8, ge=1)


# ---------------------------------------------------------------------------
# Root configuration model
# ---------------------------------------------------------------------------


class GuardianEyeConfig(BaseModel):
    """
    Root configuration model for GuardianEye.

    Loaded from ``config.yaml`` and optionally overridden by environment
    variables ``DATA_ROOT`` and ``RESULTS_ROOT``.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig
    )
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    pairs: PairsConfig = Field(default_factory=PairsConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(
    config_path: Path | str | None = None,
    *,
    data_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> GuardianEyeConfig:
    """
    Load and validate the GuardianEye configuration.

    Precedence (highest first):
    1. ``data_root`` / ``results_root`` keyword arguments (CLI overrides)
    2. ``DATA_ROOT`` / ``RESULTS_ROOT`` environment variables
    3. Values in ``config.yaml``

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  Defaults to the
        ``config.yaml`` in the repository root.
    data_root:
        Override for ``paths.data_root`` (CLI level).
    results_root:
        Override for ``paths.results_root`` (CLI level).

    Returns
    -------
    GuardianEyeConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    pydantic.ValidationError
        If the configuration values fail validation.

    """
    cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Build the config from YAML
    cfg = GuardianEyeConfig.model_validate(raw)

    # Apply env-var overrides (level 2)
    env_data = os.environ.get("DATA_ROOT")
    env_results = os.environ.get("RESULTS_ROOT")
    if env_data:
        cfg.paths.data_root = Path(env_data)
    if env_results:
        cfg.paths.results_root = Path(env_results)

    # Apply CLI-level overrides (level 1, highest)
    cfg = resolve_paths(cfg, data_root=data_root, results_root=results_root)

    return cfg


def resolve_paths(
    cfg: GuardianEyeConfig,
    *,
    data_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> GuardianEyeConfig:
    """
    Apply explicit CLI path overrides to an existing config.

    This is the highest-precedence path override layer.  Call it after
    parsing CLI arguments to apply ``--data-root`` / ``--results-root`` flags.

    Parameters
    ----------
    cfg:
        Configuration object to update (mutated in place and returned).
    data_root:
        Explicit data root path.  ``None`` means "no override".
    results_root:
        Explicit results root path.  ``None`` means "no override".

    Returns
    -------
    GuardianEyeConfig
        The same object with paths updated.

    """
    if data_root is not None:
        cfg.paths.data_root = Path(data_root)
    if results_root is not None:
        cfg.paths.results_root = Path(results_root)
    return cfg
