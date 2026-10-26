"""Unit tests for src/config.py.

Covers:
- Config loads from config.yaml
- Default values match the spec
- DATA_ROOT / RESULTS_ROOT env var overrides
- resolve_paths() CLI override (highest precedence)
- Precedence chain: CLI > env > YAML
- Invalid split fractions raise ValidationError
- Invalid fusion method raises ValidationError
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import GuardianEyeConfig, load_config, resolve_paths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Write a minimal valid config.yaml to tmp_path and return its path."""
    base = {
        "reproducibility": {"seeds": [27, 2701, 26009]},
        "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
    }
    if overrides:
        base.update(overrides)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base))
    return cfg_file


# ---------------------------------------------------------------------------
# Basic load
# ---------------------------------------------------------------------------


def test_load_config_from_repo_root() -> None:
    """load_config() with no arguments reads the repo-level config.yaml."""
    cfg = load_config()
    assert isinstance(cfg, GuardianEyeConfig)


def test_seeds_match_spec() -> None:
    """Seeds must match the spec: [27, 2701, 26009]."""
    cfg = load_config()
    assert cfg.reproducibility.seeds == [27, 2701, 26009]


def test_splits_match_spec() -> None:
    """Train/val/test must be 0.70 / 0.15 / 0.15."""
    cfg = load_config()
    assert cfg.splits.train == pytest.approx(0.70)
    assert cfg.splits.val == pytest.approx(0.15)
    assert cfg.splits.test == pytest.approx(0.15)


def test_config_file_not_found(tmp_path: Path) -> None:
    """load_config() raises FileNotFoundError for a missing YAML file."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Env-var overrides (level 2)
# ---------------------------------------------------------------------------


def test_data_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DATA_ROOT env var overrides config.yaml path."""
    cfg_file = _write_minimal_config(tmp_path)
    custom_root = tmp_path / "my_data"
    monkeypatch.setenv("DATA_ROOT", str(custom_root))

    cfg = load_config(cfg_file)
    assert cfg.paths.data_root == custom_root


def test_results_root_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RESULTS_ROOT env var overrides config.yaml path."""
    cfg_file = _write_minimal_config(tmp_path)
    custom_root = tmp_path / "my_results"
    monkeypatch.setenv("RESULTS_ROOT", str(custom_root))

    cfg = load_config(cfg_file)
    assert cfg.paths.results_root == custom_root


def test_env_vars_not_set_uses_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When env vars are absent, the YAML path is used."""
    cfg_file = _write_minimal_config(
        tmp_path,
        {"paths": {"data_root": "data", "results_root": "results",
                   "raw_subdir": "raw", "processed_subdir": "processed"}},
    )
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("RESULTS_ROOT", raising=False)

    cfg = load_config(cfg_file)
    assert cfg.paths.data_root == Path("data")
    assert cfg.paths.results_root == Path("results")


# ---------------------------------------------------------------------------
# CLI overrides — resolve_paths() (level 1, highest precedence)
# ---------------------------------------------------------------------------


def test_cli_data_root_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI data_root beats env var DATA_ROOT."""
    cfg_file = _write_minimal_config(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "env_data"))

    cli_root = tmp_path / "cli_data"
    cfg = load_config(cfg_file, data_root=cli_root)
    assert cfg.paths.data_root == cli_root


def test_cli_results_root_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI results_root beats env var RESULTS_ROOT."""
    cfg_file = _write_minimal_config(tmp_path)
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "env_results"))

    cli_root = tmp_path / "cli_results"
    cfg = load_config(cfg_file, results_root=cli_root)
    assert cfg.paths.results_root == cli_root


def test_precedence_cli_gt_env_gt_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full precedence chain: CLI > env > YAML for data_root."""
    yaml_root = Path("data")
    env_root = tmp_path / "env_data"
    cli_root = tmp_path / "cli_data"

    cfg_file = _write_minimal_config(
        tmp_path,
        {"paths": {"data_root": str(yaml_root), "results_root": "results",
                   "raw_subdir": "raw", "processed_subdir": "processed"}},
    )

    # Step 1: only YAML — expect yaml value
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = load_config(cfg_file)
    assert cfg.paths.data_root == yaml_root

    # Step 2: env set — expect env value
    monkeypatch.setenv("DATA_ROOT", str(env_root))
    cfg = load_config(cfg_file)
    assert cfg.paths.data_root == env_root

    # Step 3: CLI also set — expect CLI value
    cfg = load_config(cfg_file, data_root=cli_root)
    assert cfg.paths.data_root == cli_root


def test_resolve_paths_no_override_leaves_config_unchanged(tmp_path: Path) -> None:
    """resolve_paths() with all-None args must not mutate paths."""
    cfg_file = _write_minimal_config(tmp_path)
    cfg = load_config(cfg_file)
    original_data = cfg.paths.data_root
    original_results = cfg.paths.results_root

    resolve_paths(cfg)  # no overrides

    assert cfg.paths.data_root == original_data
    assert cfg.paths.results_root == original_results


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_splits_not_summing_to_one_raises(tmp_path: Path) -> None:
    """Splits that don't sum to 1.0 must raise ValidationError."""
    cfg_file = _write_minimal_config(
        tmp_path, {"splits": {"train": 0.80, "val": 0.15, "test": 0.15}}
    )
    with pytest.raises(ValidationError, match="splits must sum"):
        load_config(cfg_file)


def test_invalid_fusion_method_raises(tmp_path: Path) -> None:
    """An unknown fusion method must raise ValidationError."""
    cfg_file = _write_minimal_config(
        tmp_path, {"fusion": {"method": "neural_blend"}}
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


# ---------------------------------------------------------------------------
# Paths helpers
# ---------------------------------------------------------------------------


def test_paths_raw_property(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PathsConfig.raw must compose data_root + raw_subdir."""
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("RESULTS_ROOT", raising=False)
    cfg_file = _write_minimal_config(
        tmp_path,
        {"paths": {"data_root": str(tmp_path), "results_root": str(tmp_path),
                   "raw_subdir": "raw", "processed_subdir": "processed"}},
    )
    cfg = load_config(cfg_file)
    assert cfg.paths.raw == tmp_path / "raw"


def test_paths_processed_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PathsConfig.processed must compose data_root + processed_subdir."""
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("RESULTS_ROOT", raising=False)
    cfg_file = _write_minimal_config(
        tmp_path,
        {"paths": {"data_root": str(tmp_path), "results_root": str(tmp_path),
                   "raw_subdir": "raw", "processed_subdir": "processed"}},
    )
    cfg = load_config(cfg_file)
    assert cfg.paths.processed == tmp_path / "processed"
