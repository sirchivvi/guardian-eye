"""Unit tests for notebook validation and orchestration-only compliance.

Verifies:
- Both notebooks/colab_run.ipynb and notebooks/kaggle_run.ipynb exist
- Both notebooks are valid JSON matching nbformat 4
- Neither notebook defines model architectures, training loops, dataset loading,
  or metric calculations
- Neither notebook embeds hard-coded tokens or secrets
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATHS: list[Path] = [
    Path("notebooks/colab_run.ipynb"),
    Path("notebooks/kaggle_run.ipynb"),
]

PROHIBITED_SNIPPETS: list[str] = [
    "nn.Module",
    "torch.nn",
    "loss.backward",
    "optimizer.step",
    "DataLoader(",
    "def train(",
    "def evaluate(",
    "accuracy_score(",
    "f1_score(",
]


def test_notebook_files_exist() -> None:
    """Both Colab and Kaggle notebooks must exist in notebooks/."""
    for nb_path in NOTEBOOK_PATHS:
        assert nb_path.is_file(), f"Missing notebook: {nb_path}"


def test_notebooks_are_valid_nbformat_v4() -> None:
    """Notebooks must be valid JSON conforming to nbformat 4."""
    for nb_path in NOTEBOOK_PATHS:
        with nb_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert isinstance(data, dict), f"{nb_path} root must be a dict"
        assert data.get("nbformat") == 4, f"{nb_path} must be nbformat 4"
        assert "cells" in data, f"{nb_path} must contain 'cells'"
        assert isinstance(data["cells"], list), f"{nb_path} 'cells' must be a list"
        assert len(data["cells"]) > 0, f"{nb_path} must not be empty"


def test_notebooks_contain_orchestration_only() -> None:
    """Notebooks must not implement modeling, training, or metric algorithms."""
    for nb_path in NOTEBOOK_PATHS:
        with nb_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) > 0, f"{nb_path} must have code cells"

        for idx, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            for snippet in PROHIBITED_SNIPPETS:
                assert snippet not in source, (
                    f"Prohibited logic '{snippet}' found in {nb_path} cell {idx}. "
                    "Notebooks must be thin orchestration wrappers only."
                )


def test_notebooks_do_not_contain_secrets() -> None:
    """Notebooks must not contain hard-coded personal access tokens or passwords."""
    suspicious_patterns = [
        "ghp_",  # GitHub personal access token prefix
        "github_pat_",  # fine-grained PAT prefix
    ]
    for nb_path in NOTEBOOK_PATHS:
        content = nb_path.read_text(encoding="utf-8")
        for pattern in suspicious_patterns:
            assert pattern not in content, (
                f"Potential secret token pattern '{pattern}' found in {nb_path}"
            )
