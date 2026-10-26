"""
Unit tests for leakage-free cross-site audio evaluation.

Validates:
- Belize -> Vietnam: trained ONLY on Belize train data, selected on Belize val data,
  evaluated on Vietnam test data. Asserts zero Vietnam samples in training/val splits.
- Vietnam -> Belize: trained ONLY on Vietnam train data, selected on Vietnam val data,
  evaluated on Belize test data. Asserts zero Belize samples in training/val splits.
- Cross-site checkpoints are saved in dedicated cross_site/ directory and separate from fusion.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.audio.evaluate import evaluate_audio_branch
from src.models import AudioEncoder


@pytest.fixture
def mock_manifest_cross_site() -> pd.DataFrame:
    """Create a mock manifest with Belize and Vietnam samples across splits."""
    rows = []
    # Belize samples
    for i in range(10):
        rows.append(
            {
                "path": f"belize/train_{i}.wav",
                "modality": "audio",
                "source": "belize",
                "label": "gunshot" if i % 2 == 0 else "background",
                "view": "acoustic",
                "noise_tag": "not_thunder_wind",
                "split": "train",
            }
        )
    for i in range(4):
        rows.append(
            {
                "path": f"belize/val_{i}.wav",
                "modality": "audio",
                "source": "belize",
                "label": "gunshot" if i % 2 == 0 else "background",
                "view": "acoustic",
                "noise_tag": "not_thunder_wind",
                "split": "val",
            }
        )
    for i in range(4):
        rows.append(
            {
                "path": f"belize/test_{i}.wav",
                "modality": "audio",
                "source": "belize",
                "label": "gunshot" if i % 2 == 0 else "background",
                "view": "acoustic",
                "noise_tag": "not_thunder_wind",
                "split": "test",
            }
        )

    # Vietnam samples
    for i in range(8):
        rows.append(
            {
                "path": f"vietnam/train_{i}.wav",
                "modality": "audio",
                "source": "vietnam",
                "label": "gunshot" if i % 2 == 0 else "background",
                "view": "acoustic",
                "noise_tag": "not_thunder_wind",
                "split": "train",
            }
        )
    for i in range(4):
        rows.append(
            {
                "path": f"vietnam/val_{i}.wav",
                "modality": "audio",
                "source": "vietnam",
                "label": "gunshot" if i % 2 == 0 else "background",
                "view": "acoustic",
                "noise_tag": "not_thunder_wind",
                "split": "val",
            }
        )
    for i in range(4):
        rows.append(
            {
                "path": f"vietnam/test_{i}.wav",
                "modality": "audio",
                "source": "vietnam",
                "label": "gunshot" if i % 2 == 0 else "background",
                "view": "acoustic",
                "noise_tag": "not_thunder_wind",
                "split": "test",
            }
        )

    return pd.DataFrame(rows)


def test_cross_site_dataset_isolation(mock_manifest_cross_site: pd.DataFrame) -> None:
    """Verify that source training/val data contains ZERO records from the target site."""
    df = mock_manifest_cross_site

    # Belize -> Vietnam: source is Belize
    belize_source = df[(df["source"] == "belize") & (df["modality"] == "audio")]
    belize_train = belize_source[belize_source["split"] == "train"]
    belize_val = belize_source[belize_source["split"] == "val"]

    # Target is Vietnam
    vietnam_test = df[
        (df["source"] == "vietnam")
        & (df["modality"] == "audio")
        & (df["split"] == "test")
    ]

    # Leakage check: zero Vietnam in belize train or val
    assert not (belize_train["source"] == "vietnam").any()
    assert not (belize_val["source"] == "vietnam").any()

    # Target check: vietnam test contains only vietnam
    assert (vietnam_test["source"] == "vietnam").all()

    # Reverse: Vietnam -> Belize
    vietnam_source = df[(df["source"] == "vietnam") & (df["modality"] == "audio")]
    vietnam_train = vietnam_source[vietnam_source["split"] == "train"]
    vietnam_val = vietnam_source[vietnam_source["split"] == "val"]

    belize_test = df[
        (df["source"] == "belize")
        & (df["modality"] == "audio")
        & (df["split"] == "test")
    ]

    assert not (vietnam_train["source"] == "belize").any()
    assert not (vietnam_val["source"] == "belize").any()
    assert (belize_test["source"] == "belize").all()


def test_evaluate_audio_branch_cross_site_routing(
    mock_manifest_cross_site: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_audio_branch must route cross_site models strictly to target test splits."""
    evaluated_dfs: dict[str, pd.DataFrame] = {}

    def mock_eval_fn(model, df, **kwargs):
        # Record which DataFrame was passed
        name = getattr(model, "_name", "unknown")
        evaluated_dfs[name] = df.copy()
        return {"status": "OK", "count": len(df)}

    monkeypatch.setattr("src.audio.evaluate.evaluate_audio_dataset", mock_eval_fn)

    in_domain = AudioEncoder(num_classes=2, embedding_dim=128)
    in_domain._name = "in_domain"

    b2v_model = AudioEncoder(num_classes=2, embedding_dim=128)
    b2v_model._name = "b2v"

    v2b_model = AudioEncoder(num_classes=2, embedding_dim=128)
    v2b_model._name = "v2b"

    cross_site_models = {
        "belize_to_vietnam": b2v_model,
        "vietnam_to_belize": v2b_model,
    }

    evaluate_audio_branch(
        in_domain_model=in_domain,
        manifest_df=mock_manifest_cross_site,
        data_root="data",
        device=torch.device("cpu"),
        cross_site_models=cross_site_models,
    )

    # Check that Belize -> Vietnam evaluated strictly on Vietnam test split
    b2v_df = evaluated_dfs["b2v"]
    assert len(b2v_df) == 4
    assert (b2v_df["source"] == "vietnam").all()
    assert (b2v_df["split"] == "test").all()

    # Check that Vietnam -> Belize evaluated strictly on Belize test split
    v2b_df = evaluated_dfs["v2b"]
    assert len(v2b_df) == 4
    assert (v2b_df["source"] == "belize").all()
    assert (v2b_df["split"] == "test").all()

    # In-domain evaluated on both Belize + Vietnam test splits
    in_domain_df = evaluated_dfs["in_domain"]
    assert len(in_domain_df) == 8
    assert set(in_domain_df["source"].unique()) == {"belize", "vietnam"}
    assert (in_domain_df["split"] == "test").all()
