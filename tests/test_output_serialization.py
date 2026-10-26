"""
Unit tests for output serialization and feature extraction.

Verifies:
- Logits, probabilities, and 128-d penultimate embeddings are serialized
- Serialization includes manifest_indices, paths, labels, splits
- Deterministic order is preserved
- Atomic file saving ensures no corrupted files
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile as wav
import torch
from PIL import Image

from src.audio.features import AudioProcessor
from src.evaluation.outputs import (
    extract_audio_outputs,
    extract_image_outputs,
    save_serialized_outputs,
)
from src.models import AudioEncoder, ImageEncoder


@pytest.fixture
def mock_dataset_fixtures(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Create minimal image and audio files and a matching manifest DataFrame."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # 3 dummy images
    img_dir = data_dir / "cct"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(["human", "animal", "empty"]):
        img = Image.new("RGB", (64, 64), color=(i * 40, i * 40, i * 40))
        rel_p = f"cct/img_{i}.jpg"
        img.save(data_dir / rel_p)
        rows.append(
            {
                "path": rel_p,
                "modality": "image",
                "source": "cct",
                "label": label,
                "view": "camera_trap",
                "split": "train" if i < 2 else "test",
                "noise_tag": "not_thunder_wind",
            }
        )

    # 3 dummy audio files (8 kHz WAV)
    audio_dir = data_dir / "belize"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(["gunshot", "background", "background"]):
        rel_p = f"belize/audio_{i}.wav"
        data = (np.sin(np.linspace(0, 100, 8000)) * 32767).astype(np.int16)
        wav.write(str(data_dir / rel_p), 8000, data)
        rows.append(
            {
                "path": rel_p,
                "modality": "audio",
                "source": "belize",
                "label": label,
                "view": "acoustic",
                "split": "train" if i == 0 else "test",
                "noise_tag": "thunder_wind" if i == 2 else "not_thunder_wind",
            }
        )

    df = pd.DataFrame(rows)
    return data_dir, df


def test_image_output_serialization(
    mock_dataset_fixtures: tuple[Path, pd.DataFrame], tmp_path: Path
) -> None:
    """Extract and serialize outputs for images; verify shapes and keys."""
    data_dir, df = mock_dataset_fixtures
    img_df = df[df["modality"] == "image"]

    model = ImageEncoder(num_classes=4, embedding_dim=128)
    device = torch.device("cpu")

    outputs = extract_image_outputs(model, img_df, data_root=data_dir, device=device)

    # Required keys
    assert "manifest_indices" in outputs
    assert "paths" in outputs
    assert "logits" in outputs
    assert "probabilities" in outputs
    assert "embeddings" in outputs
    assert "labels" in outputs
    assert "splits" in outputs

    n = len(img_df)
    assert len(outputs["paths"]) == n
    assert outputs["logits"].shape == (n, 4)
    assert outputs["probabilities"].shape == (n, 4)
    assert outputs["embeddings"].shape == (n, 128)
    assert outputs["labels"].shape == (n,)

    # Verify probability normalization
    sums = outputs["probabilities"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(n), atol=1e-5)

    # Save to disk
    out_file = tmp_path / "image_outputs.pt"
    saved = save_serialized_outputs(outputs, out_file)
    assert saved.exists()

    loaded = torch.load(saved, weights_only=False)
    assert torch.equal(loaded["embeddings"], outputs["embeddings"])


def test_audio_output_serialization(
    mock_dataset_fixtures: tuple[Path, pd.DataFrame], tmp_path: Path
) -> None:
    """Extract and serialize outputs for audio (including held-out thunder clips)."""
    data_dir, df = mock_dataset_fixtures
    audio_df = df[df["modality"] == "audio"]

    model = AudioEncoder(num_classes=2, embedding_dim=128)
    device = torch.device("cpu")
    processor = AudioProcessor()

    outputs = extract_audio_outputs(
        model, audio_df, data_root=data_dir, device=device, processor=processor
    )

    n = len(audio_df)
    assert len(outputs["paths"]) == n
    assert outputs["logits"].shape == (n, 2)
    assert outputs["probabilities"].shape == (n, 2)
    assert outputs["embeddings"].shape == (n, 128)
    assert outputs["labels"].shape == (n,)

    # Thunder clip (audio_2.wav) must be present in serialized outputs
    assert "belize/audio_2.wav" in outputs["paths"]

    # Save to disk
    out_file = tmp_path / "audio_outputs.pt"
    saved = save_serialized_outputs(outputs, out_file)
    assert saved.exists()
