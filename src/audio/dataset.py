"""
PyTorch Dataset for GuardianEye audio branch.

Reads audio files from manifest rows, validates existence, applies the exact
8 kHz mono 32,000 sample log-mel transformation, and SpecAugment during training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.audio.features import AudioProcessor
from src.models import AUDIO_CLASS_TO_IDX

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """
    PyTorch Dataset reading audio files defined in a GuardianEye manifest DataFrame.

    Parameters
    ----------
    manifest_df:
        DataFrame containing manifest rows for audio files.
    data_root:
        Path to the root data directory.
    processor:
        Optional AudioProcessor instance (creates default if None).
    is_train:
        Whether training SpecAugment should be enabled.

    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_root: Path | str,
        processor: AudioProcessor | None = None,
        is_train: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.processor = processor or AudioProcessor()
        self.is_train = is_train

        # Filter strictly to audio modality rows
        if "modality" in manifest_df.columns:
            self.df = manifest_df[manifest_df["modality"] == "audio"].reset_index(
                names="manifest_idx"
            )
        else:
            self.df = manifest_df.reset_index(names="manifest_idx")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = row["path"]
        full_path = self.data_root / rel_path

        label_str = row["label"]
        if label_str not in AUDIO_CLASS_TO_IDX:
            raise ValueError(f"Unknown audio class label: {label_str}")
        label_idx = AUDIO_CLASS_TO_IDX[label_str]

        manifest_idx = int(row.get("manifest_idx", idx))
        source = str(row.get("source", "unknown"))
        noise_tag = str(row.get("noise_tag", "not_thunder_wind"))

        try:
            spectrogram = self.processor.process_file(
                full_path, is_train=self.is_train
            )
        except Exception as exc:
            logger.error("Failed to load audio %s: %s", full_path, exc)
            raise

        return {
            "audio": spectrogram,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "manifest_idx": manifest_idx,
            "path": rel_path,
            "source": source,
            "noise_tag": noise_tag,
            "label_str": label_str,
        }
