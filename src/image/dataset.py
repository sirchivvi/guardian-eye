"""
PyTorch Dataset for GuardianEye image branch.

Reads images from manifest rows, validates existence, applies resizing
to (224, 224), ImageNet normalization, and optional data augmentation
during training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.models import IMAGE_CLASS_TO_IDX

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(
    image: Image.Image,
    target_size: tuple[int, int] = (224, 224),
    is_train: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """Resize, augment, and normalize PIL Image to (3, H, W) float32 tensor."""
    img = image.convert("RGB")
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0

    # Data augmentation on training split only
    if is_train:
        # Random horizontal flip
        rng = np.random.default_rng(seed)
        if rng.random() > 0.5:
            arr = np.fliplr(arr)

    # Normalize (H, W, C)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # Convert to (C, H, W)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
    return tensor


class ImageDataset(Dataset):
    """
    PyTorch Dataset reading images defined in a GuardianEye manifest DataFrame.

    Parameters
    ----------
    manifest_df:
        DataFrame containing manifest rows for image files.
    data_root:
        Path to the root data directory.
    target_size:
        Tuple of (height, width) for image resizing (default (224, 224)).
    is_train:
        Whether training augmentations should be enabled.

    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_root: Path | str,
        target_size: tuple[int, int] = (224, 224),
        is_train: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.is_train = is_train

        # Filter strictly to image modality rows
        if "modality" in manifest_df.columns:
            self.df = manifest_df[manifest_df["modality"] == "image"].reset_index(
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
        if label_str not in IMAGE_CLASS_TO_IDX:
            raise ValueError(f"Unknown image class label: {label_str}")
        label_idx = IMAGE_CLASS_TO_IDX[label_str]

        manifest_idx = int(row.get("manifest_idx", idx))
        view = str(row.get("view", "unknown"))

        try:
            with Image.open(full_path) as img:
                tensor = preprocess_image(
                    img,
                    target_size=self.target_size,
                    is_train=self.is_train,
                )
        except Exception as exc:
            logger.error("Failed to load image %s: %s", full_path, exc)
            raise

        return {
            "image": tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "manifest_idx": manifest_idx,
            "path": rel_path,
            "view": view,
            "label_str": label_str,
        }
