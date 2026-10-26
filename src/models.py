"""
GuardianEye unimodal model architectures.

Implements:
- ImageEncoder: 4-block CNN, 128-d penultimate embedding, 4-class classifier head
  (human, vehicle, animal, empty) with opt-in ResNet-18 support.
- AudioEncoder: 4-block CNN for log-mel spectrograms (1, 64, 251), 128-d penultimate
  embedding, 2-class classifier head (background, gunshot).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical class labels and indices
IMAGE_CLASSES: list[str] = ["human", "vehicle", "animal", "empty"]
IMAGE_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(IMAGE_CLASSES)}

AUDIO_CLASSES: list[str] = ["background", "gunshot"]
AUDIO_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(AUDIO_CLASSES)}


class ConvBlock(nn.Module):
    """A standard Convolution-BatchNorm-ReLU-MaxPool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv, batchnorm, relu, and maxpool."""
        return self.block(x)


class ImageEncoder(nn.Module):
    """
    Image branch encoder.

    Architecture:
    - 4 Conv blocks: 3 -> 32 -> 64 -> 128 -> 256 channels.
    - Adaptive average pooling to (1, 1).
    - Linear projection to penultimate 128-d embedding.
    - Linear classification head to 4 classes (human, vehicle, animal, empty).
    - Opt-in support for pretrained ResNet-18 backbone.
    """

    def __init__(
        self,
        num_classes: int = 4,
        embedding_dim: int = 128,
        use_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained

        if use_pretrained:
            try:
                import torchvision.models as models

                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                feature_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.backbone = backbone
                self.fc_embed = nn.Linear(feature_dim, embedding_dim)
            except ImportError as err:
                raise ImportError(
                    "torchvision is required for pretrained ResNet-18 backbone."
                ) from err
        else:
            self.backbone = nn.Sequential(
                ConvBlock(3, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.fc_embed = nn.Linear(256, embedding_dim)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (B, 3, H, W).
        return_embedding:
            If True, returns (logits, embedding). Otherwise returns logits.

        """
        features = self.backbone(x)
        if self.use_pretrained and features.ndim > 2:
            features = torch.flatten(features, 1)
        embedding = self.fc_embed(features)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-d penultimate embedding."""
        features = self.backbone(x)
        if self.use_pretrained and features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.fc_embed(features)


class AudioEncoder(nn.Module):
    """
    Audio branch encoder for log-mel spectrograms.

    Architecture:
    - 4 Conv blocks on (1, 64, 251) log-mel input: 1 -> 32 -> 64 -> 128 -> 256 channels.
    - Adaptive average pooling to (1, 1).
    - Linear projection to penultimate 128-d embedding.
    - Linear classification head to 2 classes (background, gunshot).
    """

    def __init__(
        self,
        num_classes: int = 2,
        embedding_dim: int = 128,
        use_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained

        if use_pretrained:
            logger.info("Custom AudioEncoder used; pretrained audio backbone is opt-in.")

        self.backbone = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc_embed = nn.Linear(256, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input log-mel spectrogram of shape (B, 1, 64, 251).
        return_embedding:
            If True, returns (logits, embedding). Otherwise returns logits.

        """
        features = self.backbone(x)
        embedding = self.fc_embed(features)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-d penultimate embedding."""
        features = self.backbone(x)
        return self.fc_embed(features)
