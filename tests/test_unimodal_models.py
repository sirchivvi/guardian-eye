"""
Unit tests for GuardianEye unimodal models (ImageEncoder and AudioEncoder).

Verifies:
- ImageEncoder 4-block CNN architecture, forward pass, and gradient backpropagation
- AudioEncoder 4-block CNN architecture, forward pass, and gradient backpropagation
- 128-d penultimate embeddings and classification heads
- Opt-in pretrained backbone behavior (custom CNN default)
"""

from __future__ import annotations

import pytest
import torch

from src.models import (
    AUDIO_CLASSES,
    IMAGE_CLASSES,
    AudioEncoder,
    ImageEncoder,
)


def test_image_encoder_default_architecture() -> None:
    """ImageEncoder default model must produce (B, 4) logits and (B, 128) embeddings."""
    model = ImageEncoder(num_classes=4, embedding_dim=128, use_pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    # Forward without embeddings
    logits = model(x, return_embedding=False)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 4)

    # Forward with embeddings
    logits, embedding = model(x, return_embedding=True)
    assert logits.shape == (2, 4)
    assert embedding.shape == (2, 128)

    # get_embedding method
    direct_embed = model.get_embedding(x)
    assert direct_embed.shape == (2, 128)


def test_image_encoder_backward_pass() -> None:
    """Gradients must propagate through all layers of ImageEncoder."""
    model = ImageEncoder(num_classes=4, embedding_dim=128, use_pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 2], dtype=torch.long)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    # Check that gradients exist for backbone and heads
    has_grad = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert all(has_grad), "All trainable parameters must receive gradients."


def test_audio_encoder_architecture() -> None:
    """AudioEncoder model must process (B, 1, 64, 251) input and output (B, 2) and (B, 128)."""
    model = AudioEncoder(num_classes=2, embedding_dim=128, use_pretrained=False)
    x = torch.randn(2, 1, 64, 251)

    # Forward without embeddings
    logits = model(x, return_embedding=False)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 2)

    # Forward with embeddings
    logits, embedding = model(x, return_embedding=True)
    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 128)

    # get_embedding method
    direct_embed = model.get_embedding(x)
    assert direct_embed.shape == (2, 128)


def test_audio_encoder_backward_pass() -> None:
    """Gradients must propagate through all layers of AudioEncoder."""
    model = AudioEncoder(num_classes=2, embedding_dim=128, use_pretrained=False)
    x = torch.randn(2, 1, 64, 251)
    target = torch.tensor([0, 1], dtype=torch.long)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    has_grad = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert all(has_grad), "All trainable parameters must receive gradients."


def test_image_encoder_pretrained_flag() -> None:
    """Pretrained backbone is opt-in and handles torchvision availability cleanly."""
    try:
        import torchvision  # noqa: F401

        # If torchvision installed, initialization must succeed
        model = ImageEncoder(use_pretrained=True)
        assert model.use_pretrained is True
    except ImportError:
        # If torchvision not installed, must raise ImportError
        with pytest.raises(ImportError, match="torchvision is required"):
            ImageEncoder(use_pretrained=True)


def test_classes_constants() -> None:
    """Verify canonical class names for image and audio branches."""
    assert IMAGE_CLASSES == ["human", "vehicle", "animal", "empty"]
    assert AUDIO_CLASSES == ["background", "gunshot"]
