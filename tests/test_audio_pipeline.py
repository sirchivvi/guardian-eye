"""
Unit tests for GuardianEye audio pipeline and feature processing.

Validates:
- Resampling to 8 kHz mono
- Crop/pad to 32,000 samples
- Exact 64 x 251 log-mel representation
- SpecAugment isolation: active in training, strictly bypassed in evaluation
- Audio file loading and fallback
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wav
import torch

from src.audio.features import (
    EXPECTED_SPECTROGRAM_SHAPE,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    AudioProcessor,
    load_audio,
)


@pytest.fixture
def synthetic_wav_file(tmp_path: Path) -> Path:
    """Create a temporary synthetic WAV file (44.1 kHz stereo, 1.5 seconds)."""
    sr = 44100
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Stereo tone
    ch1 = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    ch2 = (np.cos(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    stereo_data = np.stack([ch1, ch2], axis=1)

    wav_path = tmp_path / "test_stereo.wav"
    wav.write(str(wav_path), sr, stereo_data)
    return wav_path


def test_load_audio_reads_wav(synthetic_wav_file: Path) -> None:
    """load_audio must read stereo WAV and return float32 tensor with correct channels and sr."""
    waveform, sr = load_audio(synthetic_wav_file)
    assert sr == 44100
    assert waveform.ndim == 2
    assert waveform.shape[0] == 2  # stereo
    assert waveform.dtype == torch.float32
    assert -1.05 <= waveform.min().item() <= 1.05


def test_audio_processor_mono_and_resample() -> None:
    """AudioProcessor must convert stereo 44.1 kHz input to mono 8.0 kHz."""
    proc = AudioProcessor(sample_rate=TARGET_SAMPLE_RATE)
    # Stereo 44.1 kHz signal
    stereo_44k = torch.randn(2, 44100)
    mono_8k = proc.to_mono_and_resample(stereo_44k, orig_sr=44100)

    assert mono_8k.shape[0] == 1  # mono
    assert mono_8k.shape[1] == 8000  # 1 second @ 8000 Hz


def test_audio_processor_crop_and_pad() -> None:
    """AudioProcessor must crop long signals and pad short signals to exactly 32,000 samples."""
    proc = AudioProcessor(target_samples=TARGET_SAMPLES)

    # Short signal: 10,000 samples -> must pad with 22,000 zeros
    short_sig = torch.ones(1, 10000)
    padded = proc.crop_or_pad(short_sig)
    assert padded.shape == (1, 32000)
    assert (padded[0, :10000] == 1.0).all()
    assert (padded[0, 10000:] == 0.0).all()

    # Long signal: 50,000 samples -> must crop to 32,000
    long_sig = torch.arange(50000, dtype=torch.float32).unsqueeze(0)
    cropped = proc.crop_or_pad(long_sig)
    assert cropped.shape == (1, 32000)
    assert (cropped[0] == torch.arange(32000, dtype=torch.float32)).all()


def test_audio_processor_exact_spectrogram_dimensions() -> None:
    """Mel spectrogram output must strictly match (1, 64, 251) dimensions."""
    proc = AudioProcessor()
    waveform = torch.randn(1, 32000)

    spec = proc.compute_log_mel(waveform)
    assert spec.shape == EXPECTED_SPECTROGRAM_SHAPE, (
        f"Expected {EXPECTED_SPECTROGRAM_SHAPE}, got {spec.shape}"
    )


def test_spec_augment_isolation() -> None:
    """SpecAugment must be active when is_train=True, and strictly bypassed when is_train=False."""
    proc = AudioProcessor()
    waveform = torch.ones(1, 32000)

    # In evaluation mode (is_train=False): deterministic and no masking
    eval_spec_1 = proc.process_waveform(waveform, sample_rate=8000, is_train=False)
    eval_spec_2 = proc.process_waveform(waveform, sample_rate=8000, is_train=False)

    assert torch.equal(eval_spec_1, eval_spec_2), "Evaluation spectrogram must be deterministic."
    # None of the bins should be zeroed by masking (since input is positive constant)
    assert (eval_spec_1 == 0.0).sum().item() == 0, "No masking should occur when is_train=False."

    # In training mode (is_train=True): SpecAugment is applied
    train_spec = proc.process_waveform(waveform, sample_rate=8000, is_train=True)
    assert train_spec.shape == EXPECTED_SPECTROGRAM_SHAPE
    # SpecAugment zeroes out frequency/time bands
    assert (train_spec == 0.0).any().item(), "SpecAugment must mask values when is_train=True."


def test_full_pipeline_from_file(synthetic_wav_file: Path) -> None:
    """Full process_file pipeline must execute cleanly on a raw WAV file."""
    proc = AudioProcessor()
    spec = proc.process_file(synthetic_wav_file, is_train=False)
    assert spec.shape == (1, 64, 251)
