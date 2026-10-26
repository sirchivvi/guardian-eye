"""
Audio feature extraction and preprocessing for GuardianEye.

Validates and executes the exact audio pipeline:
- Resample to 8 kHz mono
- Crop or pad to 32,000 samples (4.0 seconds)
- Generate a 64 x 251 log-mel representation (n_fft=512, hop_length=128, n_mels=64)
- SpecAugment (2 frequency masks max width 8, 2 time masks max width 16)
  applied ONLY on training data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE: int = 8000
TARGET_DURATION_SEC: float = 4.0
TARGET_SAMPLES: int = 32000  # 4.0 s * 8000 Hz
N_MELS: int = 64
N_FFT: int = 512
HOP_LENGTH: int = 128
EXPECTED_SPECTROGRAM_SHAPE: tuple[int, int, int] = (1, 64, 251)


def load_audio(path: Path | str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file into a (channels, samples) float32 tensor in [-1.0, 1.0].

    Attempts ``torchaudio.load`` first; falls back to ``scipy.io.wavfile`` if
    torchaudio lacks a backend (e.g. on Windows without torchcodec).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {path_obj}")

    try:
        waveform, sample_rate = torchaudio.load(str(path_obj))
        return waveform, sample_rate
    except Exception as exc:
        logger.debug(
            "torchaudio.load failed (%s); falling back to scipy.io.wavfile for %s",
            exc,
            path_obj,
        )
        try:
            import scipy.io.wavfile as wav

            sr, data = wav.read(str(path_obj))
            if data.dtype == np.int16:
                arr = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                arr = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                arr = (data.astype(np.float32) - 128.0) / 128.0
            else:
                arr = data.astype(np.float32)

            tensor = torch.from_numpy(arr)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)  # (1, T)
            elif tensor.ndim == 2:
                # scipy returns (samples, channels)
                tensor = tensor.transpose(0, 1)  # (C, T)
            return tensor, sr
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Failed to load audio {path_obj} with torchaudio and scipy: {fallback_exc}"
            ) from fallback_exc


class AudioProcessor:
    """
    Deterministic audio preprocessor and log-mel feature extractor.

    Parameters
    ----------
    sample_rate:
        Target sample rate in Hz (default 8000).
    target_samples:
        Target number of audio samples (default 32000 = 4.0s @ 8000Hz).
    n_mels:
        Number of mel frequency bins (default 64).
    n_fft:
        FFT window size (default 512).
    hop_length:
        Hop length in samples (default 128).
    freq_mask_param:
        Max width for frequency masking in SpecAugment (default 8).
    time_mask_param:
        Max width for time masking in SpecAugment (default 16).

    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        target_samples: int = TARGET_SAMPLES,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        freq_mask_param: int = 8,
        time_mask_param: int = 16,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def to_mono_and_resample(
        self, waveform: torch.Tensor, orig_sr: int
    ) -> torch.Tensor:
        """Convert multi-channel waveform to mono and resample to target_sample_rate."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if orig_sr != self.sample_rate:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Crop or zero-pad waveform to exactly target_samples (32,000 samples)."""
        length = waveform.shape[-1]
        if length < self.target_samples:
            pad = self.target_samples - length
            waveform = F.pad(waveform, (0, pad), mode="constant", value=0.0)
        elif length > self.target_samples:
            waveform = waveform[..., : self.target_samples]
        return waveform

    def compute_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram with shape (1, n_mels, 251).

        log_mel = log(clamp(mel, min=1e-5))
        """
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel

    def apply_spec_augment(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment (2 frequency masks + 2 time masks).

        Must ONLY be called on training data.
        """
        # Apply 2 frequency masks
        augmented = self.freq_mask(log_mel)
        augmented = self.freq_mask(augmented)
        # Apply 2 time masks
        augmented = self.time_mask(augmented)
        augmented = self.time_mask(augmented)
        return augmented

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        is_train: bool = False,
    ) -> torch.Tensor:
        """
        Full waveform processing pipeline.

        Returns (1, 64, 251) log-mel spectrogram. SpecAugment applied only if is_train=True.
        """
        mono_resampled = self.to_mono_and_resample(waveform, sample_rate)
        aligned = self.crop_or_pad(mono_resampled)
        log_mel = self.compute_log_mel(aligned)

        if is_train:
            log_mel = self.apply_spec_augment(log_mel)

        return log_mel

    def process_file(
        self,
        file_path: Path | str,
        is_train: bool = False,
    ) -> torch.Tensor:
        """Load audio file and process into (1, 64, 251) log-mel spectrogram."""
        waveform, sr = load_audio(file_path)
        return self.process_waveform(waveform, sr, is_train=is_train)
