"""GuardianEye audio branch package."""

from src.audio.dataset import AudioDataset
from src.audio.evaluate import evaluate_audio_branch, evaluate_audio_dataset
from src.audio.features import AudioProcessor, load_audio
from src.audio.train import (
    compute_audio_class_weights,
    train_audio_model,
    train_cross_site_audio,
    train_in_domain_audio,
)
from src.models import AUDIO_CLASS_TO_IDX, AUDIO_CLASSES, AudioEncoder

__all__ = [
    "AudioDataset",
    "AudioProcessor",
    "load_audio",
    "evaluate_audio_branch",
    "evaluate_audio_dataset",
    "compute_audio_class_weights",
    "train_audio_model",
    "train_in_domain_audio",
    "train_cross_site_audio",
    "AUDIO_CLASSES",
    "AUDIO_CLASS_TO_IDX",
    "AudioEncoder",
]
