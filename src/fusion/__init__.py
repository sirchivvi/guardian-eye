"""
GuardianEye multimodal fusion module.

Exposes:
- TemperatureScaler, CalibrationResult (temperature scaling)
- LateFusionModel, LateFusionResult (weighted score & rule head late fusion)
- extract_image_threat_probability, extract_audio_threat_probability
- fit_late_fusion_parameters, fit_rule_thresholds
- ImageOnlyBaseline, AudioOnlyBaseline, BaselinePrediction
"""

from __future__ import annotations

from src.fusion.baselines import (
    AudioOnlyBaseline,
    BaselinePrediction,
    ImageOnlyBaseline,
)
from src.fusion.calibrate import (
    CalibrationResult,
    TemperatureScaler,
)
from src.fusion.late import (
    LateFusionModel,
    LateFusionResult,
    extract_audio_threat_probability,
    extract_image_threat_probability,
    fit_late_fusion_parameters,
    fit_rule_thresholds,
)

__all__ = [
    "AudioOnlyBaseline",
    "BaselinePrediction",
    "CalibrationResult",
    "ImageOnlyBaseline",
    "LateFusionModel",
    "LateFusionResult",
    "TemperatureScaler",
    "extract_audio_threat_probability",
    "extract_image_threat_probability",
    "fit_late_fusion_parameters",
    "fit_rule_thresholds",
]
