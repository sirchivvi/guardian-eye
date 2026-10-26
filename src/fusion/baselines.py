"""
Unimodal baselines for GuardianEye multimodal fusion comparison.

Implements:
- ImageOnlyBaseline: Threat score P(human) + P(vehicle) with validation-fit decision threshold.
- AudioOnlyBaseline: Gunshot score P(gunshot) with validation-fit decision threshold.

Missing modality guarantee:
- If a baseline's own modality is dropped or missing, it strictly outputs score 0.0
  and prediction 0 (negative). It NEVER accesses or borrows the other sensor.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import safe_divide
from src.fusion.calibrate import TemperatureScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselinePrediction:
    """Outcome for a unimodal baseline evaluation."""

    score: float
    prediction: int
    modality_present: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ImageOnlyBaseline:
    """
    Image-only threat detection baseline: S = P(human) + P(vehicle) >= tau_image.

    If image modality is missing or dropped, strictly outputs score 0.0 and prediction 0.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        scaler: TemperatureScaler | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.scaler = scaler

    def predict(
        self,
        image_prob: float | None = None,
        image_logits: torch.Tensor | None = None,
        image_present: bool = True,
    ) -> BaselinePrediction:
        """
        Evaluate image baseline on a single sample.

        Parameters
        ----------
        image_prob:
            Optional scalar image threat probability.
        image_logits:
            Optional 4-class image logits tensor.
        image_present:
            Whether image sensor is available.

        Returns
        -------
        BaselinePrediction
            Predicted threat score, binary alert, and presence status.

        """
        if not image_present:
            return BaselinePrediction(
                score=0.0,
                prediction=0,
                modality_present=False,
                threshold=self.threshold,
            )

        if image_prob is not None:
            p = float(image_prob)
        elif image_logits is not None:
            logits = image_logits.unsqueeze(0) if image_logits.ndim == 1 else image_logits
            if self.scaler is not None:
                probs = self.scaler.predict_proba(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            p = float((probs[..., 0] + probs[..., 1]).squeeze().item())
        else:
            return BaselinePrediction(
                score=0.0,
                prediction=0,
                modality_present=False,
                threshold=self.threshold,
            )

        p = max(0.0, min(1.0, p))
        pred = int(p >= self.threshold)

        return BaselinePrediction(
            score=p,
            prediction=pred,
            modality_present=True,
            threshold=self.threshold,
        )

    def predict_batch(
        self,
        image_probs: torch.Tensor | np.ndarray | None,
        image_present: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Vectorized batch evaluation for image-only baseline.

        Parameters
        ----------
        image_probs:
            Array/tensor of image threat probabilities.
        image_present:
            Optional boolean mask of image availability.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with 'scores' and 'predictions'.

        """
        if image_probs is None:
            return {"scores": np.array([]), "predictions": np.array([])}

        p = np.asarray(image_probs, dtype=np.float32)
        n = len(p)
        pres = (
            np.asarray(image_present, dtype=bool)
            if image_present is not None
            else np.ones(n, dtype=bool)
        )

        scores = np.where(pres, p, 0.0).astype(np.float32)
        preds = np.where(pres & (scores >= self.threshold), 1, 0).astype(int)

        return {"scores": scores, "predictions": preds}

    def fit_threshold(
        self,
        val_image_probs: torch.Tensor | np.ndarray,
        val_targets: torch.Tensor | np.ndarray,
        threshold_step: float = 0.05,
    ) -> float:
        """
        Fit decision threshold on validation data to maximize F1 score.

        Parameters
        ----------
        val_image_probs:
            Validation image threat probabilities.
        val_targets:
            Validation ground truth binary targets.
        threshold_step:
            Grid step size.

        Returns
        -------
        float
            Optimal decision threshold.

        """
        probs = np.asarray(val_image_probs, dtype=np.float32)
        targets = np.asarray(val_targets, dtype=int)

        if len(probs) != len(targets) or len(targets) == 0:
            raise ValueError("Invalid validation arrays.")

        unique = np.unique(targets)
        if len(unique) < 2:
            logger.warning("Single class in validation targets. Keeping default threshold.")
            return self.threshold

        thresholds = np.round(np.arange(threshold_step, 1.0, threshold_step), 2)
        best_f1 = -1.0
        best_t = self.threshold

        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = int(np.sum((targets == 1) & (preds == 1)))
            fp = int(np.sum((targets == 0) & (preds == 1)))
            fn = int(np.sum((targets == 1) & (preds == 0)))
            prec = safe_divide(tp, tp + fp)
            rec = safe_divide(tp, tp + fn)
            f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.threshold = best_t
        logger.info("Image baseline threshold fit to %.2f (Val F1: %.4f)", best_t, best_f1)
        return best_t


class AudioOnlyBaseline:
    """
    Audio-only threat detection baseline: S = P(gunshot) >= tau_audio.

    If audio modality is missing or dropped, strictly outputs score 0.0 and prediction 0.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        scaler: TemperatureScaler | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.scaler = scaler

    def predict(
        self,
        audio_prob: float | None = None,
        audio_logits: torch.Tensor | None = None,
        audio_present: bool = True,
    ) -> BaselinePrediction:
        """
        Evaluate audio baseline on a single sample.

        Parameters
        ----------
        audio_prob:
            Optional scalar audio gunshot probability.
        audio_logits:
            Optional 2-class audio logits tensor.
        audio_present:
            Whether audio sensor is available.

        Returns
        -------
        BaselinePrediction
            Predicted gunshot score, binary alert, and presence status.

        """
        if not audio_present:
            return BaselinePrediction(
                score=0.0,
                prediction=0,
                modality_present=False,
                threshold=self.threshold,
            )

        if audio_prob is not None:
            p = float(audio_prob)
        elif audio_logits is not None:
            logits = audio_logits.unsqueeze(0) if audio_logits.ndim == 1 else audio_logits
            if self.scaler is not None:
                probs = self.scaler.predict_proba(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            p = float(probs[..., 1].squeeze().item())
        else:
            return BaselinePrediction(
                score=0.0,
                prediction=0,
                modality_present=False,
                threshold=self.threshold,
            )

        p = max(0.0, min(1.0, p))
        pred = int(p >= self.threshold)

        return BaselinePrediction(
            score=p,
            prediction=pred,
            modality_present=True,
            threshold=self.threshold,
        )

    def predict_batch(
        self,
        audio_probs: torch.Tensor | np.ndarray | None,
        audio_present: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Vectorized batch evaluation for audio-only baseline.

        Parameters
        ----------
        audio_probs:
            Array/tensor of audio gunshot probabilities.
        audio_present:
            Optional boolean mask of audio availability.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with 'scores' and 'predictions'.

        """
        if audio_probs is None:
            return {"scores": np.array([]), "predictions": np.array([])}

        p = np.asarray(audio_probs, dtype=np.float32)
        n = len(p)
        pres = (
            np.asarray(audio_present, dtype=bool)
            if audio_present is not None
            else np.ones(n, dtype=bool)
        )

        scores = np.where(pres, p, 0.0).astype(np.float32)
        preds = np.where(pres & (scores >= self.threshold), 1, 0).astype(int)

        return {"scores": scores, "predictions": preds}

    def fit_threshold(
        self,
        val_audio_probs: torch.Tensor | np.ndarray,
        val_targets: torch.Tensor | np.ndarray,
        threshold_step: float = 0.05,
    ) -> float:
        """
        Fit decision threshold on validation data to maximize F1 score.

        Parameters
        ----------
        val_audio_probs:
            Validation audio gunshot probabilities.
        val_targets:
            Validation ground truth binary gunshot targets.
        threshold_step:
            Grid step size.

        Returns
        -------
        float
            Optimal decision threshold.

        """
        probs = np.asarray(val_audio_probs, dtype=np.float32)
        targets = np.asarray(val_targets, dtype=int)

        if len(probs) != len(targets) or len(targets) == 0:
            raise ValueError("Invalid validation arrays.")

        unique = np.unique(targets)
        if len(unique) < 2:
            logger.warning("Single class in validation targets. Keeping default threshold.")
            return self.threshold

        thresholds = np.round(np.arange(threshold_step, 1.0, threshold_step), 2)
        best_f1 = -1.0
        best_t = self.threshold

        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = int(np.sum((targets == 1) & (preds == 1)))
            fp = int(np.sum((targets == 0) & (preds == 1)))
            fn = int(np.sum((targets == 1) & (preds == 0)))
            prec = safe_divide(tp, tp + fp)
            rec = safe_divide(tp, tp + fn)
            f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.threshold = best_t
        logger.info("Audio baseline threshold fit to %.2f (Val F1: %.4f)", best_t, best_f1)
        return best_t
