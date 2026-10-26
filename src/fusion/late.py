"""
Late fusion and decision logic for GuardianEye multimodal threat detection.

Implements:
- Weighted late-fusion scoring using calibrated image and audio threat probabilities.
- Validation-only grid search for fusion weights (w_image, w_audio) and decision threshold tau.
- Three-tier HIGH / MEDIUM / LOW rule head combining branch thresholds.
- Missing-modality conservative policy (threshold raised by tau_delta capped at tau_cap).
- Strict split boundary preservation (validation sets tune parameters; test sets evaluate).
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
class LateFusionResult:
    """Detailed prediction result for a single multimodal event."""

    fused_score: float
    weighted_prediction: int
    rule_prediction: int
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    image_prob: float
    audio_prob: float
    image_present: bool
    audio_present: bool
    effective_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


def extract_image_threat_probability(
    image_logits: torch.Tensor,
    scaler: TemperatureScaler | None = None,
) -> torch.Tensor:
    """
    Compute P(human) + P(vehicle) from image branch prediction logits.

    Parameters
    ----------
    image_logits:
        Logits tensor with shape (..., 4) for classes [human, vehicle, animal, empty].
    scaler:
        Optional calibrated TemperatureScaler.

    Returns
    -------
    torch.Tensor
        Threat probability in [0.0, 1.0] with shape (...,).

    """
    if scaler is not None:
        probs = scaler.predict_proba(image_logits)
    else:
        probs = F.softmax(image_logits, dim=-1)

    # Class 0: human, Class 1: vehicle
    threat_prob = probs[..., 0] + probs[..., 1]
    return torch.clamp(threat_prob, 0.0, 1.0)


def extract_audio_threat_probability(
    audio_logits: torch.Tensor,
    scaler: TemperatureScaler | None = None,
) -> torch.Tensor:
    """
    Compute P(gunshot) from audio branch prediction logits.

    Parameters
    ----------
    audio_logits:
        Logits tensor with shape (..., 2) for classes [background, gunshot].
    scaler:
        Optional calibrated TemperatureScaler.

    Returns
    -------
    torch.Tensor
        Threat probability in [0.0, 1.0] with shape (...,).

    """
    if scaler is not None:
        probs = scaler.predict_proba(audio_logits)
    else:
        probs = F.softmax(audio_logits, dim=-1)

    # Class 1: gunshot
    threat_prob = probs[..., 1]
    return torch.clamp(threat_prob, 0.0, 1.0)


class LateFusionModel:
    """
    Late-fusion model combining unimodal calibrated probabilities.

    Supports:
    - Weighted score: w_img * P_img + w_aud * P_aud >= tau
    - Rule head: HIGH if both sensors clear thresholds; MEDIUM if one; LOW otherwise.
    - Missing modality policy: conservative threshold raise by tau_delta (capped at tau_cap);
      rule head cannot reach HIGH with one sensor.
    """

    def __init__(
        self,
        image_weight: float = 0.6,
        audio_weight: float = 0.4,
        threshold: float = 0.5,
        rule_img_threshold: float = 0.5,
        rule_aud_threshold: float = 0.5,
        conservative_tau_delta: float = 0.15,
        tau_cap: float = 0.95,
        image_scaler: TemperatureScaler | None = None,
        audio_scaler: TemperatureScaler | None = None,
    ) -> None:
        self.image_weight = float(image_weight)
        self.audio_weight = float(audio_weight)
        self.threshold = float(threshold)
        self.rule_img_threshold = float(rule_img_threshold)
        self.rule_aud_threshold = float(rule_aud_threshold)
        self.conservative_tau_delta = float(conservative_tau_delta)
        self.tau_cap = float(tau_cap)
        self.image_scaler = image_scaler
        self.audio_scaler = audio_scaler

        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights and thresholds are in valid ranges."""
        if not (0.0 <= self.image_weight <= 1.0 and 0.0 <= self.audio_weight <= 1.0):
            raise ValueError(
                f"Weights must be in [0, 1], got image_weight={self.image_weight}, "
                f"audio_weight={self.audio_weight}"
            )
        total_w = round(self.image_weight + self.audio_weight, 4)
        if abs(total_w - 1.0) > 1e-3 and total_w != 0.0:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total_w}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

    def predict_pair(
        self,
        image_prob: float | None = None,
        audio_prob: float | None = None,
        image_logits: torch.Tensor | None = None,
        audio_logits: torch.Tensor | None = None,
    ) -> LateFusionResult:
        """
        Evaluate fusion prediction for a single image-audio observation.

        Parameters
        ----------
        image_prob:
            Optional scalar image threat probability in [0, 1].
        audio_prob:
            Optional scalar audio threat probability in [0, 1].
        image_logits:
            Optional raw image logits (used if image_prob is None).
        audio_logits:
            Optional raw audio logits (used if audio_prob is None).

        Returns
        -------
        LateFusionResult
            Prediction outcome, scores, and risk level.

        """
        # Resolve probabilities
        p_img: float | None = None
        if image_prob is not None:
            p_img = float(image_prob)
        elif image_logits is not None:
            tensor_prob = extract_image_threat_probability(
                image_logits.unsqueeze(0) if image_logits.ndim == 1 else image_logits,
                scaler=self.image_scaler,
            )
            p_img = float(tensor_prob.squeeze().item())

        p_aud: float | None = None
        if audio_prob is not None:
            p_aud = float(audio_prob)
        elif audio_logits is not None:
            tensor_prob = extract_audio_threat_probability(
                audio_logits.unsqueeze(0) if audio_logits.ndim == 1 else audio_logits,
                scaler=self.audio_scaler,
            )
            p_aud = float(tensor_prob.squeeze().item())

        image_present = p_img is not None
        audio_present = p_aud is not None

        val_img = p_img if image_present else 0.0
        val_aud = p_aud if audio_present else 0.0

        # Weighted Late Fusion Branch
        if image_present and audio_present:
            fused_score = (self.image_weight * val_img) + (self.audio_weight * val_aud)
            effective_threshold = self.threshold
        elif image_present and not audio_present:
            fused_score = val_img
            effective_threshold = min(self.threshold + self.conservative_tau_delta, self.tau_cap)
        elif not image_present and audio_present:
            fused_score = val_aud
            effective_threshold = min(self.threshold + self.conservative_tau_delta, self.tau_cap)
        else:
            fused_score = 0.0
            effective_threshold = self.threshold

        weighted_pred = int(fused_score >= effective_threshold)

        # Rule Head Branch
        rule_pred, risk_level = self.evaluate_rule_head(
            image_prob=val_img,
            audio_prob=val_aud,
            image_present=image_present,
            audio_present=audio_present,
        )

        return LateFusionResult(
            fused_score=float(fused_score),
            weighted_prediction=weighted_pred,
            rule_prediction=rule_pred,
            risk_level=risk_level,
            image_prob=float(val_img),
            audio_prob=float(val_aud),
            image_present=image_present,
            audio_present=audio_present,
            effective_threshold=float(effective_threshold),
        )

    def evaluate_rule_head(
        self,
        image_prob: float,
        audio_prob: float,
        image_present: bool,
        audio_present: bool,
    ) -> tuple[int, str]:
        """
        Evaluate the 3-tier rule head on probabilities.

        Rules:
        - Clean input:
          * HIGH: P_img >= tau_img AND P_aud >= tau_aud  -> pred = 1
          * MEDIUM: exactly one clears threshold        -> pred = 0
          * LOW: neither clears threshold                -> pred = 0
        - Missing modality:
          * One sensor cannot reach HIGH (binary alert stays 0).
          * If available sensor clears threshold: MEDIUM, pred = 0.
          * If available sensor does not clear: LOW, pred = 0.

        Parameters
        ----------
        image_prob:
            Image threat probability.
        audio_prob:
            Audio gunshot probability.
        image_present:
            Whether image sensor is available.
        audio_present:
            Whether audio sensor is available.

        Returns
        -------
        tuple[int, str]
            (binary_prediction, risk_tier)

        """
        if not image_present and not audio_present:
            return 0, "LOW"

        img_pass = image_present and (image_prob >= self.rule_img_threshold)
        aud_pass = audio_present and (audio_prob >= self.rule_aud_threshold)

        if image_present and audio_present:
            if img_pass and aud_pass:
                return 1, "HIGH"
            if img_pass or aud_pass:
                return 0, "MEDIUM"
            return 0, "LOW"

        # Missing modality case: single sensor cannot reach HIGH
        if img_pass or aud_pass:
            return 0, "MEDIUM"
        return 0, "LOW"

    def predict_batch(
        self,
        image_probs: torch.Tensor | np.ndarray | None,
        audio_probs: torch.Tensor | np.ndarray | None,
        image_present: torch.Tensor | np.ndarray | None = None,
        audio_present: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Vectorized batch prediction across multimodal samples.

        Parameters
        ----------
        image_probs:
            Array/tensor of shape (N,) with image threat probabilities.
        audio_probs:
            Array/tensor of shape (N,) with audio threat probabilities.
        image_present:
            Boolean array of shape (N,) indicating image availability.
        audio_present:
            Boolean array of shape (N,) indicating audio availability.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing 'fused_scores', 'predictions', 'rule_predictions',
            'effective_thresholds', and 'risk_levels'.

        """
        n_samples = 0
        if image_probs is not None:
            n_samples = len(image_probs)
        elif audio_probs is not None:
            n_samples = len(audio_probs)

        img_p = (
            np.asarray(image_probs, dtype=np.float32)
            if image_probs is not None
            else np.zeros(n_samples, dtype=np.float32)
        )
        aud_p = (
            np.asarray(audio_probs, dtype=np.float32)
            if audio_probs is not None
            else np.zeros(n_samples, dtype=np.float32)
        )

        if image_present is not None:
            img_pres = np.asarray(image_present, dtype=bool)
        elif image_probs is not None:
            img_pres = np.ones(n_samples, dtype=bool)
        else:
            img_pres = np.zeros(n_samples, dtype=bool)

        if audio_present is not None:
            aud_pres = np.asarray(audio_present, dtype=bool)
        elif audio_probs is not None:
            aud_pres = np.ones(n_samples, dtype=bool)
        else:
            aud_pres = np.zeros(n_samples, dtype=bool)

        fused_scores = np.zeros(n_samples, dtype=np.float32)
        effective_thresholds = np.full(n_samples, self.threshold, dtype=np.float32)
        conservative_thresh = min(self.threshold + self.conservative_tau_delta, self.tau_cap)

        # Condition masks
        both_pres = img_pres & aud_pres
        only_img = img_pres & (~aud_pres)
        only_aud = (~img_pres) & aud_pres

        fused_scores[both_pres] = (self.image_weight * img_p[both_pres]) + (
            self.audio_weight * aud_p[both_pres]
        )
        effective_thresholds[both_pres] = self.threshold

        fused_scores[only_img] = img_p[only_img]
        effective_thresholds[only_img] = conservative_thresh

        fused_scores[only_aud] = aud_p[only_aud]
        effective_thresholds[only_aud] = conservative_thresh

        weighted_preds = (fused_scores >= effective_thresholds).astype(int)

        # Vectorized rule head
        rule_preds = np.zeros(n_samples, dtype=int)
        risk_levels = np.full(n_samples, "LOW", dtype=object)

        img_clears = img_p >= self.rule_img_threshold
        aud_clears = aud_p >= self.rule_aud_threshold

        # High risk only when both present and both clear
        high_mask = both_pres & img_clears & aud_clears
        risk_levels[high_mask] = "HIGH"
        rule_preds[high_mask] = 1

        # Medium risk when exactly one clears (or single sensor clears in dropout)
        med_mask_clean = both_pres & (img_clears ^ aud_clears)
        med_mask_drop = (only_img & img_clears) | (only_aud & aud_clears)
        risk_levels[med_mask_clean | med_mask_drop] = "MEDIUM"

        return {
            "fused_scores": fused_scores,
            "weighted_predictions": weighted_preds,
            "rule_predictions": rule_preds,
            "effective_thresholds": effective_thresholds,
            "risk_levels": risk_levels,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize model configuration."""
        return {
            "image_weight": self.image_weight,
            "audio_weight": self.audio_weight,
            "threshold": self.threshold,
            "rule_img_threshold": self.rule_img_threshold,
            "rule_aud_threshold": self.rule_aud_threshold,
            "conservative_tau_delta": self.conservative_tau_delta,
            "tau_cap": self.tau_cap,
            "image_scaler": self.image_scaler.to_dict() if self.image_scaler else None,
            "audio_scaler": self.audio_scaler.to_dict() if self.audio_scaler else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LateFusionModel:
        """Construct model from dictionary."""
        img_s = (
            TemperatureScaler.from_dict(data["image_scaler"])
            if data.get("image_scaler")
            else None
        )
        aud_s = (
            TemperatureScaler.from_dict(data["audio_scaler"])
            if data.get("audio_scaler")
            else None
        )
        return cls(
            image_weight=float(data.get("image_weight", 0.6)),
            audio_weight=float(data.get("audio_weight", 0.4)),
            threshold=float(data.get("threshold", 0.5)),
            rule_img_threshold=float(data.get("rule_img_threshold", 0.5)),
            rule_aud_threshold=float(data.get("rule_aud_threshold", 0.5)),
            conservative_tau_delta=float(data.get("conservative_tau_delta", 0.15)),
            tau_cap=float(data.get("tau_cap", 0.95)),
            image_scaler=img_s,
            audio_scaler=aud_s,
        )


def fit_late_fusion_parameters(
    val_image_probs: torch.Tensor | np.ndarray,
    val_audio_probs: torch.Tensor | np.ndarray,
    val_targets: torch.Tensor | np.ndarray,
    weight_step: float = 0.05,
    threshold_step: float = 0.05,
) -> tuple[float, float, float, dict[str, Any]]:
    """
    Grid-search optimal fusion weights and decision threshold on validation data.

    Criteria:
    - Maximize validation F1 score.
    - Tie-breaker: prefer lower False Positive Rate (FPR = FP / (FP + TN)).
    - Uses validation split strictly (never test data).

    Parameters
    ----------
    val_image_probs:
        Validation image threat probabilities.
    val_audio_probs:
        Validation audio threat probabilities.
    val_targets:
        Validation binary targets (0 or 1).
    weight_step:
        Grid search step size for image_weight in [0, 1].
    threshold_step:
        Grid search step size for threshold in (0, 1).

    Returns
    -------
    tuple[float, float, float, dict[str, Any]]
        (best_image_weight, best_audio_weight, best_threshold, summary_dict)

    """
    p_img = np.asarray(val_image_probs, dtype=np.float32)
    p_aud = np.asarray(val_audio_probs, dtype=np.float32)
    y_true = np.asarray(val_targets, dtype=int)

    n_samples = len(y_true)
    if n_samples == 0 or len(p_img) != n_samples or len(p_aud) != n_samples:
        raise ValueError("Inputs must have identical non-zero lengths.")

    unique_targets = np.unique(y_true)
    if len(unique_targets) < 2:
        logger.warning(
            "Validation targets contain only one class (%s). Grid search cannot optimize F1. "
            "Returning default parameters.",
            unique_targets.tolist(),
        )
        return 0.6, 0.4, 0.5, {
            "status": "NOT YET RUN",
            "reason": (
                "Single class present in validation split targets; "
                "F1 optimization undefined."
            ),
        }

    # Weight steps from 0.0 to 1.0 (step 0.05)
    weights = np.round(np.arange(0.0, 1.0 + (weight_step / 2), weight_step), 2)
    # Threshold steps from threshold_step to 1.0 - threshold_step
    thresholds = np.round(np.arange(threshold_step, 1.0, threshold_step), 2)

    best_f1 = -1.0
    best_fpr = 1.0
    best_w = 0.6
    best_tau = 0.5

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    for w in weights:
        w_aud = round(1.0 - w, 2)
        fused = (w * p_img) + (w_aud * p_aud)

        for tau in thresholds:
            y_pred = (fused >= tau).astype(int)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            fpr = float(safe_divide(fp, fp + tn) or 0.0)

            # Maximize F1; break ties by lower FPR
            if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-6 and fpr < best_fpr):
                best_f1 = f1
                best_fpr = fpr
                best_w = float(w)
                best_tau = float(tau)

    summary = {
        "status": "COMPLETE",
        "best_image_weight": best_w,
        "best_audio_weight": round(1.0 - best_w, 2),
        "best_threshold": best_tau,
        "best_validation_f1": float(best_f1),
        "validation_fpr": float(best_fpr),
        "n_samples": n_samples,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

    logger.info(
        "Late fusion grid search completed: w_img=%.2f, w_aud=%.2f, tau=%.2f -> "
        "Val F1=%.4f, FPR=%.4f",
        best_w,
        round(1.0 - best_w, 2),
        best_tau,
        best_f1,
        best_fpr,
    )

    return best_w, round(1.0 - best_w, 2), best_tau, summary


def fit_rule_thresholds(
    val_image_probs: torch.Tensor | np.ndarray,
    val_audio_probs: torch.Tensor | np.ndarray,
    val_image_targets: torch.Tensor | np.ndarray,
    val_audio_targets: torch.Tensor | np.ndarray,
    threshold_step: float = 0.05,
) -> tuple[float, float, dict[str, Any]]:
    """
    Select unimodal rule-head thresholds maximizing individual branch validation F1.

    Parameters
    ----------
    val_image_probs:
        Validation image threat probabilities P(human) + P(vehicle).
    val_audio_probs:
        Validation audio threat probabilities P(gunshot).
    val_image_targets:
        Binary target for image threat (1 if human/vehicle, 0 otherwise).
    val_audio_targets:
        Binary target for gunshot (1 if gunshot, 0 otherwise).
    threshold_step:
        Grid step for thresholds.

    Returns
    -------
    tuple[float, float, dict[str, Any]]
        (best_img_threshold, best_aud_threshold, summary_dict)

    """
    def _fit_single_branch(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
        unique = np.unique(targets)
        if len(unique) < 2:
            return 0.5, 0.0
        thresholds = np.round(np.arange(threshold_step, 1.0, threshold_step), 2)
        best_t = 0.5
        best_f1 = -1.0
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
        return best_t, float(best_f1)

    p_img = np.asarray(val_image_probs, dtype=np.float32)
    p_aud = np.asarray(val_audio_probs, dtype=np.float32)
    y_img = np.asarray(val_image_targets, dtype=int)
    y_aud = np.asarray(val_audio_targets, dtype=int)

    tau_img, f1_img = _fit_single_branch(p_img, y_img)
    tau_aud, f1_aud = _fit_single_branch(p_aud, y_aud)

    summary = {
        "status": "COMPLETE",
        "rule_img_threshold": tau_img,
        "rule_img_f1": f1_img,
        "rule_aud_threshold": tau_aud,
        "rule_aud_f1": f1_aud,
    }
    return tau_img, tau_aud, summary
