"""
Post-hoc probability calibration via temperature scaling for GuardianEye models.

Implements:
- TemperatureScaler: PyTorch module parameterizing temperature as T = exp(s) > 0.
- Optimization of temperature on validation split logits using Negative Log-Likelihood.
- Explicit handling of insufficient or single-class validation data with NOT YET RUN status.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome and metadata from a temperature calibration fit."""

    temperature: float
    status: str  # "COMPLETE" or "NOT YET RUN"
    reason: str | None = None
    nll_before: float | None = None
    nll_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)


class TemperatureScaler(nn.Module):
    """
    Temperature scaling module for post-hoc logit calibration.

    Computes:
        logits_scaled = logits / temperature
        temperature = exp(log_temperature) > 0

    Parameters are optimized on validation logits strictly to prevent test leakage.
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError(f"initial_temperature must be > 0, got {initial_temperature}")
        init_log_temp = float(torch.log(torch.tensor(initial_temperature, dtype=torch.float32)))
        self.log_temperature = nn.Parameter(torch.tensor([init_log_temp], dtype=torch.float32))

    @property
    def temperature(self) -> float:
        """Return the strictly positive temperature scalar."""
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale raw logits by the learned positive temperature.

        Parameters
        ----------
        logits:
            Tensor of unscaled prediction logits with shape (..., num_classes).

        Returns
        -------
        torch.Tensor
            Scaled logits of the same shape.

        """
        temp = torch.exp(self.log_temperature)
        return logits / temp

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute calibrated softmax probabilities from raw logits.

        Parameters
        ----------
        logits:
            Tensor of unscaled prediction logits.

        Returns
        -------
        torch.Tensor
            Calibrated probabilities summing to 1 across the last dimension.

        """
        scaled = self.forward(logits)
        return F.softmax(scaled, dim=-1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> CalibrationResult:
        """
        Fit temperature parameter using validation logits and labels.

        Guarantees:
        - Only validation data is used.
        - Temperature is strictly positive.
        - Returns NOT YET RUN status if validation data is empty or lacks both classes.

        Parameters
        ----------
        logits:
            Validation logits of shape (N, C).
        labels:
            Validation ground truth class indices of shape (N,).
        lr:
            Learning rate for optimizer.
        max_iter:
            Maximum optimization steps.

        Returns
        -------
        CalibrationResult
            Calibration outcome details.

        """
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                f"Expected logits of shape (N, C) and labels of shape (N,); "
                f"got logits.shape={logits.shape}, labels.shape={labels.shape}"
            )
        if len(logits) != len(labels):
            raise ValueError(
                f"Mismatched sample counts: len(logits)={len(logits)} != len(labels)={len(labels)}"
            )

        n_samples = len(labels)
        if n_samples == 0:
            logger.warning("Calibration received 0 validation samples. Reporting NOT YET RUN.")
            return CalibrationResult(
                temperature=1.0,
                status="NOT YET RUN",
                reason="Validation split contains 0 samples.",
            )

        unique_classes = torch.unique(labels)
        if len(unique_classes) < 2:
            logger.warning(
                "Validation split has fewer than 2 classes (%s observed). "
                "Calibration undefined; reporting NOT YET RUN.",
                unique_classes.tolist(),
            )
            return CalibrationResult(
                temperature=1.0,
                status="NOT YET RUN",
                reason=(
                    f"Validation split lacks multiple classes (only classes "
                    f"{unique_classes.tolist()} observed); calibration cannot be fit."
                ),
            )

        criterion = nn.CrossEntropyLoss()
        logits_detached = logits.detach()
        labels_detached = labels.detach().long()

        with torch.no_grad():
            init_loss = float(criterion(self.forward(logits_detached), labels_detached).item())

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits_detached)
            loss = criterion(scaled, labels_detached)
            loss.backward()
            return loss

        optimizer.step(_closure)

        with torch.no_grad():
            final_loss = float(criterion(self.forward(logits_detached), labels_detached).item())

        temp_val = self.temperature
        logger.info(
            "Temperature calibration fitted: T = %.4f (NLL: %.4f -> %.4f)",
            temp_val,
            init_loss,
            final_loss,
        )

        return CalibrationResult(
            temperature=temp_val,
            status="COMPLETE",
            nll_before=init_loss,
            nll_after=final_loss,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler state to dictionary."""
        return {
            "temperature": self.temperature,
            "log_temperature": float(self.log_temperature.item()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemperatureScaler:
        """Construct scaler from serialized state."""
        scaler = cls(initial_temperature=float(data.get("temperature", 1.0)))
        if "log_temperature" in data:
            with torch.no_grad():
                scaler.log_temperature.copy_(
                    torch.tensor([float(data["log_temperature"])], dtype=torch.float32)
                )
        return scaler
