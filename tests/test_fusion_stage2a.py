"""
Unit tests for GuardianEye Phase 4 Stage 2A: Late Fusion, Calibration, and Baselines.

Tests:
- Temperature scaling calibration, positive temperature constraint, and NLL minimization.
- Honest handling of single-class or unpopulated validation data (NOT YET RUN).
- Weighted late-fusion scoring and grid search on validation data.
- High/Medium/Low rule head behavior under clean and missing modality conditions.
- Conservative threshold adjustment (tau_delta capped at tau_cap) on sensor dropout.
- Image-only and audio-only unimodal baselines with strict missing-modality isolation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fusion.baselines import AudioOnlyBaseline, ImageOnlyBaseline
from src.fusion.calibrate import TemperatureScaler
from src.fusion.late import (
    LateFusionModel,
    extract_audio_threat_probability,
    extract_image_threat_probability,
    fit_late_fusion_parameters,
    fit_rule_thresholds,
)

# ---------------------------------------------------------------------------
# Calibration Tests
# ---------------------------------------------------------------------------


class TestTemperatureScaler:
    """Tests for post-hoc logit calibration using TemperatureScaler."""

    def test_positive_temperature_initialization(self) -> None:
        scaler = TemperatureScaler(initial_temperature=1.5)
        assert scaler.temperature > 0.0
        assert pytest.approx(scaler.temperature, rel=1e-3) == 1.5

        with pytest.raises(ValueError, match="initial_temperature must be > 0"):
            TemperatureScaler(initial_temperature=-0.5)

        with pytest.raises(ValueError, match="initial_temperature must be > 0"):
            TemperatureScaler(initial_temperature=0.0)

    def test_temperature_forward_scaling(self) -> None:
        scaler = TemperatureScaler(initial_temperature=2.0)
        logits = torch.tensor([[2.0, -2.0], [4.0, 0.0]])
        scaled = scaler(logits)

        # Logits should be divided by 2.0
        expected = torch.tensor([[1.0, -1.0], [2.0, 0.0]])
        assert torch.allclose(scaled, expected)

    def test_predict_proba_sums_to_one(self) -> None:
        scaler = TemperatureScaler(initial_temperature=1.2)
        logits = torch.randn(10, 4)
        probs = scaler.predict_proba(logits)

        assert probs.shape == (10, 4)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_fit_on_synthetic_overconfident_logits(self) -> None:
        # Create artificially overconfident logits on 2 classes
        torch.manual_seed(42)
        n = 200
        labels = torch.randint(0, 2, (n,))
        # Scaled up logits (overconfident)
        logits = torch.zeros(n, 2)
        for i in range(n):
            if labels[i] == 1:
                logits[i] = torch.tensor([-5.0, 5.0])
            else:
                logits[i] = torch.tensor([5.0, -5.0])

        # Add noise to induce miscalibration
        noise = torch.randn(n, 2) * 1.5
        noisy_logits = logits + noise

        scaler = TemperatureScaler(initial_temperature=1.0)
        result = scaler.fit(noisy_logits, labels, max_iter=50)

        assert result.status == "COMPLETE"
        assert scaler.temperature > 0.0
        assert result.nll_after is not None and result.nll_before is not None
        assert result.nll_after <= result.nll_before + 1e-4

    def test_insufficient_or_single_class_validation_data_reports_not_run(self) -> None:
        scaler = TemperatureScaler()

        # 0 samples
        res_empty = scaler.fit(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long))
        assert res_empty.status == "NOT YET RUN"
        assert res_empty.temperature == 1.0

        # Only one class present (all class 0)
        logits_single = torch.randn(20, 2)
        labels_single = torch.zeros(20, dtype=torch.long)
        res_single = scaler.fit(logits_single, labels_single)
        assert res_single.status == "NOT YET RUN"
        assert "Validation split lacks multiple classes" in str(res_single.reason)
        assert res_single.temperature == 1.0

    def test_serialization(self) -> None:
        scaler = TemperatureScaler(initial_temperature=1.85)
        d = scaler.to_dict()
        assert pytest.approx(d["temperature"], rel=1e-3) == 1.85

        reloaded = TemperatureScaler.from_dict(d)
        assert pytest.approx(reloaded.temperature, rel=1e-3) == 1.85


# ---------------------------------------------------------------------------
# Threat Probability Extraction Tests
# ---------------------------------------------------------------------------


class TestThreatExtraction:
    """Verify conversion of branch logits to calibrated threat probabilities."""

    def test_extract_image_threat_probability(self) -> None:
        # Classes: 0: human, 1: vehicle, 2: animal, 3: empty
        # Case A: High probability on human (class 0)
        logits_human = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        prob_human = extract_image_threat_probability(logits_human)
        assert prob_human.item() > 0.99

        # Case B: High probability on vehicle (class 1)
        logits_vehicle = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
        prob_vehicle = extract_image_threat_probability(logits_vehicle)
        assert prob_vehicle.item() > 0.99

        # Case C: High probability on animal (class 2) -> threat should be ~0
        logits_animal = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
        prob_animal = extract_image_threat_probability(logits_animal)
        assert prob_animal.item() < 0.01

    def test_extract_audio_threat_probability(self) -> None:
        # Classes: 0: background, 1: gunshot
        logits_gunshot = torch.tensor([[0.0, 8.0]])
        prob_gunshot = extract_audio_threat_probability(logits_gunshot)
        assert prob_gunshot.item() > 0.99

        logits_bg = torch.tensor([[8.0, 0.0]])
        prob_bg = extract_audio_threat_probability(logits_bg)
        assert prob_bg.item() < 0.01


# ---------------------------------------------------------------------------
# Late Fusion Tests
# ---------------------------------------------------------------------------


class TestLateFusion:
    """Verify weighted score, rule head, parameter fitting, and missing modalities."""

    def test_clean_weighted_score(self) -> None:
        model = LateFusionModel(
            image_weight=0.6,
            audio_weight=0.4,
            threshold=0.5,
        )
        # P_img = 0.8, P_aud = 0.5 -> Score = 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.20 = 0.68
        res = model.predict_pair(image_prob=0.8, audio_prob=0.5)
        assert pytest.approx(res.fused_score, rel=1e-3) == 0.68
        assert res.weighted_prediction == 1
        assert res.effective_threshold == 0.5

    def test_rule_head_three_tiers_clean(self) -> None:
        model = LateFusionModel(rule_img_threshold=0.5, rule_aud_threshold=0.5)

        # Both clear threshold -> HIGH, pred = 1
        res_high = model.predict_pair(image_prob=0.7, audio_prob=0.8)
        assert res_high.risk_level == "HIGH"
        assert res_high.rule_prediction == 1

        # Only image clears -> MEDIUM, pred = 0
        res_med_img = model.predict_pair(image_prob=0.7, audio_prob=0.2)
        assert res_med_img.risk_level == "MEDIUM"
        assert res_med_img.rule_prediction == 0

        # Only audio clears -> MEDIUM, pred = 0
        res_med_aud = model.predict_pair(image_prob=0.3, audio_prob=0.8)
        assert res_med_aud.risk_level == "MEDIUM"
        assert res_med_aud.rule_prediction == 0

        # Neither clears -> LOW, pred = 0
        res_low = model.predict_pair(image_prob=0.2, audio_prob=0.2)
        assert res_low.risk_level == "LOW"
        assert res_low.rule_prediction == 0

    def test_missing_modality_conservative_threshold_adjustment(self) -> None:
        # Base tau = 0.50, delta = 0.15, tau_cap = 0.95 -> effective tau = 0.65
        model = LateFusionModel(
            threshold=0.50,
            conservative_tau_delta=0.15,
            tau_cap=0.95,
        )

        # Audio dropped: score should be exactly image_prob
        res_aud_drop = model.predict_pair(image_prob=0.60, audio_prob=None)
        assert res_aud_drop.image_present is True
        assert res_aud_drop.audio_present is False
        assert pytest.approx(res_aud_drop.fused_score, rel=1e-3) == 0.60
        assert pytest.approx(res_aud_drop.effective_threshold, rel=1e-3) == 0.65
        # 0.60 < 0.65 -> prediction is 0
        assert res_aud_drop.weighted_prediction == 0

        # Audio dropped with high image prob above conservative threshold
        res_aud_drop_high = model.predict_pair(image_prob=0.75, audio_prob=None)
        assert res_aud_drop_high.weighted_prediction == 1

        # Image dropped: score should be exactly audio_prob
        res_img_drop = model.predict_pair(image_prob=None, audio_prob=0.70)
        assert res_img_drop.image_present is False
        assert res_img_drop.audio_present is True
        assert pytest.approx(res_img_drop.fused_score, rel=1e-3) == 0.70
        assert pytest.approx(res_img_drop.effective_threshold, rel=1e-3) == 0.65
        assert res_img_drop.weighted_prediction == 1

    def test_missing_modality_rule_head_cannot_reach_high(self) -> None:
        model = LateFusionModel(rule_img_threshold=0.5, rule_aud_threshold=0.5)

        # Single sensor available and clears threshold -> MEDIUM, never HIGH
        res_img_only = model.predict_pair(image_prob=0.99, audio_prob=None)
        assert res_img_only.risk_level == "MEDIUM"
        assert res_img_only.rule_prediction == 0  # Binary alert positive only on HIGH

        res_aud_only = model.predict_pair(image_prob=None, audio_prob=0.99)
        assert res_aud_only.risk_level == "MEDIUM"
        assert res_aud_only.rule_prediction == 0

        # Single sensor available but does not clear -> LOW
        res_img_low = model.predict_pair(image_prob=0.20, audio_prob=None)
        assert res_img_low.risk_level == "LOW"
        assert res_img_low.rule_prediction == 0

    def test_both_modalities_dropped(self) -> None:
        model = LateFusionModel()
        res = model.predict_pair(image_prob=None, audio_prob=None)
        assert res.fused_score == 0.0
        assert res.weighted_prediction == 0
        assert res.rule_prediction == 0
        assert res.risk_level == "LOW"

    def test_tau_cap_enforced(self) -> None:
        # threshold = 0.90, delta = 0.15, cap = 0.95 -> min(1.05, 0.95) = 0.95
        model = LateFusionModel(threshold=0.90, conservative_tau_delta=0.15, tau_cap=0.95)
        res = model.predict_pair(image_prob=0.92, audio_prob=None)
        assert pytest.approx(res.effective_threshold, rel=1e-3) == 0.95
        assert res.weighted_prediction == 0

    def test_validation_grid_search_optimizes_f1(self) -> None:
        # Construct synthetic validation set with clear separation
        np.random.seed(42)
        n = 100
        y_val = np.random.randint(0, 2, n)
        # Positive samples have higher image and audio probs
        p_img = np.where(y_val == 1, np.random.uniform(0.6, 0.9, n), np.random.uniform(0.1, 0.4, n))
        p_aud = np.where(y_val == 1, np.random.uniform(0.5, 0.9, n), np.random.uniform(0.0, 0.3, n))

        w_img, w_aud, tau, summary = fit_late_fusion_parameters(
            val_image_probs=p_img,
            val_audio_probs=p_aud,
            val_targets=y_val,
            weight_step=0.1,
            threshold_step=0.1,
        )

        assert summary["status"] == "COMPLETE"
        assert 0.0 <= w_img <= 1.0
        assert pytest.approx(w_img + w_aud, rel=1e-3) == 1.0
        assert 0.0 < tau < 1.0
        assert summary["best_validation_f1"] > 0.85

    def test_grid_search_single_class_validation_handles_honestly(self) -> None:
        # All validation targets are 0
        p_img = np.random.uniform(0.1, 0.4, 50)
        p_aud = np.random.uniform(0.1, 0.4, 50)
        y_val = np.zeros(50, dtype=int)

        w_img, w_aud, tau, summary = fit_late_fusion_parameters(
            val_image_probs=p_img,
            val_audio_probs=p_aud,
            val_targets=y_val,
        )
        assert summary["status"] == "NOT YET RUN"
        assert "Single class present in validation split targets" in summary["reason"]

    def test_fit_rule_thresholds(self) -> None:
        y_img = np.array([1, 1, 0, 0, 1, 0])
        p_img = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])

        y_aud = np.array([1, 0, 1, 0, 0, 0])
        p_aud = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.3])

        t_img, t_aud, summary = fit_rule_thresholds(p_img, p_aud, y_img, y_aud)
        assert summary["status"] == "COMPLETE"
        assert 0.3 <= t_img <= 0.8
        assert 0.3 <= t_aud <= 0.8

    def test_predict_batch_vectorized(self) -> None:
        model = LateFusionModel(image_weight=0.6, audio_weight=0.4, threshold=0.5)

        img_p = np.array([0.8, 0.2, 0.9, 0.1])
        aud_p = np.array([0.7, 0.8, 0.1, 0.2])
        img_pres = np.array([True, True, True, False])
        aud_pres = np.array([True, True, False, True])

        batch_res = model.predict_batch(
            image_probs=img_p,
            audio_probs=aud_p,
            image_present=img_pres,
            audio_present=aud_pres,
        )

        assert len(batch_res["fused_scores"]) == 4
        assert len(batch_res["weighted_predictions"]) == 4
        assert len(batch_res["risk_levels"]) == 4

        # Compare sample 0 with single predict_pair
        single_0 = model.predict_pair(image_prob=0.8, audio_prob=0.7)
        assert pytest.approx(batch_res["fused_scores"][0], rel=1e-3) == single_0.fused_score
        assert batch_res["weighted_predictions"][0] == single_0.weighted_prediction
        assert batch_res["risk_levels"][0] == single_0.risk_level

    def test_late_fusion_model_serialization(self) -> None:
        model = LateFusionModel(
            image_weight=0.7,
            audio_weight=0.3,
            threshold=0.55,
            rule_img_threshold=0.6,
            rule_aud_threshold=0.65,
        )
        d = model.to_dict()
        reloaded = LateFusionModel.from_dict(d)

        assert reloaded.image_weight == 0.7
        assert reloaded.audio_weight == 0.3
        assert reloaded.threshold == 0.55
        assert reloaded.rule_img_threshold == 0.6
        assert reloaded.rule_aud_threshold == 0.65


# ---------------------------------------------------------------------------
# Unimodal Baselines Tests
# ---------------------------------------------------------------------------


class TestUnimodalBaselines:
    """Verify single-modality baselines and strict missing-modality isolation."""

    def test_image_baseline_clean_prediction(self) -> None:
        baseline = ImageOnlyBaseline(threshold=0.5)
        res_high = baseline.predict(image_prob=0.75, image_present=True)
        assert res_high.score == 0.75
        assert res_high.prediction == 1
        assert res_high.modality_present is True

        res_low = baseline.predict(image_prob=0.30, image_present=True)
        assert res_low.score == 0.30
        assert res_low.prediction == 0

    def test_image_baseline_missing_modality_predicts_negative_without_borrowing(self) -> None:
        baseline = ImageOnlyBaseline(threshold=0.5)
        # Even if a caller passed an image prob, image_present=False forces score 0.0
        res = baseline.predict(image_prob=0.99, image_present=False)
        assert res.score == 0.0
        assert res.prediction == 0
        assert res.modality_present is False

        # None input
        res_none = baseline.predict(image_prob=None, image_present=True)
        assert res_none.score == 0.0
        assert res_none.prediction == 0
        assert res_none.modality_present is False

    def test_audio_baseline_clean_prediction(self) -> None:
        baseline = AudioOnlyBaseline(threshold=0.6)
        res_high = baseline.predict(audio_prob=0.85, audio_present=True)
        assert res_high.score == 0.85
        assert res_high.prediction == 1
        assert res_high.modality_present is True

        res_low = baseline.predict(audio_prob=0.40, audio_present=True)
        assert res_low.score == 0.40
        assert res_low.prediction == 0

    def test_audio_baseline_missing_modality_predicts_negative_without_borrowing(self) -> None:
        baseline = AudioOnlyBaseline(threshold=0.6)
        res = baseline.predict(audio_prob=0.95, audio_present=False)
        assert res.score == 0.0
        assert res.prediction == 0
        assert res.modality_present is False

        res_none = baseline.predict(audio_prob=None, audio_present=True)
        assert res_none.score == 0.0
        assert res_none.prediction == 0
        assert res_none.modality_present is False

    def test_baseline_threshold_fitting(self) -> None:
        val_probs = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
        val_targets = np.array([1, 1, 0, 0, 1, 0])

        base_img = ImageOnlyBaseline(threshold=0.5)
        fit_t = base_img.fit_threshold(val_probs, val_targets)
        assert 0.3 <= fit_t <= 0.8
        assert base_img.threshold == fit_t

    def test_baseline_batch_evaluation(self) -> None:
        baseline = ImageOnlyBaseline(threshold=0.5)
        probs = np.array([0.8, 0.2, 0.9, 0.6])
        pres = np.array([True, True, False, True])

        batch = baseline.predict_batch(probs, image_present=pres)
        expected_scores = np.array([0.8, 0.2, 0.0, 0.6])
        expected_preds = np.array([1, 0, 0, 1])

        assert np.allclose(batch["scores"], expected_scores)
        assert np.array_equal(batch["predictions"], expected_preds)
