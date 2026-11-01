"""
Unit Tests for Analytics Agent (src/analytics/).
Compliance Notice: Research use only, not for clinical decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analytics.analyzer import analyze_ecg
from analytics.explainability import WaveformExplainer
from analytics.risk_scorer import RiskScorer
from pipeline.schemas import AnalyticsResult, MANDATORY_DISCLAIMER, RiskTier


def test_risk_scorer_calibration():
    scorer = RiskScorer()
    # Test scaled risk below and above threshold
    sub_risk = scorer.compute_scaled_risk(probability=0.15, threshold=0.30)
    assert sub_risk < 50.0

    supra_risk = scorer.compute_scaled_risk(probability=0.45, threshold=0.30)
    assert supra_risk > 50.0

    # Test confidence interval
    ci_lower, ci_upper = scorer.calculate_confidence_interval(0.40)
    assert 0.0 <= ci_lower < 0.40 < ci_upper <= 1.0


def test_multi_morbidity_synergism():
    scorer = RiskScorer()
    probs_single = {"NORM": 0.2, "MI": 0.6, "STTC": 0.1, "CD": 0.1, "HYP": 0.05}
    thresh = {"NORM": 0.5, "MI": 0.3, "STTC": 0.3, "CD": 0.35, "HYP": 0.25}

    details_single = scorer.evaluate_condition_risks(probs_single, thresh)
    score_single, tier_single, _ = scorer.compute_composite_risk(details_single)

    # Co-occurrence of MI + CD
    probs_multi = {"NORM": 0.1, "MI": 0.6, "STTC": 0.1, "CD": 0.6, "HYP": 0.05}
    details_multi = scorer.evaluate_condition_risks(probs_multi, thresh)
    score_multi, tier_multi, notes = scorer.compute_composite_risk(details_multi)

    assert score_multi > score_single
    assert any("Multi-morbidity" in n for n in notes)


def test_waveform_explainer():
    signal = np.sin(np.linspace(0, 10 * np.pi, 5000)).astype(np.float32)
    explainer = WaveformExplainer(n_steps=10)
    summary = explainer.explain(signal, target_condition="MI")

    assert summary.saliency_energy > 0.0
    assert len(summary.segment_attributions) > 0
    assert MANDATORY_DISCLAIMER in summary.disclaimer


def test_full_analyze_ecg():
    signal = np.zeros(5000, dtype=np.float32)
    result = analyze_ecg(signal, include_explainability=True)

    assert isinstance(result, AnalyticsResult)
    assert 0.0 <= result.composite_risk_score <= 100.0
    assert isinstance(result.composite_risk_tier, RiskTier)
    assert MANDATORY_DISCLAIMER in result.disclaimer
