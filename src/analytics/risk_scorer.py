"""
Cardiovascular Risk Scoring and Multi-Morbidity Stratification Engine.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Provides calibrated risk scores, confidence intervals, risk tiers,
and composite cardiovascular indices from ECG prediction probabilities.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pathlib import Path

from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ConditionRiskDetail,
    RiskTier,
)

# Standard default class weighting for composite cardiac risk calculation
DEFAULT_WEIGHTS = {
    "MI": 0.35,     # Myocardial infarction (acute mortality / necrosis)
    "STTC": 0.25,   # ST-T changes (ischemia / repolarization instability)
    "CD": 0.20,     # Conduction disturbance (AV block / dysrhythmia)
    "HYP": 0.20,    # Hypertrophy (structural overload / heart failure)
}

# Clinical descriptors and contributing mechanisms per condition
PATHOLOGY_CORRELATES = {
    "MI": {
        "label": "Myocardial Infarction",
        "description": "Coronary artery occlusion resulting in myocardial ischemic injury or transmural necrosis.",
        "high_risk_factors": [
            "Loss of anterior/lateral R-wave progression",
            "Pathological Q-waves or acute terminal ST-elevation vector",
            "Heightened vulnerability to post-infarction ventricular arrhythmias",
        ],
    },
    "STTC": {
        "label": "ST-T Segment Abnormality",
        "description": "Repolarization irregularity secondary to subendocardial ischemia, strain, or drug/electrolyte effects.",
        "high_risk_factors": [
            "ST-segment horizontal/downsloping depression or symmetric T-wave inversion",
            "Impaired potassium channel repolarization reserve (IKr/hERG vulnerability)",
            "Underlying ischemic coronary insufficiency or myocardial strain",
        ],
    },
    "CD": {
        "label": "Conduction Disturbance",
        "description": "Impaired ventricular depolarization propagation across bundle branches or fascicles.",
        "high_risk_factors": [
            "Prolonged QRS duration (>120 ms) indicating bundle branch conduction delay",
            "Loss of synchronous bi-ventricular contraction mechanics",
            "Elevated risk for complete atrioventricular dissociation or fatal heart block",
        ],
    },
    "HYP": {
        "label": "Ventricular Hypertrophy",
        "description": "Myocardial mass enlargement resulting from chronic hemodynamic afterload or cardiomyopathy.",
        "high_risk_factors": [
            "Augmented QRS voltage reflecting hypertrophied ventricular myocardium",
            "Secondary ventricular repolarization strain patterns",
            "Elevated progression rate toward diastolic dysfunction and congestive heart failure",
        ],
    },
}


class RiskScorer:
    """
    Calibrates raw neural network probabilities into clinically informative
    risk scores (0-100), confidence intervals, and composite risk tiers.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        effective_sample_size: int = 500,
    ):
        self.weights = weights or DEFAULT_WEIGHTS
        self.effective_n = effective_sample_size

    def calculate_confidence_interval(
        self,
        probability: float,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute Wilson score 95% confidence interval for predicted probability.
        """
        p = max(1e-4, min(1.0 - 1e-4, probability))
        n = self.effective_n
        z = 1.96 if confidence_level == 0.95 else 1.645  # standard normal quantile

        denominator = 1.0 + (z**2) / n
        center = (p + (z**2) / (2.0 * n)) / denominator
        margin = (z / denominator) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))

        ci_lower = max(0.0, center - margin)
        ci_upper = min(1.0, center + margin)
        return (round(ci_lower, 4), round(ci_upper, 4))

    def compute_scaled_risk(
        self,
        probability: float,
        threshold: float,
    ) -> float:
        """
        Map probability to a normalized 0-100 risk score centered at operating threshold.
        
        If probability equals threshold -> risk = 50.0.
        If probability < threshold -> smooth sub-threshold scale (0 - 50).
        If probability > threshold -> steep supra-threshold scale (50 - 100).
        """
        p = max(0.0, min(1.0, probability))
        t = max(0.05, min(0.95, threshold))

        if p <= t:
            # Scale linearly or quadratic from 0 to 50
            scaled = 50.0 * (p / t) ** 1.2
        else:
            # Scale from 50 to 100
            excess = (p - t) / (1.0 - t)
            scaled = 50.0 + 50.0 * (excess ** 0.8)

        return round(max(0.0, min(100.0, scaled)), 2)

    def determine_tier(self, risk_score: float) -> RiskTier:
        """Map 0-100 risk score to standardized RiskTier enum."""
        if risk_score < 25.0:
            return RiskTier.LOW
        elif risk_score < 50.0:
            return RiskTier.MODERATE
        elif risk_score < 75.0:
            return RiskTier.HIGH
        else:
            return RiskTier.CRITICAL

    def evaluate_condition_risks(
        self,
        probabilities: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> Dict[str, ConditionRiskDetail]:
        """
        Compute individual ConditionRiskDetail records for each evaluated condition.
        """
        details: Dict[str, ConditionRiskDetail] = {}

        for cond, prob in probabilities.items():
            if cond == "NORM":
                # NORM is healthy baseline; lower prob indicates higher abnormality
                thresh = thresholds.get(cond, 0.50)
                norm_risk = round(max(0.0, min(100.0, (1.0 - prob) * 100.0)), 2)
                tier = self.determine_tier(norm_risk)
                ci = self.calculate_confidence_interval(prob)
                factors = ["Baseline physiological sinus pattern" if prob >= thresh else "Loss of normal sinus baseline morphology"]
                details[cond] = ConditionRiskDetail(
                    condition=cond,
                    probability=prob,
                    risk_score=norm_risk,
                    risk_tier=tier,
                    confidence_interval=ci,
                    contributing_factors=factors,
                    disclaimer=MANDATORY_DISCLAIMER,
                )
            else:
                thresh = thresholds.get(cond, 0.30)
                risk_score = self.compute_scaled_risk(prob, thresh)
                tier = self.determine_tier(risk_score)
                ci = self.calculate_confidence_interval(prob)

                factors: List[str] = []
                meta = PATHOLOGY_CORRELATES.get(cond, {})
                if prob >= thresh:
                    factors.extend(meta.get("high_risk_factors", []))
                else:
                    factors.append(f"Sub-threshold evidence for {meta.get('label', cond)} (prob: {prob:.3f} < thresh: {thresh:.2f})")

                details[cond] = ConditionRiskDetail(
                    condition=cond,
                    probability=prob,
                    risk_score=risk_score,
                    risk_tier=tier,
                    confidence_interval=ci,
                    contributing_factors=factors,
                    disclaimer=MANDATORY_DISCLAIMER,
                )

        return details

    def compute_composite_risk(
        self,
        condition_risks: Dict[str, ConditionRiskDetail],
    ) -> Tuple[float, RiskTier, List[str]]:
        """
        Compute integrated composite cardiovascular risk index across pathologies.
        Applies synergistic penalty for multi-morbidity (e.g. concurrent MI + CD).
        """
        pathology_keys = ["MI", "STTC", "CD", "HYP"]
        total_weight = sum(self.weights.get(k, 0.25) for k in pathology_keys)

        weighted_sum = 0.0
        active_pathologies: List[str] = []

        for k in pathology_keys:
            if k in condition_risks:
                detail = condition_risks[k]
                w = self.weights.get(k, 0.25) / total_weight
                weighted_sum += detail.risk_score * w
                if detail.risk_score >= 50.0:  # Exceeded threshold
                    active_pathologies.append(k)

        # Multi-morbidity synergism penalty
        # Concurrent active conditions exponentially escalate risk
        if len(active_pathologies) >= 2:
            synergy_boost = 10.0 * (len(active_pathologies) - 1)
            weighted_sum = min(100.0, weighted_sum + synergy_boost)

        composite_score = round(max(0.0, min(100.0, weighted_sum)), 2)
        composite_tier = self.determine_tier(composite_score)

        # Summary notes
        notes: List[str] = []
        if len(active_pathologies) > 0:
            notes.append(f"Elevated risk flags detected: {', '.join(active_pathologies)}")
            if len(active_pathologies) >= 2:
                notes.append(f"Multi-morbidity interaction detected: co-occurrence of {', '.join(active_pathologies)} elevates composite risk to {composite_tier.value}")
        else:
            notes.append("No pathological ECG conditions exceeded operating detection thresholds.")

        return composite_score, composite_tier, notes
