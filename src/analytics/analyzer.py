"""
Master Analytics Module for ECG-AI-Research-1Lead.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Combines predictive multi-morbidity risk scoring and Integrated Gradients
explainability into a unified AnalyticsResult container.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import yaml

# Ensure project root and src are importable
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ecg_api
from analytics.explainability import WaveformExplainer
from analytics.risk_scorer import RiskScorer
from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    AnalyticsResult,
    ECGInferenceResult,
    ExplainabilitySummary,
)

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "pipeline.yaml"


def _load_analytics_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else _CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("analytics", {})
    return {}


def analyze_ecg(
    ecg: Union[torch.Tensor, np.ndarray, Sequence[float]],
    inference_result: Optional[ECGInferenceResult] = None,
    config_path: Optional[Union[str, Path]] = None,
    target_condition: Optional[str] = None,
    include_explainability: bool = True,
) -> AnalyticsResult:
    """
    Perform comprehensive risk analysis and waveform explainability.

    RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.

    Parameters:
        ecg: 1-lead ECG signal (5000 samples).
        inference_result: Optional precomputed ECGInferenceResult to avoid redundant model passes.
        config_path: Optional path to override pipeline config yaml.
        target_condition: Specific condition to generate explainability attribution for.
        include_explainability: Whether to calculate Integrated Gradients attribution.

    Returns:
        AnalyticsResult: Pydantic schema with composite risk, condition risk tiers,
        CI bounds, top salient regions, segment attributions, and compliance disclaimer.
    """
    # 1. Obtain inference if not passed
    if inference_result is None:
        inference_result = ecg_api.infer(ecg, config_path=config_path)

    # 2. Load configuration
    cfg = _load_analytics_config(config_path)
    weights = cfg.get("risk_scoring", {}).get("weights")
    n_steps = cfg.get("explainability", {}).get("n_steps", 50)
    baseline_mode = cfg.get("explainability", {}).get("baseline", "zero")

    # 3. Calculate stratified condition risks and composite index
    scorer = RiskScorer(weights=weights)
    condition_risks = scorer.evaluate_condition_risks(
        probabilities=inference_result.probabilities,
        thresholds=inference_result.thresholds,
    )
    composite_score, composite_tier, summary_notes = scorer.compute_composite_risk(
        condition_risks=condition_risks
    )

    # 4. Compute explainability attributions if requested
    explainability: Optional[ExplainabilitySummary] = None
    if include_explainability:
        # Convert signal to tensor
        if isinstance(ecg, np.ndarray):
            signal_tensor = torch.from_numpy(ecg).float().squeeze()
        elif isinstance(ecg, torch.Tensor):
            signal_tensor = ecg.float().squeeze()
        else:
            signal_tensor = torch.tensor(ecg, dtype=torch.float32).squeeze()

        explainer = WaveformExplainer(
            sampling_rate=inference_result.sampling_rate,
            n_steps=n_steps,
            baseline_mode=baseline_mode,
        )
        explainability = explainer.explain(
            signal=signal_tensor,
            target_condition=target_condition,
            detected_conditions=inference_result.detected_conditions,
        )

        # Append top salient segment findings to notes
        if explainability and explainability.segment_attributions:
            top_seg = explainability.segment_attributions[0]
            summary_notes.append(
                f"Peak physiological attribution localized to {top_seg.segment_name} "
                f"({top_seg.relative_contribution_pct:.1f}% of salient energy). "
                f"Physiological correlate: {top_seg.clinical_correlate}"
            )

    # 5. Assemble validated AnalyticsResult
    result = AnalyticsResult(
        composite_risk_score=composite_score,
        composite_risk_tier=composite_tier,
        condition_risks=condition_risks,
        explainability=explainability,
        summary_notes=summary_notes,
        disclaimer=MANDATORY_DISCLAIMER,
    )

    return result
