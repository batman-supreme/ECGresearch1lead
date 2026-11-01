"""
Pipeline package for ECG-AI-Research-1Lead.
RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.
"""

from src.pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ResearchBaseModel,
    ECGConditionPrediction,
    ECGInferenceResult,
    RiskTier,
    WaveformSegmentAttribution,
    ConditionRiskDetail,
    ExplainabilitySummary,
    AnalyticsResult,
    TargetGene,
    ConditionTargetMapping,
    BridgeResult,
    ResidueConfidenceSummary,
    BindingPocket,
    ProteinStructureResult,
    ADMETProfile,
    DockingPose,
    DrugCandidate,
    DrugScreeningResult,
    PipelineExecutionMetadata,
    PipelineRunRecord,
)

__all__ = [
    "MANDATORY_DISCLAIMER",
    "ResearchBaseModel",
    "ECGConditionPrediction",
    "ECGInferenceResult",
    "RiskTier",
    "WaveformSegmentAttribution",
    "ConditionRiskDetail",
    "ExplainabilitySummary",
    "AnalyticsResult",
    "TargetGene",
    "ConditionTargetMapping",
    "BridgeResult",
    "ResidueConfidenceSummary",
    "BindingPocket",
    "ProteinStructureResult",
    "ADMETProfile",
    "DockingPose",
    "DrugCandidate",
    "DrugScreeningResult",
    "PipelineExecutionMetadata",
    "PipelineRunRecord",
    "PipelineOrchestrator",
    "run_pipeline",
]


def __getattr__(name: str):
    """Lazy load orchestrator symbols to prevent circular import during submodule loading."""
    if name in ("PipelineOrchestrator", "run_pipeline"):
        from src.pipeline.orchestrator import PipelineOrchestrator, run_pipeline
        globals()["PipelineOrchestrator"] = PipelineOrchestrator
        globals()["run_pipeline"] = run_pipeline
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
