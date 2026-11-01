"""
ECG-AI-Research-1Lead Inter-Module Pydantic Schemas.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Provides strict, validated Pydantic data schemas for data flowing between:
- ECG Inference (src/ecg_api.py)
- Risk Analytics & Explainability (src/analytics/)
- Condition-Gene-UniProt Bridge (src/bridge/)
- AlphaFold Structure & Pocket Profiling (src/protein/)
- Ligand Docking & ADMET/hERG Screening (src/drug/)
- Pipeline Orchestration & Run Logging (src/pipeline/)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator


MANDATORY_DISCLAIMER: str = "Research use only, not for clinical decisions."


class ResearchBaseModel(BaseModel):
    """Base schema enforcing compliance disclaimer on all module contracts."""

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="ignore",
    )

    disclaimer: str = Field(
        default=MANDATORY_DISCLAIMER,
        description="Mandatory regulatory research disclaimer."
    )

    @field_validator("disclaimer", mode="after")
    @classmethod
    def verify_disclaimer(cls, v: str) -> str:
        if MANDATORY_DISCLAIMER not in v:
            return f"{v} | {MANDATORY_DISCLAIMER}"
        return v


# ==============================================================================
# 1. ECG Inference Schemas (src/ecg_api.py)
# ==============================================================================

class ECGConditionPrediction(ResearchBaseModel):
    """Detailed prediction record for a single ECG diagnostic condition."""

    condition: str = Field(description="ECG class identifier (NORM, MI, STTC, CD, HYP)")
    probability: float = Field(ge=0.0, le=1.0, description="Model sigmoid probability output")
    threshold: float = Field(ge=0.0, le=1.0, description="Pre-optimized operational decision threshold")
    is_detected: bool = Field(description="True if probability >= threshold")


class ECGInferenceResult(ResearchBaseModel):
    """Comprehensive output container returned by ecg_api.infer()."""

    probabilities: Dict[str, float] = Field(
        description="Mapping of class labels to predicted probabilities."
    )
    predictions: Dict[str, bool] = Field(
        description="Mapping of class labels to binary thresholded decisions."
    )
    detected_conditions: List[str] = Field(
        default_factory=list,
        description="List of condition labels that exceeded operational thresholds."
    )
    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Threshold values applied during evaluation."
    )
    sampling_rate: int = Field(default=500, description="Sampling rate in Hertz (Hz).")
    signal_length: int = Field(default=5000, description="Number of waveform samples (10s @ 500Hz).")
    device: str = Field(default="cpu", description="Compute device utilized for inference.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of inference execution."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional auxiliary provenance metadata."
    )

    @property
    def is_normal(self) -> bool:
        """Convenience property: True if NORM is detected and no pathology is flagged."""
        pathologies = [c for c in self.detected_conditions if c != "NORM"]
        return len(pathologies) == 0 and self.predictions.get("NORM", False)

    @property
    def positive_conditions(self) -> List[str]:
        """Convenience alias for detected_conditions."""
        return self.detected_conditions

    def __iter__(self):
        """Allows unpacking as: probabilities, predictions = infer(ecg)."""
        return iter((self.probabilities, self.predictions))


# ==============================================================================
# 2. Risk Analytics & Explainability Schemas (src/analytics/)
# ==============================================================================

class RiskTier(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class WaveformSegmentAttribution(ResearchBaseModel):
    """Attribution metrics for a physiological ECG interval."""

    segment_name: str = Field(description="Segment name: P_wave, PR_interval, QRS_complex, ST_segment, T_wave")
    start_sample: int = Field(ge=0, le=5000)
    end_sample: int = Field(ge=0, le=5000)
    attribution_score: float = Field(description="Integrated gradient or saliency energy in segment")
    relative_contribution_pct: float = Field(ge=0.0, le=100.0, description="Percentage of total salient attribution")
    clinical_correlate: str = Field(description="Physiological description of the interval")


class ConditionRiskDetail(ResearchBaseModel):
    """Stratified risk assessment for an individual diagnostic entity."""

    condition: str
    probability: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=100.0, description="Normalized risk score (0-100)")
    risk_tier: RiskTier
    confidence_interval: Tuple[float, float] = Field(description="Estimated (lower, upper) 95% CI")
    contributing_factors: List[str] = Field(default_factory=list)


class ExplainabilitySummary(ResearchBaseModel):
    """Waveform attribution analysis and salient morphological regions."""

    method: str = Field(default="integrated_gradients", description="Explainability technique used")
    baseline: str = Field(default="zero", description="Baseline signal used for attribution")
    saliency_energy: float = Field(description="Total integrated gradient energy across signal")
    top_salient_intervals: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Top salient waveform sample index ranges [start, end]"
    )
    segment_attributions: List[WaveformSegmentAttribution] = Field(
        default_factory=list,
        description="Attributions mapped to physiological ECG intervals"
    )


class AnalyticsResult(ResearchBaseModel):
    """Output container produced by src/analytics/."""

    composite_risk_score: float = Field(ge=0.0, le=100.0, description="Aggregated cardiovascular risk index")
    composite_risk_tier: RiskTier = Field(description="Global risk category")
    condition_risks: Dict[str, ConditionRiskDetail] = Field(
        default_factory=dict,
        description="Individual risk breakdowns per condition"
    )
    explainability: Optional[ExplainabilitySummary] = Field(
        default=None,
        description="Explainability saliency and segment attributions"
    )
    summary_notes: List[str] = Field(
        default_factory=list,
        description="Key analytical findings and alerts"
    )


# ==============================================================================
# 3. Target Bridge Schemas (src/bridge/)
# ==============================================================================

class TargetGene(ResearchBaseModel):
    """Validated biological target gene from configs/targets.yaml."""

    gene: str = Field(description="Official HGNC Gene Symbol (e.g. SCN5A, ADRB1)")
    uniprot_id: str = Field(description="Verified Human UniProt Accession (e.g. Q14524, P08588)")
    name: str = Field(description="Full official protein name")
    organism: str = Field(default="Homo sapiens")
    role: str = Field(description="Physiological and pathological mechanistic role")
    pathway: str = Field(description="Associated biological/electrophysiological pathway")
    priority: int = Field(default=1, ge=1, description="Rank priority within the condition")


class ConditionTargetMapping(ResearchBaseModel):
    """Curated condition-to-targets entry in configs/targets.yaml."""

    condition: str
    name: str
    description: str
    is_pathology: bool
    targets: List[TargetGene] = Field(default_factory=list)


class BridgeResult(ResearchBaseModel):
    """Output container produced by src/bridge/."""

    queried_conditions: List[str] = Field(description="Conditions submitted for target resolution")
    resolved_targets: List[TargetGene] = Field(
        default_factory=list,
        description="Unique validated targets linked to detected conditions"
    )
    uniprot_ids: List[str] = Field(
        default_factory=list,
        description="List of unique UniProt IDs to query in protein phase"
    )
    source_file: str = Field(
        default="configs/targets.yaml",
        description="Configuration registry providing the ground-truth mappings"
    )


# ==============================================================================
# 4. Protein Structure & Pocket Schemas (src/protein/)
# ==============================================================================

class ResidueConfidenceSummary(ResearchBaseModel):
    """AlphaFold pLDDT residue confidence distribution metrics."""

    mean_plddt: float = Field(ge=0.0, le=100.0, description="Mean pLDDT across all residues")
    very_high_plddt_pct: float = Field(ge=0.0, le=100.0, description="Percentage of residues with pLDDT >= 90")
    confident_plddt_pct: float = Field(ge=0.0, le=100.0, description="Percentage of residues with 70 <= pLDDT < 90")
    low_plddt_pct: float = Field(ge=0.0, le=100.0, description="Percentage of residues with 50 <= pLDDT < 70")
    very_low_plddt_pct: float = Field(ge=0.0, le=100.0, description="Percentage of residues with pLDDT < 50")
    passes_qc: bool = Field(description="True if mean_plddt >= operational cutoff (e.g. 70.0)")


class BindingPocket(ResearchBaseModel):
    """Identified 3D ligand binding site or pocket coordinates."""

    pocket_id: int = Field(ge=1)
    center_x: float
    center_y: float
    center_z: float
    box_size_x: float = Field(default=20.0, ge=5.0)
    box_size_y: float = Field(default=20.0, ge=5.0)
    box_size_z: float = Field(default=20.0, ge=5.0)
    druggability_score: float = Field(ge=0.0, le=1.0, description="Predicted pocket druggability score")
    residue_indices: List[int] = Field(default_factory=list, description="Residue sequence indices forming pocket")
    description: str = Field(default="", description="Functional site annotation (e.g., active site, channel pore)")


class ProteinStructureResult(ResearchBaseModel):
    """Output container produced by src/protein/."""

    uniprot_id: str
    gene_symbol: str
    protein_name: str
    structure_file_path: str = Field(description="Local path to cached AlphaFold PDB structure")
    cached_locally: bool = Field(default=True)
    plddt_summary: ResidueConfidenceSummary
    detected_pockets: List[BindingPocket] = Field(default_factory=list)
    selected_pocket: Optional[BindingPocket] = Field(default=None)
    sequence_length: int = Field(default=0)


# ==============================================================================
# 5. Drug Screening & Docking Schemas (src/drug/)
# ==============================================================================

class ADMETProfile(ResearchBaseModel):
    """Physicochemical, pharmacokinetic, and cardiac safety assessment."""

    molecular_weight: float = Field(gt=0.0, description="Molecular weight (g/mol)")
    logP: float = Field(description="Calculated octanol-water partition coefficient")
    h_bond_donors: int = Field(ge=0, description="Number of hydrogen bond donors")
    h_bond_acceptors: int = Field(ge=0, description="Number of hydrogen bond acceptors")
    tpsa: float = Field(ge=0.0, description="Topological polar surface area (Å²)")
    rotatable_bonds: int = Field(ge=0, description="Number of rotatable bonds")
    lipinski_violations: int = Field(ge=0, le=4, description="Lipinski Rule of 5 violations")
    passes_lipinski: bool = Field(description="True if lipinski_violations <= 1")
    herg_inhibition_probability: float = Field(
        ge=0.0, le=1.0,
        description="Estimated probability of hERG liability (cardiac QT prolongation risk)"
    )
    herg_risk_tier: str = Field(default="Low", description="Low, Moderate, High cardiac liability")
    passes_cardiac_safety: bool = Field(description="True if hERG inhibition probability is below cutoff")


class DockingPose(ResearchBaseModel):
    """Conformational pose predicted by AutoDock Vina."""

    mode: int = Field(ge=1)
    affinity_kcal_mol: float = Field(description="Binding free energy in kcal/mol (more negative = stronger)")
    rmsd_lower_bound: float = Field(default=0.0, ge=0.0)
    rmsd_upper_bound: float = Field(default=0.0, ge=0.0)


class DrugCandidate(ResearchBaseModel):
    """Screened small-molecule candidate evaluated for therapeutic modulation."""

    compound_id: str = Field(description="Identifier (e.g. CHEMBL id or internal library code)")
    name: str = Field(description="Compound or drug name")
    smiles: str = Field(description="Canonical SMILES chemical structure")
    target_gene: str = Field(description="Associated target gene symbol")
    target_uniprot_id: str = Field(description="Target UniProt ID")
    pocket_id: int = Field(default=1)
    binding_affinity_kcal_mol: float = Field(description="Best mode binding affinity in kcal/mol")
    poses: List[DockingPose] = Field(default_factory=list)
    admet: ADMETProfile
    composite_score: float = Field(
        description="Multi-objective score balancing binding affinity, drug-likeness, and hERG safety"
    )
    rank: int = Field(ge=1, description="Relative ranking within the screened cohort")


class DrugScreeningResult(ResearchBaseModel):
    """Output container produced by src/drug/."""

    target_gene: str
    target_uniprot_id: str
    total_ligands_evaluated: int = Field(default=0)
    candidates_passing_filters: int = Field(default=0)
    ranked_candidates: List[DrugCandidate] = Field(default_factory=list)
    top_lead: Optional[DrugCandidate] = Field(default=None)


# ==============================================================================
# 6. Pipeline Orchestration & Run Logging Schemas (src/pipeline/)
# ==============================================================================

class PipelineExecutionMetadata(ResearchBaseModel):
    """Audit provenance tracking individual pipeline executions."""

    run_id: str = Field(description="Unique run identifier (e.g., run_20261009_220000_abcd)")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC start time of the pipeline execution"
    )
    execution_duration_sec: float = Field(ge=0.0, description="Total pipeline execution duration in seconds")
    model_checkpoint: str = Field(description="Path to neural network model checkpoint")
    input_source: str = Field(default="synthetic_or_array", description="Source of ECG waveform")


class PipelineRunRecord(ResearchBaseModel):
    """Complete, end-to-end serialized run log saved to results/runs/<run_id>.json."""

    metadata: PipelineExecutionMetadata
    ecg_inference: ECGInferenceResult
    analytics: Optional[AnalyticsResult] = Field(default=None)
    bridge: Optional[BridgeResult] = Field(default=None)
    protein_structures: List[ProteinStructureResult] = Field(default_factory=list)
    drug_screenings: List[DrugScreeningResult] = Field(default_factory=list)
    summary_findings: List[str] = Field(default_factory=list)


# Exported symbols
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
]
