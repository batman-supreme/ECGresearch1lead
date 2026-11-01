"""
Master Pipeline Orchestrator for ECG-AI-Research-1Lead.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Coordinates the end-to-end execution flow:
ECG Signal -> Inference -> Risk Analytics & Explainability -> Target Bridge
-> AlphaFold Protein Structures -> Drug Screening & Docking -> Run Logging.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
import predict
from analytics.analyzer import analyze_ecg
from bridge.mapper import resolve_targets
from drug.screener import screen_pipeline_targets
from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    AnalyticsResult,
    BridgeResult,
    DrugScreeningResult,
    ECGInferenceResult,
    PipelineExecutionMetadata,
    PipelineRunRecord,
    ProteinStructureResult,
)
from protein.analyzer import process_targets

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "pipeline.yaml"
_RUNS_DIR = _PROJECT_ROOT / "results" / "runs"


def _load_pipeline_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    target = Path(config_path) if config_path else _CONFIG_PATH
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class PipelineOrchestrator:
    """
    Executes and audits the entire cross-disciplinary pipeline from raw ECG waveform
    to ranked therapeutic candidates, persisting run records to results/runs/<run_id>.json.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        runs_dir: Optional[Union[str, Path]] = None,
    ):
        self.config_path = config_path or _CONFIG_PATH
        self.cfg = _load_pipeline_config(self.config_path)

        # Setup runs output directory
        if runs_dir:
            self.runs_dir = Path(runs_dir).resolve()
        else:
            rel_dir = self.cfg.get("paths", {}).get("runs_dir", "results/runs")
            self.runs_dir = (_PROJECT_ROOT / rel_dir).resolve()

        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        ecg: Union[torch.Tensor, np.ndarray, Sequence[float], str, Path],
        run_name: Optional[str] = None,
        include_explainability: bool = True,
        max_targets_to_model: int = 2,
    ) -> PipelineRunRecord:
        """
        Execute full end-to-end pipeline run.

        RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.

        Parameters:
            ecg: 1-lead ECG signal array (5000 samples) or path to waveform file.
            run_name: Optional custom run identifier tag.
            include_explainability: Whether to calculate Integrated Gradients attribution.
            max_targets_to_model: Maximum number of biological targets to dock small molecules against.

        Returns:
            PipelineRunRecord: Complete audit log conforming strictly to Pydantic schema.
        """
        start_time = time.time()
        timestamp_now = datetime.now(timezone.utc)
        unique_suffix = uuid.uuid4().hex[:8]
        timestamp_str = timestamp_now.strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{run_name or 'ecg'}_{timestamp_str}_{unique_suffix}"

        input_source = "memory_array"
        # 1. ECG Ingestion & Inference
        if isinstance(ecg, (str, Path)):
            input_source = str(ecg)
            inference_result = ecg_api.infer_from_file(ecg, config_path=self.config_path)
            # Load raw signal for downstream analytics
            if str(ecg).endswith(".npy"):
                raw_signal = np.load(ecg)
            elif str(ecg).endswith(".pt"):
                raw_signal = torch.load(ecg, map_location="cpu").numpy()
            elif str(ecg).endswith(".pdf"):
                from samsung_pdf import extract_samsung_ecg
                raw_signal = extract_samsung_ecg(ecg).numpy()
            elif str(ecg).lower().endswith((".csv", ".tsv", ".txt")):
                raw_signal = ecg_api.load_ecg_csv(ecg)
            else:
                raw_signal = ecg_api.load_ecg_csv(ecg)
        else:
            raw_signal = ecg
            inference_result = ecg_api.infer(ecg, config_path=self.config_path)

        # 2. Risk Analytics & Waveform Explainability
        analytics_result: AnalyticsResult = analyze_ecg(
            ecg=raw_signal,
            inference_result=inference_result,
            config_path=self.config_path,
            include_explainability=include_explainability,
        )

        # 3. Target Bridge (Strictly configs/targets.yaml)
        # Query targets for detected conditions (or fallback to NORM/MI baseline if normal)
        query_conditions = inference_result.detected_conditions
        if not query_conditions or query_conditions == ["NORM"]:
            query_conditions = ["NORM"]

        bridge_result: BridgeResult = resolve_targets(
            conditions=query_conditions,
            targets_yaml_path=_PROJECT_ROOT / "configs" / "targets.yaml",
        )

        # 4. Protein Structure Retrieval & Quality Filtering (AlphaFold DB)
        protein_structures: List[ProteinStructureResult] = []
        if bridge_result.resolved_targets:
            protein_structures = process_targets(
                targets=bridge_result.resolved_targets,
                max_targets=max_targets_to_model,
            )

        # 5. Small-Molecule Drug Screening & Vina Docking
        drug_screenings: List[DrugScreeningResult] = []
        if protein_structures:
            drug_screenings = screen_pipeline_targets(
                protein_results=protein_structures,
                config_path=self.config_path,
            )

        # 6. Synthesize Executive Research Findings
        findings: List[str] = []
        findings.append(f"Inference detected conditions: {', '.join(inference_result.detected_conditions) or 'None'}")
        findings.append(f"Cardiovascular Composite Risk: {analytics_result.composite_risk_score}/100 ({analytics_result.composite_risk_tier.value} Tier)")

        if analytics_result.explainability and analytics_result.explainability.segment_attributions:
            top_seg = analytics_result.explainability.segment_attributions[0]
            findings.append(f"Primary morphological attribution: {top_seg.segment_name} ({top_seg.relative_contribution_pct:.1f}%)")

        if bridge_result.resolved_targets:
            gene_names = [t.gene for t in bridge_result.resolved_targets[:max_targets_to_model]]
            findings.append(f"Prioritized biological targets: {', '.join(gene_names)}")

        for screening in drug_screenings:
            if screening.top_lead:
                lead = screening.top_lead
                findings.append(
                    f"Top candidate for {screening.target_gene}: {lead.name} "
                    f"(Vina: {lead.binding_affinity_kcal_mol:.2f} kcal/mol, "
                    f"hERG Risk: {lead.admet.herg_risk_tier}, Score: {lead.composite_score:.1f})"
                )

        duration = round(time.time() - start_time, 3)

        # 7. Construct PipelineRunRecord
        metadata = PipelineExecutionMetadata(
            run_id=run_id,
            timestamp_utc=timestamp_now,
            execution_duration_sec=duration,
            model_checkpoint=str(predict.CHECKPOINT_PATH),
            input_source=input_source,
            disclaimer=MANDATORY_DISCLAIMER,
        )

        record = PipelineRunRecord(
            metadata=metadata,
            ecg_inference=inference_result,
            analytics=analytics_result,
            bridge=bridge_result,
            protein_structures=protein_structures,
            drug_screenings=drug_screenings,
            summary_findings=findings,
            disclaimer=MANDATORY_DISCLAIMER,
        )

        # 8. Persist Run to results/runs/<run_id>.json
        run_file = self.runs_dir / f"{run_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=2))

        return record


# Convenience function
def run_pipeline(
    ecg: Union[torch.Tensor, np.ndarray, Sequence[float], str, Path],
    run_name: Optional[str] = None,
    include_explainability: bool = True,
    max_targets_to_model: int = 2,
    config_path: Optional[Union[str, Path]] = None,
) -> PipelineRunRecord:
    """
    Execute full end-to-end pipeline run.
    """
    orchestrator = PipelineOrchestrator(config_path=config_path)
    return orchestrator.run(
        ecg=ecg,
        run_name=run_name,
        include_explainability=include_explainability,
        max_targets_to_model=max_targets_to_model,
    )
