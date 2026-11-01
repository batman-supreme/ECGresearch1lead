"""
Master Protein Processing Module for ECG-AI-Research-1Lead.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Orchestrates AlphaFold DB structure retrieval, caching in data/structures/,
pLDDT/PAE quality assessment, and 3D binding pocket identification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

# Ensure project root and src are importable
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ProteinStructureResult,
    TargetGene,
)
from protein.pockets import PocketDetector
from protein.quality import StructureQualityEvaluator
from protein.retriever import AlphaFoldRetriever

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "pipeline.yaml"


def process_protein(
    uniprot_id: str,
    gene_symbol: str = "",
    protein_name: str = "",
    cache_dir: Optional[Union[str, Path]] = None,
    force_refresh: bool = False,
    config_path: Optional[Union[str, Path]] = None,
) -> ProteinStructureResult:
    """
    Retrieve and evaluate 3D protein structure from AlphaFold DB with local disk caching.

    RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.

    Parameters:
        uniprot_id: Valid Human UniProtKB accession (e.g., 'P08588', 'Q14524').
        gene_symbol: Optional HGNC gene symbol (e.g., 'ADRB1').
        protein_name: Optional human-readable protein title.
        cache_dir: Optional override for local caching folder (default: data/structures/).
        force_refresh: If True, re-downloads even if already cached.
        config_path: Optional override for pipeline configuration yaml.

    Returns:
        ProteinStructureResult: Validated Pydantic container containing local PDB path,
        pLDDT confidence statistics, detected binding pockets, and compliance disclaimer.
    """
    uid = uniprot_id.strip().upper()

    # 1. Retrieve structure from AlphaFold DB (cached in data/structures/)
    retriever = AlphaFoldRetriever(cache_dir=cache_dir)
    retrieval_info = retriever.retrieve_structure(
        uniprot_id=uid,
        download_pae=False,
        force_refresh=force_refresh,
    )

    pdb_path = retrieval_info["pdb_path"]
    meta = retrieval_info.get("metadata", {})

    resolved_gene = gene_symbol or meta.get("gene") or uid
    resolved_name = protein_name or meta.get("uniprotDescription") or f"Protein {uid}"

    # 2. Evaluate structure quality and pLDDT distribution
    evaluator = StructureQualityEvaluator(
        high_confidence_cutoff=70.0,
        very_high_confidence_cutoff=90.0,
    )
    plddt_summary = evaluator.evaluate_plddt(pdb_path)
    _, seq_len = evaluator.parse_plddt_scores(pdb_path)

    # 3. Detect candidate ligand binding pockets for Vina docking
    detector = PocketDetector(min_plddt=70.0, box_size_angstrom=20.0)
    pockets = detector.detect_pockets(pdb_path, max_pockets=3)
    selected_pocket = pockets[0] if len(pockets) > 0 else None

    # 4. Construct validated ProteinStructureResult
    return ProteinStructureResult(
        uniprot_id=uid,
        gene_symbol=resolved_gene,
        protein_name=resolved_name,
        structure_file_path=str(pdb_path),
        cached_locally=True,
        plddt_summary=plddt_summary,
        detected_pockets=pockets,
        selected_pocket=selected_pocket,
        sequence_length=seq_len,
        disclaimer=MANDATORY_DISCLAIMER,
    )


def process_targets(
    targets: Sequence[TargetGene],
    max_targets: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_refresh: bool = False,
) -> List[ProteinStructureResult]:
    """
    Batch process a sequence of TargetGene objects resolved by the bridge.
    """
    results: List[ProteinStructureResult] = []
    cohort = targets[:max_targets] if max_targets is not None else targets

    for target in cohort:
        try:
            res = process_protein(
                uniprot_id=target.uniprot_id,
                gene_symbol=target.gene,
                protein_name=target.name,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
            )
            results.append(res)
        except Exception as e:
            print(f"[Warning] Failed processing structure for {target.gene} ({target.uniprot_id}): {e}")

    return results
