"""
Master Drug Screening Orchestrator.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Integrates Ligand Library, ADMET/hERG Profiling, Vina Docking Simulation,
and Multi-Objective Ranking for AlphaFold target proteins.
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

from drug.admet import ADMETPredictor
from drug.docking import VinaDockingEngine
from drug.library import LigandLibrary
from drug.ranking import CandidateRanker
from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    BindingPocket,
    DrugCandidate,
    DrugScreeningResult,
    ProteinStructureResult,
)

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "pipeline.yaml"


def _load_drug_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    target = Path(config_path) if config_path else _CONFIG_PATH
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("drug", {})
    return {}


def screen_drugs_for_target(
    protein_result: ProteinStructureResult,
    custom_ligands: Optional[List[Dict[str, str]]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> DrugScreeningResult:
    """
    Screen and rank small-molecule candidate ligands against a protein target.

    RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.

    Parameters:
        protein_result: Evaluated ProteinStructureResult from src/protein/.
        custom_ligands: Optional list of additional ligands to screen.
        config_path: Optional pipeline configuration yaml path.

    Returns:
        DrugScreeningResult: Container with ranked DrugCandidate records,
        ADMET profiles, Vina docking affinities, poses, and top lead.
    """
    cfg = _load_drug_config(config_path)
    herg_cutoff = float(cfg.get("admet", {}).get("herg_inhibition_threshold", 0.50))
    exhaustiveness = int(cfg.get("vina", {}).get("exhaustiveness", 8))
    num_modes = int(cfg.get("vina", {}).get("num_modes", 3))

    # 1. Resolve target binding pocket
    pocket = protein_result.selected_pocket
    if pocket is None and protein_result.detected_pockets:
        pocket = protein_result.detected_pockets[0]
    elif pocket is None:
        # Fallback default cavity
        pocket = BindingPocket(
            pocket_id=1,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            box_size_x=20.0,
            box_size_y=20.0,
            box_size_z=20.0,
            druggability_score=0.65,
            description="Default central binding grid",
            disclaimer=MANDATORY_DISCLAIMER,
        )

    # 2. Retrieve ligands for target gene
    library = LigandLibrary(custom_ligands=custom_ligands)
    target_ligands = library.get_ligands_for_target(protein_result.gene_symbol)

    # 3. Initialize components
    admet_predictor = ADMETPredictor(herg_cutoff=herg_cutoff)
    docking_engine = VinaDockingEngine(exhaustiveness=exhaustiveness, num_modes=num_modes)
    ranker = CandidateRanker(strict_cardiac_safety=False)  # Rank all with safety penalty

    candidates: List[DrugCandidate] = []

    for lig in target_ligands:
        smiles = lig.get("smiles", "")
        if not smiles:
            continue

        # A. ADMET & Cardiac Safety Profiling
        admet_prof = admet_predictor.profile(smiles)

        # B. Vina Docking & Affinity Estimation
        heavy_atoms = sum(admet_predictor.parse_smiles_atoms(smiles).values())
        affinity_kcal, poses = docking_engine.dock_ligand(
            smiles=smiles,
            pocket=pocket,
            heavy_atoms=heavy_atoms,
            rotatable_bonds=admet_prof.rotatable_bonds,
            logp=admet_prof.logP,
            hbd=admet_prof.h_bond_donors,
            hba=admet_prof.h_bond_acceptors,
        )

        # C. Composite MPO Score
        comp_score = ranker.compute_composite_score(
            affinity_kcal_mol=affinity_kcal,
            admet=admet_prof,
        )

        cand = DrugCandidate(
            compound_id=lig.get("compound_id", "DRUG-UNKNOWN"),
            name=lig.get("name", "Unknown Compound"),
            smiles=smiles,
            target_gene=protein_result.gene_symbol,
            target_uniprot_id=protein_result.uniprot_id,
            pocket_id=pocket.pocket_id,
            binding_affinity_kcal_mol=affinity_kcal,
            poses=poses,
            admet=admet_prof,
            composite_score=comp_score,
            rank=1,
            disclaimer=MANDATORY_DISCLAIMER,
        )
        candidates.append(cand)

    # 4. Rank candidates
    ranked = ranker.rank_candidates(candidates)

    return ranker.create_screening_result(
        target_gene=protein_result.gene_symbol,
        target_uniprot_id=protein_result.uniprot_id,
        total_evaluated=len(target_ligands),
        ranked_candidates=ranked,
    )


def screen_pipeline_targets(
    protein_results: Sequence[ProteinStructureResult],
    config_path: Optional[Union[str, Path]] = None,
) -> List[DrugScreeningResult]:
    """
    Batch screen small-molecule candidate cohorts for all evaluated target proteins.
    """
    results: List[DrugScreeningResult] = []
    for prot in protein_results:
        res = screen_drugs_for_target(prot, config_path=config_path)
        results.append(res)
    return results
