"""
Unit Tests for Chem Agent (src/drug/).
Compliance Notice: Research use only, not for clinical decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from drug.admet import ADMETPredictor
from drug.docking import VinaDockingEngine
from drug.library import LigandLibrary
from drug.ranking import CandidateRanker
from drug.screener import screen_drugs_for_target
from protein.analyzer import process_protein
from pipeline.schemas import (
    ADMETProfile,
    BindingPocket,
    DrugScreeningResult,
    MANDATORY_DISCLAIMER,
)


def test_ligand_library():
    lib = LigandLibrary()
    adrb1_ligs = lib.get_ligands_for_target("ADRB1")
    assert len(adrb1_ligs) >= 4
    names = [l["name"] for l in adrb1_ligs]
    assert "Metoprolol" in names
    assert "Atenolol" in names


def test_admet_profiling():
    predictor = ADMETPredictor()
    # Test Metoprolol: CC(C)NCC(O)COc1ccc(CCOCC)cc1
    profile = predictor.profile("CC(C)NCC(O)COc1ccc(CCOCC)cc1")

    assert isinstance(profile, ADMETProfile)
    assert 200.0 < profile.molecular_weight < 350.0
    assert profile.passes_lipinski is True
    assert 0.0 <= profile.herg_inhibition_probability <= 1.0
    assert MANDATORY_DISCLAIMER in profile.disclaimer


def test_vina_docking_engine():
    engine = VinaDockingEngine()
    pocket = BindingPocket(
        pocket_id=1,
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        box_size_x=20.0,
        box_size_y=20.0,
        box_size_z=20.0,
        druggability_score=0.75,
        disclaimer=MANDATORY_DISCLAIMER,
    )
    affinity, poses = engine.dock_ligand(
        smiles="CC(C)NCC(O)COc1ccc(CCOCC)cc1",
        pocket=pocket,
        heavy_atoms=20,
    )

    assert affinity < 0.0  # Favorable negative free energy
    assert len(poses) == 3
    assert poses[0].mode == 1
    assert poses[0].affinity_kcal_mol == affinity


def test_candidate_ranker():
    ranker = CandidateRanker()
    admet_good = ADMETProfile(
        molecular_weight=280.0,
        logP=1.5,
        h_bond_donors=2,
        h_bond_acceptors=4,
        tpsa=50.0,
        rotatable_bonds=4,
        lipinski_violations=0,
        passes_lipinski=True,
        herg_inhibition_probability=0.20,
        herg_risk_tier="Low",
        passes_cardiac_safety=True,
        disclaimer=MANDATORY_DISCLAIMER,
    )
    score_good = ranker.compute_composite_score(-9.0, admet_good)

    admet_bad = ADMETProfile(
        molecular_weight=550.0,
        logP=5.8,
        h_bond_donors=6,
        h_bond_acceptors=12,
        tpsa=140.0,
        rotatable_bonds=11,
        lipinski_violations=3,
        passes_lipinski=False,
        herg_inhibition_probability=0.85,
        herg_risk_tier="High",
        passes_cardiac_safety=False,
        disclaimer=MANDATORY_DISCLAIMER,
    )
    score_bad = ranker.compute_composite_score(-9.0, admet_bad)

    assert score_good > score_bad


def test_screen_drugs_for_target():
    prot = process_protein("P08588", "ADRB1")
    result = screen_drugs_for_target(prot)

    assert isinstance(result, DrugScreeningResult)
    assert result.target_gene == "ADRB1"
    assert len(result.ranked_candidates) > 0
    assert result.top_lead is not None
    assert result.top_lead.rank == 1
    assert MANDATORY_DISCLAIMER in result.disclaimer
