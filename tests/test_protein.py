"""
Unit Tests for Protein Agent (src/protein/).
Compliance Notice: Research use only, not for clinical decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from protein.analyzer import process_protein
from protein.pockets import PocketDetector
from protein.quality import StructureQualityEvaluator
from protein.retriever import AlphaFoldRetriever
from pipeline.schemas import MANDATORY_DISCLAIMER, ProteinStructureResult


def test_alphafold_retriever_cache(tmp_path: Path):
    retriever = AlphaFoldRetriever(cache_dir=tmp_path)
    # Test retrieving P08588
    info = retriever.retrieve_structure("P08588")
    assert info["cached_locally"] is True
    assert Path(info["pdb_path"]).exists()
    assert Path(info["pdb_path"]).stat().st_size > 0


def test_structure_quality_evaluator(tmp_path: Path):
    # Use cached or local structure
    retriever = AlphaFoldRetriever(cache_dir=tmp_path)
    info = retriever.retrieve_structure("P08588")

    evaluator = StructureQualityEvaluator()
    summary = evaluator.evaluate_plddt(info["pdb_path"])

    assert 0.0 <= summary.mean_plddt <= 100.0
    assert summary.passes_qc is True
    assert MANDATORY_DISCLAIMER in summary.disclaimer


def test_pocket_detector(tmp_path: Path):
    retriever = AlphaFoldRetriever(cache_dir=tmp_path)
    info = retriever.retrieve_structure("P08588")

    detector = PocketDetector()
    pockets = detector.detect_pockets(info["pdb_path"])

    assert len(pockets) > 0
    top_pocket = pockets[0]
    assert 0.0 <= top_pocket.druggability_score <= 1.0
    assert top_pocket.box_size_x == 20.0
    assert MANDATORY_DISCLAIMER in top_pocket.disclaimer


def test_process_protein_pipeline():
    result = process_protein("P08588", "ADRB1")
    assert isinstance(result, ProteinStructureResult)
    assert result.gene_symbol == "ADRB1"
    assert len(result.detected_pockets) > 0
    assert result.selected_pocket is not None
    assert MANDATORY_DISCLAIMER in result.disclaimer
