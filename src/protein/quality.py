"""
AlphaFold Protein Structure Quality and pLDDT/PAE Filtering Engine.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Parses AlphaFold PDB files to calculate pLDDT confidence metrics
and evaluates Predicted Aligned Error (PAE) matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ResidueConfidenceSummary,
)


class StructureQualityEvaluator:
    """
    Evaluates per-residue pLDDT confidence scores from AlphaFold PDB B-factors
    and computes distribution statistics and quality control pass/fail decisions.
    """

    def __init__(
        self,
        high_confidence_cutoff: float = 70.0,
        very_high_confidence_cutoff: float = 90.0,
    ):
        self.high_cutoff = high_confidence_cutoff
        self.very_high_cutoff = very_high_confidence_cutoff

    def parse_plddt_scores(self, pdb_file_path: Union[str, Path]) -> Tuple[List[float], int]:
        """
        Extract per-residue pLDDT scores from the B-factor column of C-alpha atoms.

        Standard PDB column convention for B-factor: characters 61-66 (0-indexed: 60:66).
        """
        path = Path(pdb_file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDB structure file not found: {path}")

        ca_plddt_scores: List[float] = []
        seen_residues: set[int] = set()

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    # Focus on C-alpha (CA) atoms to sample one pLDDT per residue
                    if atom_name == "CA":
                        try:
                            res_seq = int(line[22:26].strip())
                            if res_seq not in seen_residues:
                                seen_residues.add(res_seq)
                                b_factor = float(line[60:66].strip())
                                ca_plddt_scores.append(b_factor)
                        except (ValueError, IndexError):
                            continue

        if not ca_plddt_scores:
            # Fallback if no CA atoms found: parse all ATOM B-factors
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("ATOM") and len(line) >= 66:
                        try:
                            b_factor = float(line[60:66].strip())
                            ca_plddt_scores.append(b_factor)
                        except ValueError:
                            continue

        return ca_plddt_scores, len(seen_residues) or len(ca_plddt_scores)

    def evaluate_plddt(
        self,
        pdb_file_path: Union[str, Path],
    ) -> ResidueConfidenceSummary:
        """
        Compute comprehensive pLDDT distribution statistics:
        - Very High (pLDDT >= 90)
        - Confident (70 <= pLDDT < 90)
        - Low (50 <= pLDDT < 70)
        - Very Low (pLDDT < 50)
        """
        scores, total_res = self.parse_plddt_scores(pdb_file_path)

        if not scores:
            return ResidueConfidenceSummary(
                mean_plddt=0.0,
                very_high_plddt_pct=0.0,
                confident_plddt_pct=0.0,
                low_plddt_pct=0.0,
                very_low_plddt_pct=100.0,
                passes_qc=False,
                disclaimer=MANDATORY_DISCLAIMER,
            )

        arr = np.array(scores, dtype=np.float64)
        n = len(arr)

        very_high = float(np.sum(arr >= self.very_high_cutoff))
        confident = float(np.sum((arr >= self.high_cutoff) & (arr < self.very_high_cutoff)))
        low = float(np.sum((arr >= 50.0) & (arr < self.high_cutoff)))
        very_low = float(np.sum(arr < 50.0))

        mean_val = float(np.mean(arr))
        passes = mean_val >= self.high_cutoff

        return ResidueConfidenceSummary(
            mean_plddt=round(mean_val, 2),
            very_high_plddt_pct=round((very_high / n) * 100.0, 2),
            confident_plddt_pct=round((confident / n) * 100.0, 2),
            low_plddt_pct=round((low / n) * 100.0, 2),
            very_low_plddt_pct=round((very_low / n) * 100.0, 2),
            passes_qc=passes,
            disclaimer=MANDATORY_DISCLAIMER,
        )

    def evaluate_pae(
        self,
        pae_json_path: Union[str, Path],
    ) -> Optional[Dict[str, float]]:
        """
        Parse Predicted Aligned Error (PAE) JSON and compute average domain alignment error.
        """
        path = Path(pae_json_path)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 0 and "predicted_aligned_error" in data[0]:
                matrix = np.array(data[0]["predicted_aligned_error"], dtype=np.float32)
                mean_pae = float(np.mean(matrix))
                max_pae = float(np.max(matrix))
                compact_fraction = float(np.mean(matrix < 15.0)) * 100.0
                return {
                    "mean_pae_angstrom": round(mean_pae, 2),
                    "max_pae_angstrom": round(max_pae, 2),
                    "confident_pair_pct": round(compact_fraction, 2),
                }
        except Exception:
            return None

        return None
