"""
Multi-Objective Drug Candidate Ranking and Safety Filtering Engine.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Ranks screened small molecules using a composite multi-parameter optimization (MPO)
score balancing Vina binding affinity, Lipinski drug-likeness, and cardiac hERG safety.
"""

from __future__ import annotations

from typing import List, Optional

from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ADMETProfile,
    DockingPose,
    DrugCandidate,
    DrugScreeningResult,
)


class CandidateRanker:
    """
    Ranks screened ligands by integrating binding affinity, drug-likeness,
    and hERG cardiac liability into a single composite score.
    """

    def __init__(
        self,
        weight_affinity: float = 0.50,
        weight_admet: float = 0.25,
        weight_herg: float = 0.25,
        strict_cardiac_safety: bool = True,
    ):
        self.w_aff = weight_affinity
        self.w_admet = weight_admet
        self.w_herg = weight_herg
        self.strict_cardiac = strict_cardiac_safety

    def compute_composite_score(
        self,
        affinity_kcal_mol: float,
        admet: ADMETProfile,
    ) -> float:
        """
        Calculates normalized composite score (0-100).
        - Higher binding affinity (more negative kcal/mol) increases score.
        - Fewer Lipinski violations increases score.
        - Low hERG liability increases score; high hERG penalizes severely.
        """
        # 1. Affinity score: -4 kcal/mol -> 0 pts, -12 kcal/mol -> 100 pts
        aff_pts = ((-affinity_kcal_mol - 4.0) / 8.0) * 100.0
        aff_score = max(0.0, min(100.0, aff_pts))

        # 2. Drug-likeness score: 100 - (30 * violations)
        admet_score = max(0.0, 100.0 - (admet.lipinski_violations * 30.0))

        # 3. Cardiac safety score: (1.0 - herg_probability) * 100
        herg_score = max(0.0, (1.0 - admet.herg_inhibition_probability) * 100.0)

        composite = (
            self.w_aff * aff_score
            + self.w_admet * admet_score
            + self.w_herg * herg_score
        )

        # Apply hard penalty if flagged as high cardiac risk
        if not admet.passes_cardiac_safety:
            composite *= 0.50  # 50% safety penalty for cardiac liability

        return round(max(0.0, min(100.0, composite)), 2)

    def rank_candidates(
        self,
        candidates: List[DrugCandidate],
    ) -> List[DrugCandidate]:
        """
        Filter and sort candidates in descending order of composite score.
        """
        # Optionally filter out unsafe compounds
        eligible: List[DrugCandidate] = []
        for cand in candidates:
            if self.strict_cardiac and not cand.admet.passes_cardiac_safety:
                # If strict, exclude candidates failing cardiac safety
                continue
            eligible.append(cand)

        if not eligible and candidates:
            # If all candidates had moderate/high liability, retain all with safety warnings
            eligible = list(candidates)

        # Sort descending by composite score, then by affinity (more negative first)
        eligible.sort(
            key=lambda c: (c.composite_score, -c.binding_affinity_kcal_mol),
            reverse=True,
        )

        # Assign final rank indices
        for rank_idx, cand in enumerate(eligible, start=1):
            cand.rank = rank_idx

        return eligible

    def create_screening_result(
        self,
        target_gene: str,
        target_uniprot_id: str,
        total_evaluated: int,
        ranked_candidates: List[DrugCandidate],
    ) -> DrugScreeningResult:
        """
        Wrap ranked cohort into validated DrugScreeningResult container.
        """
        top_lead = ranked_candidates[0] if len(ranked_candidates) > 0 else None

        return DrugScreeningResult(
            target_gene=target_gene,
            target_uniprot_id=target_uniprot_id,
            total_ligands_evaluated=total_evaluated,
            candidates_passing_filters=len(ranked_candidates),
            ranked_candidates=ranked_candidates,
            top_lead=top_lead,
            disclaimer=MANDATORY_DISCLAIMER,
        )
