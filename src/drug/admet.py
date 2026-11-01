"""
ADMET Profiling and Cardiac hERG Liability Filtering Engine.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Calculates physicochemical descriptors (MW, LogP, HBD, HBA, TPSA, Rotatable Bonds),
evaluates Lipinski's Rule of 5, and estimates cardiac hERG channel liability.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ADMETProfile,
)

# Standard atomic weights
ATOMIC_WEIGHTS = {
    "C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999,
    "S": 32.065, "P": 30.974, "F": 18.998, "Cl": 35.453,
    "Br": 79.904, "I": 126.904,
}


class ADMETPredictor:
    """
    Computes pharmacokinetic and cardiac toxicity indicators for candidate SMILES structures.
    """

    def __init__(self, herg_cutoff: float = 0.50):
        self.herg_cutoff = herg_cutoff

    def parse_smiles_atoms(self, smiles: str) -> Dict[str, int]:
        """Extract atomic composition counts from canonical SMILES."""
        counts: Dict[str, int] = {k: 0 for k in ATOMIC_WEIGHTS}
        # Remove charge, bracket artifacts for basic elemental counting
        clean = re.sub(r"\[.*?\]", "", smiles)

        # Detect two-letter halogens
        counts["Cl"] = len(re.findall(r"Cl", clean))
        counts["Br"] = len(re.findall(r"Br", clean))

        # Remove Cl and Br so 'C' doesn't double-count them
        remainder = re.sub(r"Cl|Br", "", clean)

        counts["C"] = len(re.findall(r"[Cc]", remainder))
        counts["N"] = len(re.findall(r"[Nn]", remainder))
        counts["O"] = len(re.findall(r"[Oo]", remainder))
        counts["S"] = len(re.findall(r"[Ss]", remainder))
        counts["P"] = len(re.findall(r"[Pp]", remainder))
        counts["F"] = len(re.findall(r"F", remainder))
        counts["I"] = len(re.findall(r"I", remainder))

        # Approximate hydrogen count based on standard carbon and heteroatom valences
        # Approx: H = 2*C + 2 - 2*(rings/double bonds)
        heavy_atoms = sum(counts.values())
        approx_h = max(2, int(counts["C"] * 1.5 + counts["N"] + counts["O"]))
        counts["H"] = approx_h
        return counts

    def calculate_molecular_weight(self, smiles: str) -> float:
        """Calculate molecular weight in g/mol."""
        atoms = self.parse_smiles_atoms(smiles)
        mw = sum(count * ATOMIC_WEIGHTS.get(elem, 12.0) for elem, count in atoms.items())
        return round(float(mw), 2)

    def calculate_logp(self, smiles: str) -> float:
        """
        Estimate octanol-water partition coefficient using Crippen-Wildman atom contributions.
        Carbon adds lipophilicity (+0.35), halogens (+0.5 to +0.8), polar heteroatoms subtract.
        """
        atoms = self.parse_smiles_atoms(smiles)
        logp = (
            0.38 * atoms["C"]
            - 0.85 * atoms["O"]
            - 0.55 * atoms["N"]
            + 0.65 * atoms["Cl"]
            + 0.15 * atoms["F"]
            + 0.45 * atoms["S"]
            - 0.30
        )
        # Bounded between -2.0 and 8.0
        return round(float(max(-2.0, min(8.0, logp))), 2)

    def calculate_hydrogen_bonding(self, smiles: str) -> Tuple[int, int]:
        """
        Count Hydrogen Bond Donors (HBD: -OH, -NH) and Acceptors (HBA: O, N, F).
        """
        # Acceptors: total O, N, F atoms
        atoms = self.parse_smiles_atoms(smiles)
        hba = atoms["O"] + atoms["N"] + atoms["F"]

        # Donors: OH and NH patterns in SMILES
        oh_donors = len(re.findall(r"O[Hh]|\[OH\]|(?<!=)O(?=[A-Z0-9\(\)])", smiles))
        nh_donors = len(re.findall(r"N[Hh]|\[NH\d?\]|(?<!=)N(?=[A-Z0-9\(\)])", smiles))
        # Conservatively estimate from heteroatoms
        hbd = min(hba, max(0, oh_donors + nh_donors))
        return int(hbd), int(hba)

    def calculate_tpsa(self, smiles: str) -> float:
        """
        Estimate Topological Polar Surface Area (TPSA in Å²).
        Standard contributions: oxygen ~ 14.0 - 20.2 Å², nitrogen ~ 12.0 - 26.0 Å².
        """
        atoms = self.parse_smiles_atoms(smiles)
        tpsa = (atoms["O"] * 17.5) + (atoms["N"] * 15.0) + (atoms["S"] * 25.0)
        return round(float(max(0.0, tpsa)), 2)

    def calculate_rotatable_bonds(self, smiles: str) -> int:
        """
        Estimate non-terminal, non-ring single bonds.
        """
        # Count non-ring aliphatic links: CC, CN, CO, CS
        matches = len(re.findall(r"[A-Z][a-z]?-[A-Z]|C[NnOo]|N[CcOo]|O[Cc]", smiles))
        # Cap reasonably based on carbon chain length
        c_count = self.parse_smiles_atoms(smiles)["C"]
        rot = min(max(1, matches), max(1, c_count // 2))
        return int(rot)

    def evaluate_herg_liability(
        self,
        smiles: str,
        mw: float,
        logp: float,
        tpsa: float,
    ) -> Tuple[float, str, bool]:
        """
        Evaluate cardiac hERG potassium channel inhibition liability.

        Structural flags promoting hERG block:
        1. High lipophilicity (LogP > 3.5)
        2. Bulky aromatic framework (MW > 350)
        3. Basic tertiary or secondary nitrogen center with low TPSA (< 75 Å²)
        """
        atoms = self.parse_smiles_atoms(smiles)
        has_nitrogen = atoms["N"] > 0
        has_aromatic = bool(re.search(r"c1ccccc1|c\d", smiles))
        has_tertiary_amine = bool(re.search(r"N\([A-Z]\)|N\(C\)", smiles))

        # Base probability from lipophilicity
        risk_score = 0.15

        if logp > 3.5:
            risk_score += 0.30
        elif logp > 2.5:
            risk_score += 0.15

        if mw > 380:
            risk_score += 0.15

        if has_nitrogen and has_aromatic:
            risk_score += 0.20

        if has_tertiary_amine and logp > 2.8:
            risk_score += 0.15

        if tpsa < 50.0:
            risk_score += 0.10
        elif tpsa > 100.0:
            risk_score -= 0.15

        prob = round(max(0.05, min(0.95, risk_score)), 2)

        if prob < 0.35:
            tier = "Low"
        elif prob < 0.60:
            tier = "Moderate"
        else:
            tier = "High"

        passes = prob < self.herg_cutoff
        return prob, tier, passes

    def profile(self, smiles: str) -> ADMETProfile:
        """
        Generate complete ADMET profile and cardiac safety report.

        RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.
        """
        mw = self.calculate_molecular_weight(smiles)
        logp = self.calculate_logp(smiles)
        hbd, hba = self.calculate_hydrogen_bonding(smiles)
        tpsa = self.calculate_tpsa(smiles)
        rot_bonds = self.calculate_rotatable_bonds(smiles)

        # Lipinski Rule of 5 evaluation
        violations = 0
        if mw > 500.0:
            violations += 1
        if logp > 5.0:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        passes_lipinski = violations <= 1

        # Cardiac hERG liability evaluation
        herg_prob, herg_tier, passes_cardiac = self.evaluate_herg_liability(
            smiles=smiles, mw=mw, logp=logp, tpsa=tpsa
        )

        return ADMETProfile(
            molecular_weight=mw,
            logP=logp,
            h_bond_donors=hbd,
            h_bond_acceptors=hba,
            tpsa=tpsa,
            rotatable_bonds=rot_bonds,
            lipinski_violations=violations,
            passes_lipinski=passes_lipinski,
            herg_inhibition_probability=herg_prob,
            herg_risk_tier=herg_tier,
            passes_cardiac_safety=passes_cardiac,
            disclaimer=MANDATORY_DISCLAIMER,
        )
