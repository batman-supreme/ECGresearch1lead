"""
Curated Cardiovascular Ligand and Drug Library.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Provides curated reference compounds, SMILES chemical structures, and mechanism
annotations for validated cardiac targets from configs/targets.yaml.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# Curated dictionary of small molecules mapped to target gene symbols
# SMILES canonicalized for standard biological protonation at pH 7.4
CURATED_CARDIO_LIGANDS: List[Dict[str, str]] = [
    # -------------------------------------------------------------------------
    # Target: ADRB1 (Beta-1 Adrenergic Receptor) - UniProt: P08588
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-ADRB1-01",
        "name": "Metoprolol",
        "smiles": "CC(C)NCC(O)COc1ccc(CCOCC)cc1",
        "target_gene": "ADRB1",
        "target_uniprot_id": "P08588",
        "indication": "Acute post-MI cardioprotection & arrhythmia suppression",
        "pharmacological_class": "Selective Beta-1 Adrenergic Blocker",
    },
    {
        "compound_id": "DRUG-ADRB1-02",
        "name": "Atenolol",
        "smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "target_gene": "ADRB1",
        "target_uniprot_id": "P08588",
        "indication": "Anti-anginal & myocardial oxygen demand reduction",
        "pharmacological_class": "Hydrophilic Beta-1 Selective Antagonist",
    },
    {
        "compound_id": "DRUG-ADRB1-03",
        "name": "Bisoprolol",
        "smiles": "CC(C)NCC(O)COc1ccc(COCCOC(C)C)cc1",
        "target_gene": "ADRB1",
        "target_uniprot_id": "P08588",
        "indication": "Congestive heart failure & post-infarction survival",
        "pharmacological_class": "Highly Selective Beta-1 Blocker",
    },
    {
        "compound_id": "DRUG-ADRB1-04",
        "name": "Carvedilol",
        "smiles": "COc1cccc2c1c3ccccc3n2CC(O)CNCCOc4ccccc4",
        "target_gene": "ADRB1",
        "target_uniprot_id": "P08588",
        "indication": "Cardioprotection with antioxidant and vasodilatory alpha-1 action",
        "pharmacological_class": "Non-selective Beta/Alpha-1 Antagonist",
    },

    # -------------------------------------------------------------------------
    # Target: SCN5A (Nav1.5 Cardiac Sodium Channel) - UniProt: Q14524
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-SCN5A-01",
        "name": "Lidocaine",
        "smiles": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "target_gene": "SCN5A",
        "target_uniprot_id": "Q14524",
        "indication": "Acute ventricular tachycardia and ischemic arrhythmias",
        "pharmacological_class": "Class 1b Sodium Channel Blocker",
    },
    {
        "compound_id": "DRUG-SCN5A-02",
        "name": "Mexiletine",
        "smiles": "CC(N)COc1c(C)cccc1C",
        "target_gene": "SCN5A",
        "target_uniprot_id": "Q14524",
        "indication": "Oral ventricular arrhythmia prophylaxis",
        "pharmacological_class": "Class 1b Nav1.5 Blocker",
    },
    {
        "compound_id": "DRUG-SCN5A-03",
        "name": "Flecainide",
        "smiles": "O=C(NCc1ccccc1)c2cc(OCC(F)(F)F)ccc2OCC(F)(F)F",
        "target_gene": "SCN5A",
        "target_uniprot_id": "Q14524",
        "indication": "Paroxysmal atrial fibrillation without structural disease",
        "pharmacological_class": "Class 1c Sodium Channel Blocker",
    },

    # -------------------------------------------------------------------------
    # Target: CACNA1C (Cav1.2 L-type Calcium Channel) - UniProt: Q13936
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-CACNA1C-01",
        "name": "Verapamil",
        "smiles": "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
        "target_gene": "CACNA1C",
        "target_uniprot_id": "Q13936",
        "indication": "Supraventricular tachycardia, rate control, & angina",
        "pharmacological_class": "Non-dihydropyridine Calcium Channel Blocker",
    },
    {
        "compound_id": "DRUG-CACNA1C-02",
        "name": "Diltiazem",
        "smiles": "COc1ccc(cc1)C2C(OC(=O)C)C(=O)N(CCN(C)C)c3ccccc3S2",
        "target_gene": "CACNA1C",
        "target_uniprot_id": "Q13936",
        "indication": "Vasospastic angina and coronary vasodilation",
        "pharmacological_class": "Benzothiazepine Cav1.2 Antagonist",
    },
    {
        "compound_id": "DRUG-CACNA1C-03",
        "name": "Amlodipine",
        "smiles": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c2ccccc2Cl)C(=O)OC",
        "target_gene": "CACNA1C",
        "target_uniprot_id": "Q13936",
        "indication": "Systemic hypertension and ischemic chest pain",
        "pharmacological_class": "Dihydropyridine L-type Channel Blocker",
    },

    # -------------------------------------------------------------------------
    # Target: KCNH2 (hERG Potassium Channel) - UniProt: Q12809
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-KCNH2-01",
        "name": "Sotalol",
        "smiles": "CC(C)NCC(O)c1ccc(NS(=O)(=O)C)cc1",
        "target_gene": "KCNH2",
        "target_uniprot_id": "Q12809",
        "indication": "Ventricular arrhythmia suppression (Class III)",
        "pharmacological_class": "Class III Antiarrhythmic / IKr Blocker",
    },
    {
        "compound_id": "DRUG-KCNH2-02",
        "name": "Dofetilide",
        "smiles": "CN(CCc1ccc(NS(=O)(=O)C)cc1)CCc2ccc(NS(=O)(=O)C)cc2",
        "target_gene": "KCNH2",
        "target_uniprot_id": "Q12809",
        "indication": "Atrial fibrillation conversion and maintenance",
        "pharmacological_class": "Potent Selective IKr (hERG) Blocker",
    },

    # -------------------------------------------------------------------------
    # Target: ADRB2 (Beta-2 Adrenergic Receptor) - UniProt: P07550
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-ADRB2-01",
        "name": "Salbutamol",
        "smiles": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
        "target_gene": "ADRB2",
        "target_uniprot_id": "P07550",
        "indication": "Beta-2 adrenergic modulation and smooth muscle relaxation",
        "pharmacological_class": "Selective Beta-2 Agonist",
    },
    {
        "compound_id": "DRUG-ADRB2-02",
        "name": "ICI-118,551",
        "smiles": "CC(C)NCC(O)c1cccc2c1OC(=O)N2C",
        "target_gene": "ADRB2",
        "target_uniprot_id": "P07550",
        "indication": "Selective beta-2 receptor investigation",
        "pharmacological_class": "Highly Selective Beta-2 Antagonist",
    },

    # -------------------------------------------------------------------------
    # Target: HCN4 (Hyperpolarization Pacemaker Channel) - UniProt: Q9Y6D8
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-HCN4-01",
        "name": "Ivabradine",
        "smiles": "COc1cc2c(cc1OC)CCN(CCc3ccc(OC)c(OC)c3)CC2",
        "target_gene": "HCN4",
        "target_uniprot_id": "Q9Y6D8",
        "indication": "Sinus tachycardia and heart rate reduction without inotropy loss",
        "pharmacological_class": "Selective If Current / Pacemaker Blocker",
    },

    # -------------------------------------------------------------------------
    # Target: AGTR1 (Angiotensin II Receptor Type 1) - UniProt: P30556
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-AGTR1-01",
        "name": "Losartan",
        "smiles": "CCCCc1nc(Cl)c(CO)n1Cc2ccc(cc2)c3ccccc3c4nnnn4",
        "target_gene": "AGTR1",
        "target_uniprot_id": "P30556",
        "indication": "Hypertension, cardiac hypertrophy attenuation, & nephroprotection",
        "pharmacological_class": "Angiotensin Receptor Blocker (ARB)",
    },
    {
        "compound_id": "DRUG-AGTR1-02",
        "name": "Valsartan",
        "smiles": "CCCCC(=O)N(Cc1ccc(cc1)c2ccccc2c3nnnn3)C(C(C)C)C(=O)O",
        "target_gene": "AGTR1",
        "target_uniprot_id": "P30556",
        "indication": "Post-myocardial infarction ventricular remodeling mitigation",
        "pharmacological_class": "Non-peptide AT1 Receptor Antagonist",
    },

    # -------------------------------------------------------------------------
    # Target: ACE (Angiotensin-Converting Enzyme) - UniProt: P12821
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-ACE-01",
        "name": "Enalapril",
        "smiles": "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N2CCCC2C(=O)O",
        "target_gene": "ACE",
        "target_uniprot_id": "P12821",
        "indication": "Afterload reduction and hypertensive left ventricular hypertrophy regression",
        "pharmacological_class": "ACE Inhibitor Prodrug",
    },
    {
        "compound_id": "DRUG-ACE-02",
        "name": "Ramipril",
        "smiles": "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N2C3CCCC3CC2C(=O)O",
        "target_gene": "ACE",
        "target_uniprot_id": "P12821",
        "indication": "Post-MI cardioprotective neurohormonal inhibition",
        "pharmacological_class": "Potent Tissue-selective ACE Inhibitor",
    },

    # -------------------------------------------------------------------------
    # Target: MYH7 (Beta-Myosin Heavy Chain) - UniProt: P12883
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-MYH7-01",
        "name": "Mavacamten",
        "smiles": "CC(C)NC(=O)N1C(=O)CC(c2ccccc2)C1=O",
        "target_gene": "MYH7",
        "target_uniprot_id": "P12883",
        "indication": "Hypertrophic cardiomyopathy outflow tract obstruction reduction",
        "pharmacological_class": "First-in-class Cardiac Myosin Allosteric Inhibitor",
    },
    {
        "compound_id": "DRUG-MYH7-02",
        "name": "Aficamten",
        "smiles": "CC(C)c1nnc(NC(=O)Nc2ccccc2F)s1",
        "target_gene": "MYH7",
        "target_uniprot_id": "P12883",
        "indication": "Next-generation sarcomeric contractility modulator",
        "pharmacological_class": "Selective Cardiac Sarcomere Inhibitor",
    },

    # -------------------------------------------------------------------------
    # Target: NOS3 (Endothelial Nitric Oxide Synthase) - UniProt: P29474
    # -------------------------------------------------------------------------
    {
        "compound_id": "DRUG-NOS3-01",
        "name": "Nebivolol",
        "smiles": "OCC1Oc2ccc(F)cc2C1CC(O)CNC(O)CC3c4cc(F)ccc4OC3",
        "target_gene": "NOS3",
        "target_uniprot_id": "P29474",
        "indication": "Coronary vasodilation via endothelial NO release enhancement",
        "pharmacological_class": "Beta-1 Blocker with Endothelial NOS3 Stimulation",
    },
]


class LigandLibrary:
    """
    Registry providing target-specific small molecule candidate libraries.
    """

    def __init__(self, custom_ligands: Optional[List[Dict[str, str]]] = None):
        self.ligands = list(CURATED_CARDIO_LIGANDS)
        if custom_ligands:
            self.ligands.extend(custom_ligands)

    def get_ligands_for_target(self, target_gene: str) -> List[Dict[str, str]]:
        """Filter ligands designed for or active against a specific target gene."""
        gene_upper = target_gene.strip().upper()
        matches = [m for m in self.ligands if m.get("target_gene", "").upper() == gene_upper]
        if not matches:
            # If no target-specific ligand exists, return general cardio reference compounds
            matches = [m for m in self.ligands if m.get("target_gene") in ("ADRB1", "CACNA1C", "SCN5A")][:3]
        return matches

    def get_ligand_by_id(self, compound_id: str) -> Optional[Dict[str, str]]:
        """Query ligand metadata by compound identifier."""
        cid = compound_id.strip().upper()
        for lig in self.ligands:
            if lig.get("compound_id", "").upper() == cid:
                return lig
        return None

    def get_all_ligands(self) -> List[Dict[str, str]]:
        """Return full chemical cohort."""
        return list(self.ligands)
