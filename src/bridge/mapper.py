"""
Condition-to-Gene-to-UniProt Target Bridge.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

STRICT RULE:
All target mappings are derived strictly from configs/targets.yaml.
Never invent, fabricate, or hallucinate target associations.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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
    BridgeResult,
    ConditionTargetMapping,
    TargetGene,
)

_DEFAULT_TARGETS_YAML = _PROJECT_ROOT / "configs" / "targets.yaml"

# Standard UniProtKB accession regex pattern
_UNIPROT_REGEX = re.compile(r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")


class TargetBridge:
    """
    Authoritative mapper connecting diagnosed ECG conditions to validated
    molecular targets using configs/targets.yaml.
    """

    def __init__(self, targets_yaml_path: Optional[Union[str, Path]] = None):
        self.targets_yaml_path = Path(targets_yaml_path or _DEFAULT_TARGETS_YAML).resolve()
        if not self.targets_yaml_path.exists():
            raise FileNotFoundError(
                f"Required target mapping configuration not found at {self.targets_yaml_path}. "
                f"Targets must be explicitly registered in configs/targets.yaml."
            )
        self._mappings: Dict[str, ConditionTargetMapping] = {}
        self._gene_index: Dict[str, TargetGene] = {}
        self._uniprot_index: Dict[str, TargetGene] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Parse configs/targets.yaml and build index mappings."""
        with open(self.targets_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        conditions_dict = data.get("conditions", {})
        if not conditions_dict:
            raise ValueError(f"No conditions found in {self.targets_yaml_path}")

        for cond_key, cond_info in conditions_dict.items():
            raw_targets = cond_info.get("targets", [])
            target_objects: List[TargetGene] = []

            for t in raw_targets:
                u_id = str(t.get("uniprot_id", "")).strip().upper()
                gene = str(t.get("gene", "")).strip().upper()

                if not self.is_valid_uniprot_id(u_id):
                    raise ValueError(
                        f"Invalid UniProt accession '{u_id}' registered for gene '{gene}' in {cond_key}."
                    )

                target_gene = TargetGene(
                    gene=gene,
                    uniprot_id=u_id,
                    name=t.get("name", ""),
                    organism=t.get("organism", "Homo sapiens"),
                    role=t.get("role", ""),
                    pathway=t.get("pathway", ""),
                    priority=int(t.get("priority", 1)),
                    disclaimer=MANDATORY_DISCLAIMER,
                )
                target_objects.append(target_gene)

                # Update global indexes
                self._gene_index[gene] = target_gene
                self._uniprot_index[u_id] = target_gene

            # Sort condition targets by priority
            target_objects.sort(key=lambda x: x.priority)

            self._mappings[cond_key] = ConditionTargetMapping(
                condition=cond_key,
                name=cond_info.get("name", cond_key),
                description=cond_info.get("description", ""),
                is_pathology=bool(cond_info.get("is_pathology", True)),
                targets=target_objects,
                disclaimer=MANDATORY_DISCLAIMER,
            )

    @staticmethod
    def is_valid_uniprot_id(uniprot_id: str) -> bool:
        """Validate string conforms to official UniProtKB accession format."""
        return bool(_UNIPROT_REGEX.match(uniprot_id.strip().upper()))

    def get_registered_conditions(self) -> List[str]:
        """Return list of valid conditions defined in the registry."""
        return list(self._mappings.keys())

    def get_condition_mapping(self, condition: str) -> Optional[ConditionTargetMapping]:
        """Retrieve full mapping object for a condition."""
        return self._mappings.get(condition.strip().upper())

    def resolve_targets(
        self,
        conditions: Sequence[str],
        max_targets_per_condition: Optional[int] = None,
    ) -> BridgeResult:
        """
        Map a list of detected ECG conditions to validated molecular targets.

        RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.

        Rules:
        - Only targets explicitly defined in configs/targets.yaml are returned.
        - Never invents, guesses, or fabricates mappings.
        - Deduplicates shared targets (e.g. SCN5A implicated in both MI and CD).
        """
        queried: List[str] = [c.strip().upper() for c in conditions]
        resolved: List[TargetGene] = []
        seen_uniprots: set[str] = set()

        for cond in queried:
            mapping = self._mappings.get(cond)
            if mapping is None:
                # Strictly reject unknown conditions without guessing
                continue

            cond_targets = mapping.targets
            if max_targets_per_condition is not None:
                cond_targets = cond_targets[:max_targets_per_condition]

            for target in cond_targets:
                if target.uniprot_id not in seen_uniprots:
                    seen_uniprots.add(target.uniprot_id)
                    resolved.append(target)

        # Sort combined targets by priority
        resolved.sort(key=lambda x: x.priority)
        uniprot_ids = [t.uniprot_id for t in resolved]

        return BridgeResult(
            queried_conditions=queried,
            resolved_targets=resolved,
            uniprot_ids=uniprot_ids,
            source_file=str(self.targets_yaml_path),
            disclaimer=MANDATORY_DISCLAIMER,
        )

    def get_target_by_gene(self, gene_symbol: str) -> Optional[TargetGene]:
        """Query target metadata by official HGNC gene symbol."""
        return self._gene_index.get(gene_symbol.strip().upper())

    def get_target_by_uniprot(self, uniprot_id: str) -> Optional[TargetGene]:
        """Query target metadata by UniProt accession."""
        return self._uniprot_index.get(uniprot_id.strip().upper())


# Module-level convenience function
_DEFAULT_BRIDGE: Optional[TargetBridge] = None


def resolve_targets(
    conditions: Sequence[str],
    targets_yaml_path: Optional[Union[str, Path]] = None,
    max_targets_per_condition: Optional[int] = None,
) -> BridgeResult:
    """
    Convenience function mapping condition labels to validated target genes.
    """
    global _DEFAULT_BRIDGE
    if targets_yaml_path or _DEFAULT_BRIDGE is None:
        bridge = TargetBridge(targets_yaml_path=targets_yaml_path)
        if targets_yaml_path is None:
            _DEFAULT_BRIDGE = bridge
    else:
        bridge = _DEFAULT_BRIDGE

    return bridge.resolve_targets(
        conditions=conditions,
        max_targets_per_condition=max_targets_per_condition,
    )
