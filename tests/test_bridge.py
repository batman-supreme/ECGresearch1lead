"""
Unit Tests for Target Bridge Agent (src/bridge/).
Compliance Notice: Research use only, not for clinical decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bridge.mapper import TargetBridge, resolve_targets
from pipeline.schemas import BridgeResult, MANDATORY_DISCLAIMER


def test_resolve_targets_mi():
    bridge = TargetBridge()
    result = bridge.resolve_targets(["MI"])

    assert isinstance(result, BridgeResult)
    assert len(result.resolved_targets) > 0
    genes = [t.gene for t in result.resolved_targets]
    assert "ADRB1" in genes
    assert "P08588" in result.uniprot_ids
    assert MANDATORY_DISCLAIMER in result.disclaimer


def test_target_deduplication():
    # SCN5A is in both MI and CD; verify it appears exactly once
    result = resolve_targets(["MI", "CD"])
    genes = [t.gene for t in result.resolved_targets]
    assert genes.count("SCN5A") == 1


def test_invalid_condition_rejection():
    # Bridge must never hallucinate mappings for non-existent conditions
    result = resolve_targets(["FAKE_UNKNOWN_DISEASE_123"])
    assert len(result.resolved_targets) == 0


def test_uniprot_regex_validation():
    bridge = TargetBridge()
    assert bridge.is_valid_uniprot_id("P08588") is True
    assert bridge.is_valid_uniprot_id("Q14524") is True
    assert bridge.is_valid_uniprot_id("INVALID_ID_99999") is False
