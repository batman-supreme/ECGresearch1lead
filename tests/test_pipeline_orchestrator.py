"""
Integration Tests for Pipeline Orchestrator and Logging (src/pipeline/).
Compliance Notice: Research use only, not for clinical decisions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pipeline.orchestrator import PipelineOrchestrator, run_pipeline
from pipeline.schemas import MANDATORY_DISCLAIMER, PipelineRunRecord


def test_end_to_end_orchestration(tmp_path: Path):
    orchestrator = PipelineOrchestrator(runs_dir=tmp_path)
    signal = np.sin(np.linspace(0, 10 * np.pi, 5000)).astype(np.float32)

    record = orchestrator.run(
        ecg=signal,
        run_name="test_orchestrator",
        include_explainability=True,
        max_targets_to_model=1,
    )

    assert isinstance(record, PipelineRunRecord)
    assert record.metadata.run_id.startswith("run_test_orchestrator_")
    assert record.metadata.execution_duration_sec > 0.0
    assert len(record.summary_findings) > 0
    assert MANDATORY_DISCLAIMER in record.disclaimer

    # Verify JSON persistence
    saved_file = tmp_path / f"{record.metadata.run_id}.json"
    assert saved_file.exists()
    assert saved_file.stat().st_size > 0

    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["run_id"] == record.metadata.run_id
    assert MANDATORY_DISCLAIMER in data["disclaimer"]
    assert "ecg_inference" in data
    assert "analytics" in data
    assert "bridge" in data


def test_convenience_run_pipeline():
    signal = np.zeros(5000, dtype=np.float32)
    record = run_pipeline(
        ecg=signal,
        run_name="quick_test",
        include_explainability=False,
        max_targets_to_model=1,
    )

    assert record.ecg_inference is not None
    assert record.analytics is not None
    assert MANDATORY_DISCLAIMER in record.disclaimer
