"""
Unit Tests for ECG Inference API (src/ecg_api.py).
Compliance Notice: Research use only, not for clinical decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import ecg_api
from pipeline.schemas import ECGInferenceResult, MANDATORY_DISCLAIMER


def test_infer_numpy_array():
    signal = np.sin(np.linspace(0, 10 * np.pi, 5000)).astype(np.float32)
    result = ecg_api.infer(signal)

    assert isinstance(result, ECGInferenceResult)
    assert len(result.probabilities) == 5
    assert set(result.probabilities.keys()) == {"NORM", "MI", "STTC", "CD", "HYP"}
    assert MANDATORY_DISCLAIMER in result.disclaimer
    assert all(0.0 <= p <= 1.0 for p in result.probabilities.values())


def test_infer_torch_tensor():
    signal = torch.randn(5000, dtype=torch.float32)
    result = ecg_api.infer(signal)

    assert isinstance(result, ECGInferenceResult)
    assert result.signal_length == 5000
    assert result.sampling_rate == 500


def test_infer_shape_handling():
    # Test shape (1, 5000)
    sig_2d = torch.zeros((1, 5000))
    res = ecg_api.infer(sig_2d)
    assert res.signal_length == 5000

    # Test invalid shape raises ValueError
    invalid = torch.zeros(4000)
    with pytest.raises(ValueError):
        ecg_api.infer(invalid)


def test_infer_unpacking_compatibility():
    signal = np.zeros(5000, dtype=np.float32)
    probs, preds = ecg_api.infer(signal)

    assert isinstance(probs, dict)
    assert isinstance(preds, dict)
    assert "MI" in probs
    assert "MI" in preds


def test_model_info():
    info = ecg_api.get_model_info()
    assert "model_class" in info
    assert "checkpoint_path" in info
    assert MANDATORY_DISCLAIMER in info["disclaimer"]
