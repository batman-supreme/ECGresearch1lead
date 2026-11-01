"""
ECG API Module for ECG-AI-Research-1Lead.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Provides a unified programmatic inference API wrapping the existing
predict.py model without altering existing model weights or logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml

# Ensure project root and src directory are on sys.path
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Wrap existing predict module without modifying it
import predict
from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ECGInferenceResult,
    ECGConditionPrediction,
)

# Load pipeline configuration if available
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_ecg_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from configs/pipeline.yaml or return defaults."""
    target_path = Path(config_path) if config_path else _CONFIG_PATH
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "disclaimer": MANDATORY_DISCLAIMER,
        "ecg": {
            "sampling_rate": 500,
            "signal_length": 5000,
            "classes": predict.CLASSES,
            "thresholds": predict.THRESHOLDS,
        },
    }


def _prepare_signal_tensor(ecg: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> torch.Tensor:
    """
    Validate and convert arbitrary input into a 1D torch.Tensor of shape (5000,).
    
    Accepts:
        - torch.Tensor: shape (5000,), (1, 5000), or (1, 1, 5000)
        - np.ndarray: shape (5000,), (1, 5000), or (1, 1, 5000)
        - Sequence/list of floats of length 5000
    """
    if isinstance(ecg, np.ndarray):
        tensor = torch.from_numpy(ecg).float()
    elif isinstance(ecg, torch.Tensor):
        tensor = ecg.float()
    elif isinstance(ecg, (list, tuple)):
        tensor = torch.tensor(ecg, dtype=torch.float32)
    else:
        raise TypeError(
            f"Unsupported ECG input type: {type(ecg)}. "
            f"Expected torch.Tensor, np.ndarray, or list of floats."
        )

    # Squeeze out any unnecessary singleton dimensions (e.g. (1, 5000) -> (5000,))
    tensor = tensor.squeeze()

    if tensor.ndim != 1 or tensor.shape[0] != 5000:
        raise ValueError(
            f"Expected 1-lead ECG signal with exactly 5000 samples, "
            f"but received tensor with shape {tuple(tensor.shape)}."
        )

    return tensor


def infer(
    ecg: Union[torch.Tensor, np.ndarray, Sequence[float]],
    config_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ECGInferenceResult:
    """
    Run single-lead ECG inference by wrapping existing predict.predict_ecg.

    RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.

    Parameters:
        ecg: Waveform data with 5000 samples (torch.Tensor, numpy.ndarray, or list).
        config_path: Optional path to override pipeline config yaml.
        metadata: Optional dictionary with provenance details.

    Returns:
        ECGInferenceResult: Pydantic schema containing probabilities, thresholded
        predictions, detected conditions, device, and mandatory disclaimer.
        Can be unpacked directly via `probabilities, predictions = infer(ecg)`.
    """
    # 1. Prepare tensor conforming strictly to predict_ecg expectations
    signal_tensor = _prepare_signal_tensor(ecg)

    # 2. Call existing predict.py logic directly without altering its behavior
    raw_probabilities_tensor, raw_predictions_dict = predict.predict_ecg(signal_tensor)

    # 3. Format probabilities mapping
    classes: List[str] = predict.CLASSES
    thresholds: Dict[str, float] = predict.THRESHOLDS

    # Load configured thresholds if available
    cfg = load_ecg_config(config_path)
    if "ecg" in cfg and "thresholds" in cfg["ecg"]:
        thresholds = cfg["ecg"]["thresholds"]

    probabilities: Dict[str, float] = {}
    predictions: Dict[str, bool] = {}
    detected_conditions: List[str] = []

    for idx, class_name in enumerate(classes):
        prob = float(raw_probabilities_tensor[idx].item())
        thresh = float(thresholds.get(class_name, predict.THRESHOLDS.get(class_name, 0.5)))
        detected = prob >= thresh

        probabilities[class_name] = round(prob, 5)
        predictions[class_name] = detected

        if detected:
            detected_conditions.append(class_name)

    # 4. Construct validated Pydantic result container
    result = ECGInferenceResult(
        probabilities=probabilities,
        predictions=predictions,
        detected_conditions=detected_conditions,
        thresholds=thresholds,
        sampling_rate=int(cfg.get("ecg", {}).get("sampling_rate", 500)),
        signal_length=5000,
        device=str(predict.DEVICE),
        disclaimer=MANDATORY_DISCLAIMER,
        metadata=metadata or {},
    )

    return result


def load_ecg_csv(file_path: Union[str, Path], target_length: int = 5000) -> np.ndarray:
    """
    Intelligently ingest arbitrary ECG CSV/TSV/TXT files.

    Supports:
        - Single column with or without header (e.g. Lead_I, voltage, ecg, amplitude)
        - Multi-column CSVs (automatically identifies Lead-I / voltage column vs time)
        - Single row of comma/semicolon/tab-separated numeric values
        - Flexible length: interpolates / resamples or trims to target_length (default 5000).

    RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")

    signal = None

    # 1. Try reading with pandas for robust tabular parsing
    try:
        import pandas as pd
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path)

        # Handle row-vector format (1 row, many columns)
        if df.shape[0] == 1 and df.shape[1] >= 100:
            df = df.T

        if df.shape[1] == 1:
            signal = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
        else:
            # Check for standard Lead-I column names
            candidates = ["lead_1", "lead_i", "leadi", "lead1", "ecg", "signal", "voltage", "mv", "amplitude", "val"]
            found_col = None
            for col in df.columns:
                clean_col = str(col).strip().lower().replace(" ", "_").replace("-", "_")
                if any(c in clean_col for c in candidates):
                    found_col = col
                    break

            if found_col is not None:
                signal = pd.to_numeric(df[found_col], errors="coerce").dropna().values
            else:
                # Find all numeric columns with >= 100 values
                numeric_cols = []
                for col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(s) > 100:
                        numeric_cols.append((col, s.values))

                if len(numeric_cols) >= 2:
                    # If column 0 is strictly increasing time/timestamp, pick column 1 as voltage
                    v0 = numeric_cols[0][1]
                    if len(v0) > 10 and np.all(np.diff(v0[:100]) >= 0):
                        signal = numeric_cols[1][1]
                    else:
                        signal = numeric_cols[0][1]
                elif len(numeric_cols) == 1:
                    signal = numeric_cols[0][1]
    except Exception:
        signal = None

    # 2. Fallback to NumPy loadtxt if pandas parsing failed
    if signal is None or len(signal) == 0:
        try:
            signal = np.loadtxt(path, delimiter=",").flatten()
        except Exception:
            try:
                signal = np.loadtxt(path).flatten()
            except Exception as err:
                raise ValueError(f"Could not parse numeric ECG signal from '{path}': {err}")

    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) == 0:
        raise ValueError(f"No numeric data found in ECG file: {path}")

    # 3. Resample or interpolate to target_length (5000 samples @ 500 Hz)
    if len(signal) != target_length:
        if len(signal) > target_length and len(signal) < target_length * 1.5:
            signal = signal[:target_length]
        else:
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, target_length)
            signal = np.interp(x_new, x_old, signal).astype(np.float32)

    return signal


def infer_from_file(
    file_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> ECGInferenceResult:
    """
    Helper function to load an ECG waveform from disk and execute inference.

    Supports:
        - .npy: NumPy 1D array
        - .csv / .tsv / .txt: Single column, multi-column (Lead-I extraction), or row vectors
        - .pt: PyTorch tensor
        - .pdf: Samsung Health ECG PDF export (using samsung_pdf extractor)

    RESEARCH USE ONLY. NOT FOR CLINICAL DECISIONS.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")

    ext = path.suffix.lower()
    meta = {"source_file": str(path), "format": ext}

    if ext == ".npy":
        raw = np.load(path)
        return infer(raw, config_path=config_path, metadata=meta)

    elif ext == ".pt":
        raw = torch.load(path, map_location="cpu")
        return infer(raw, config_path=config_path, metadata=meta)

    elif ext in (".csv", ".tsv", ".txt"):
        raw = load_ecg_csv(path)
        meta["device_type"] = "ECG CSV / Tabular Record"
        return infer(raw, config_path=config_path, metadata=meta)

    elif ext == ".pdf":
        try:
            from samsung_pdf import extract_samsung_ecg
            signal = extract_samsung_ecg(path)
            meta["device_type"] = "Samsung Health ECG PDF"
            return infer(signal, config_path=config_path, metadata=meta)
        except ImportError as e:
            raise RuntimeError(f"Samsung PDF extractor unavailable: {e}")

    else:
        raise ValueError(f"Unsupported file format: {ext}. Expected .csv, .pdf, .npy, .pt, or .txt")


def get_model_info() -> Dict[str, Any]:
    """Retrieve runtime model metadata and compliance notice."""
    return {
        "model_class": predict.ECGCNN.__name__,
        "checkpoint_path": str(predict.CHECKPOINT_PATH),
        "device": str(predict.DEVICE),
        "classes": list(predict.CLASSES),
        "thresholds": dict(predict.THRESHOLDS),
        "disclaimer": MANDATORY_DISCLAIMER,
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING ECG API (ecg_api.py)")
    print("=" * 70)

    # 1. Test model info
    info = get_model_info()
    print("Model Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 2. Test synthetic signal (5000 zeros / baseline)
    print("\nRunning inference on baseline 5000-sample signal...")
    dummy_signal = np.sin(np.linspace(0, 10 * np.pi, 5000)).astype(np.float32)
    result = infer(dummy_signal)

    print("\nInference Result:")
    print("Probabilities:", result.probabilities)
    print("Predictions:  ", result.predictions)
    print("Detected:     ", result.detected_conditions)
    print("Is Normal:    ", result.is_normal)
    print("Disclaimer:   ", result.disclaimer)

    # 3. Test tuple unpacking backwards compatibility
    probs, preds = result
    assert isinstance(probs, dict)
    assert isinstance(preds, dict)
    print("\nUnpacking test (probabilities, predictions = infer(ecg)): PASSED")

    print("\n" + "=" * 70)
    print("ECG API VERIFICATION COMPLETE")
    print("=" * 70)
