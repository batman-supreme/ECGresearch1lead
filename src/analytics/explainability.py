"""
ECG Waveform Explainability Engine.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Implements Integrated Gradients attribution and physiological segment mapping
for 1-lead ECG deep convolutional networks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Ensure project root and src are importable
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import predict
from pipeline.schemas import (
    MANDATORY_DISCLAIMER,
    ExplainabilitySummary,
    WaveformSegmentAttribution,
)


class WaveformExplainer:
    """
    Computes Integrated Gradients attribution and physiological segment attributions
    for 1-lead ECG neural networks.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        sampling_rate: int = 500,
        n_steps: int = 50,
        baseline_mode: str = "zero",
    ):
        self.device = device or predict.DEVICE
        self.model = model or predict.model
        self.sampling_rate = sampling_rate
        self.n_steps = n_steps
        self.baseline_mode = baseline_mode
        self.classes = predict.CLASSES

    def _create_baseline(self, signal: torch.Tensor) -> torch.Tensor:
        """Create baseline reference tensor matching input shape."""
        if self.baseline_mode == "zero":
            return torch.zeros_like(signal)
        elif self.baseline_mode == "mean":
            return torch.full_like(signal, signal.mean().item())
        else:
            return torch.zeros_like(signal)

    def compute_integrated_gradients(
        self,
        signal: torch.Tensor,
        target_class_idx: int,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution for a specific target class.

        Formula:
            IG(x) = (x - x_0) * (1/m) * sum_{k=1}^m grad_F(x_0 + (k/m)*(x - x_0))

        Returns:
            np.ndarray: 1D attribution array of shape (5000,).
        """
        self.model.eval()

        # Format input as (1, 1, 5000)
        if signal.ndim == 1:
            input_tensor = signal.unsqueeze(0).unsqueeze(0).to(self.device).float()
        elif signal.ndim == 2:
            input_tensor = signal.unsqueeze(0).to(self.device).float()
        else:
            input_tensor = signal.to(self.device).float()

        baseline = self._create_baseline(input_tensor)

        # Scale alphas along batch dimension
        alphas = torch.linspace(0.0, 1.0, self.n_steps, device=self.device)
        
        # Interpolate: (n_steps, 1, 5000)
        # input_tensor is (1, 1, 5000), baseline is (1, 1, 5000)
        delta = input_tensor - baseline
        interpolated = baseline + alphas.view(-1, 1, 1) * delta  # (n_steps, 1, 5000)
        interpolated.requires_grad_(True)

        # Forward pass on batch of interpolated steps
        logits = self.model(interpolated)  # (n_steps, num_classes)
        target_logits = logits[:, target_class_idx].sum()

        # Compute gradients with respect to interpolated inputs
        grads = torch.autograd.grad(
            outputs=target_logits,
            inputs=interpolated,
            create_graph=False,
            retain_graph=False,
        )[0]  # (n_steps, 1, 5000)

        # Average gradients across steps and multiply by difference
        avg_grads = grads.mean(dim=0, keepdim=True)  # (1, 1, 5000)
        integrated_grads = (delta * avg_grads).squeeze().detach().cpu().numpy()

        return integrated_grads

    def detect_r_peaks(self, waveform: np.ndarray) -> np.ndarray:
        """
        Locate R-peak indices in a 500Hz Lead-I ECG waveform using pure NumPy.
        Uses distance and height thresholds calibrated for adult resting heart rate.
        """
        min_distance = int(0.4 * self.sampling_rate)  # 200 samples = 0.4s
        normalized = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-6)

        # Find local maxima where derivative changes from positive to negative
        dx = np.diff(normalized)
        candidate_peaks = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0] + 1

        if len(candidate_peaks) == 0:
            return np.array([], dtype=int)

        # Filter by amplitude threshold
        thresh = 0.8
        filtered = candidate_peaks[normalized[candidate_peaks] > thresh]

        if len(filtered) < 3:
            # Fallback for inverted or lower-amplitude leads
            thresh = 0.4
            filtered = candidate_peaks[np.abs(normalized[candidate_peaks]) > thresh]

        if len(filtered) <= 1:
            return filtered

        # Apply minimum distance suppression
        peaks: List[int] = [int(filtered[0])]
        for p in filtered[1:]:
            p_int = int(p)
            if p_int - peaks[-1] >= min_distance:
                peaks.append(p_int)
            elif normalized[p_int] > normalized[peaks[-1]]:
                peaks[-1] = p_int

        return np.array(peaks, dtype=int)

    def attribute_morphological_segments(
        self,
        waveform: np.ndarray,
        attribution: np.ndarray,
    ) -> List[WaveformSegmentAttribution]:
        """
        Partition attribution into physiological ECG intervals:
        - P Wave: Atrial depolarization (-200ms to -120ms relative to R)
        - PR Segment: AV node conduction delay (-120ms to -40ms)
        - QRS Complex: Ventricular depolarization (-40ms to +50ms)
        - ST Segment: Early ventricular repolarization (+50ms to +150ms)
        - T Wave: Rapid ventricular repolarization (+150ms to +350ms)
        - TP Segment: Resting electrical diastole
        """
        peaks = self.detect_r_peaks(waveform)
        n_samples = len(waveform)

        # Segment offsets relative to R-peak at 500 Hz:
        # 1 sample = 2 ms
        segment_defs = [
            ("P_wave", -100, -60, "Atrial depolarization; sensitive to atrial overload and P-mitrale"),
            ("PR_interval", -60, -20, "AV nodal conduction delay; sensitive to heart blocks"),
            ("QRS_complex", -20, 25, "Ventricular depolarization; hallmark of bundle branch blocks & LVH"),
            ("ST_segment", 25, 75, "Ventricular plateau phase; key indicator of acute ischemia/infarction"),
            ("T_wave", 75, 175, "Ventricular repolarization; sensitive to ischemia & potassium channelopathies"),
        ]

        abs_attrib = np.abs(attribution)
        total_energy = float(np.sum(abs_attrib)) + 1e-9

        segment_energies: Dict[str, float] = {name: 0.0 for name, _, _, _ in segment_defs}
        segment_energies["TP_segment"] = 0.0

        sample_assigned = np.zeros(n_samples, dtype=bool)

        if len(peaks) > 0:
            for r in peaks:
                for name, start_off, end_off, _ in segment_defs:
                    s_idx = max(0, r + start_off)
                    e_idx = min(n_samples, r + end_off)
                    if e_idx > s_idx:
                        energy = float(np.sum(abs_attrib[s_idx:e_idx]))
                        segment_energies[name] += energy
                        sample_assigned[s_idx:e_idx] = True

            # Unassigned baseline samples go into TP_segment
            tp_energy = float(np.sum(abs_attrib[~sample_assigned]))
            segment_energies["TP_segment"] = tp_energy
        else:
            # Fallback uniform division if no distinct peaks could be detected
            tenth = n_samples // 5
            segment_energies["P_wave"] = float(np.sum(abs_attrib[0:tenth]))
            segment_energies["PR_interval"] = float(np.sum(abs_attrib[tenth:2*tenth]))
            segment_energies["QRS_complex"] = float(np.sum(abs_attrib[2*tenth:3*tenth]))
            segment_energies["ST_segment"] = float(np.sum(abs_attrib[3*tenth:4*tenth]))
            segment_energies["T_wave"] = float(np.sum(abs_attrib[4*tenth:n_samples]))
            segment_energies["TP_segment"] = 0.0

        # Construct segment attribution objects
        attributions: List[WaveformSegmentAttribution] = []
        for name, s_off, e_off, correlate in segment_defs:
            energy = segment_energies[name]
            pct = (energy / total_energy) * 100.0
            first_r = peaks[0] if len(peaks) > 0 else 1000
            s_idx = max(0, first_r + s_off)
            e_idx = min(n_samples, first_r + e_off)

            attributions.append(
                WaveformSegmentAttribution(
                    segment_name=name,
                    start_sample=int(s_idx),
                    end_sample=int(e_idx),
                    attribution_score=round(energy, 5),
                    relative_contribution_pct=round(pct, 2),
                    clinical_correlate=correlate,
                    disclaimer=MANDATORY_DISCLAIMER,
                )
            )

        # Add TP segment
        tp_energy = segment_energies["TP_segment"]
        tp_pct = (tp_energy / total_energy) * 100.0
        attributions.append(
            WaveformSegmentAttribution(
                segment_name="TP_baseline",
                start_sample=0,
                end_sample=n_samples,
                attribution_score=round(tp_energy, 5),
                relative_contribution_pct=round(tp_pct, 2),
                clinical_correlate="Electrical diastole and isoelectric baseline stability",
                disclaimer=MANDATORY_DISCLAIMER,
            )
        )

        # Sort segments by highest attribution percentage descending
        attributions.sort(key=lambda x: x.relative_contribution_pct, reverse=True)
        return attributions

    def extract_top_salient_intervals(
        self,
        attribution: np.ndarray,
        top_k: int = 5,
        window_size: int = 100,
    ) -> List[Tuple[int, int]]:
        """
        Locate window_size intervals with highest integrated gradient energy.
        """
        abs_attrib = np.abs(attribution)
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(abs_attrib, kernel, mode="valid")

        intervals: List[Tuple[int, int]] = []
        used_mask = np.zeros(len(smoothed), dtype=bool)

        for _ in range(top_k):
            available = np.where(~used_mask)[0]
            if len(available) == 0:
                break
            best_idx = available[np.argmax(smoothed[available])]
            start = int(best_idx)
            end = int(min(len(attribution), best_idx + window_size))
            intervals.append((start, end))

            # Suppress nearby indices to ensure diverse salient regions
            suppress_start = max(0, best_idx - window_size // 2)
            suppress_end = min(len(used_mask), best_idx + window_size // 2)
            used_mask[suppress_start:suppress_end] = True

        return intervals

    def explain(
        self,
        signal: Union[torch.Tensor, np.ndarray],
        target_condition: Optional[str] = None,
        detected_conditions: Optional[List[str]] = None,
    ) -> ExplainabilitySummary:
        """
        Execute full explainability workflow for an ECG recording.
        """
        if isinstance(signal, np.ndarray):
            signal_tensor = torch.from_numpy(signal).float()
            signal_np = signal.squeeze()
        else:
            signal_tensor = signal.float()
            signal_np = signal.detach().cpu().numpy().squeeze()

        # Determine target class index
        if target_condition and target_condition in self.classes:
            target_class_idx = self.classes.index(target_condition)
        elif detected_conditions and len(detected_conditions) > 0:
            # Pick first non-NORM pathology if available
            pathologies = [c for c in detected_conditions if c != "NORM"]
            chosen = pathologies[0] if pathologies else detected_conditions[0]
            target_class_idx = self.classes.index(chosen)
        else:
            # Default to MI (most clinically acute class)
            target_class_idx = self.classes.index("MI")

        # 1. Compute Integrated Gradients
        attribution = self.compute_integrated_gradients(
            signal=signal_tensor,
            target_class_idx=target_class_idx,
        )

        total_energy = float(np.sum(np.abs(attribution)))

        # 2. Extract top salient intervals
        top_intervals = self.extract_top_salient_intervals(
            attribution=attribution,
            top_k=5,
            window_size=int(0.2 * self.sampling_rate),  # 200ms windows
        )

        # 3. Morphological segment attribution
        segment_attributions = self.attribute_morphological_segments(
            waveform=signal_np,
            attribution=attribution,
        )

        return ExplainabilitySummary(
            method="integrated_gradients",
            baseline=self.baseline_mode,
            saliency_energy=round(total_energy, 4),
            top_salient_intervals=top_intervals,
            segment_attributions=segment_attributions,
            disclaimer=MANDATORY_DISCLAIMER,
        )
