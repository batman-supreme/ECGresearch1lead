"""
Extended Command-Line Pipeline Runner.

CRITICAL COMPLIANCE NOTICE:
Research use only, not for clinical decisions.

Wraps single-sample execution (PDF, NPY, CSV) with full cross-disciplinary
risk analysis, AlphaFold retrieval, and drug screening.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root and src are importable
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.orchestrator import run_pipeline
from pipeline.schemas import MANDATORY_DISCLAIMER


def main():
    parser = argparse.ArgumentParser(
        description="ECG-AI Multimodal Research Pipeline (CLI)",
        epilog=f"Compliance Notice: {MANDATORY_DISCLAIMER}",
    )
    parser.add_argument(
        "ecg_path",
        nargs="?",
        default=None,
        help="Path to ECG file (.pdf, .npy, .pt, or .csv). If omitted, runs synthetic demo.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="cli_run",
        help="Identifier tag for output audit run log.",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Disable Integrated Gradients explainability calculation.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=2,
        help="Maximum number of molecular targets to dock ligands against (default: 2).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ECG-AI RESEARCH PLATFORM: MULTIMODAL PIPELINE EXECUTION")
    print("=" * 80)
    print(f"NOTICE: {MANDATORY_DISCLAIMER}")
    print("=" * 80)

    if args.ecg_path:
        target_path = Path(args.ecg_path).resolve()
        if not target_path.exists():
            print(f"Error: File not found: {target_path}")
            sys.exit(1)
        ecg_input = target_path
        print(f"Input Source: {target_path}")
    else:
        import numpy as np
        print("Input Source: Synthetic Demo 1-Lead Waveform (5000 samples @ 500 Hz)")
        ecg_input = np.sin(np.linspace(0, 10 * np.pi, 5000)).astype(np.float32)

    print("\nExecuting orchestrator...")
    record = run_pipeline(
        ecg=ecg_input,
        run_name=args.run_name,
        include_explainability=not args.no_explain,
        max_targets_to_model=args.max_targets,
    )

    print(f"\n[DONE] Pipeline execution completed in {record.metadata.execution_duration_sec}s")
    print(f"Run ID: {record.metadata.run_id}")
    print(f"Saved run audit record to: results/runs/{record.metadata.run_id}.json\n")

    print("-" * 80)
    print("EXECUTIVE FINDINGS:")
    for finding in record.summary_findings:
        print(f" * {finding}")
    print("-" * 80)


if __name__ == "__main__":
    main()
