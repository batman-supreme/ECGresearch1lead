import sys
from pathlib import Path

import torch

from model import ECGCNN
from samsung_pdf import extract_samsung_ecg


# ================================================================
# Configuration
# ================================================================

PROJECT_ROOT = Path(r"D:\ECG-AI-Research-1Lead")

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pt"
)

# Optimized thresholds obtained from validation analysis
THRESHOLDS = {
    "NORM": 0.50,
    "MI": 0.30,
    "STTC": 0.30,
    "CD": 0.35,
    "HYP": 0.25,
}

CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


# ================================================================
# Device
# ================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ================================================================
# Load model
# ================================================================

model = ECGCNN().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("=" * 70)
print("1-LEAD ECG AI PIPELINE")
print("=" * 70)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Validation loss:",
    f"{checkpoint['validation_loss']:.6f}"
)

print("=" * 70)


# ================================================================
# Analyze Samsung ECG
# ================================================================

def analyze_samsung_ecg(pdf_path):

    print()
    print("=" * 70)
    print("EXTRACTING SAMSUNG ECG")
    print("=" * 70)

    # ------------------------------------------------------------
    # Extract waveform
    # ------------------------------------------------------------

    signal = extract_samsung_ecg(
        pdf_path
    )

    print()
    print("Samsung signal ready.")
    print(
        "Signal shape:",
        signal.shape
    )

    print(
        "Signal dtype:",
        signal.dtype
    )

    # ------------------------------------------------------------
    # Verify signal
    # ------------------------------------------------------------

    if signal.shape != torch.Size([5000]):

        raise RuntimeError(
            "Samsung ECG signal must contain "
            "exactly 5000 samples."
        )

    # ------------------------------------------------------------
    # Add batch + channel dimensions
    #
    # Original:
    #
    # [5000]
    #
    # Model requires:
    #
    # [batch, channel, samples]
    #
    # [1, 1, 5000]
    # ------------------------------------------------------------

    model_input = signal.unsqueeze(0).unsqueeze(0)

    model_input = model_input.to(device)

    print()
    print(
        "Model input shape:",
        model_input.shape
    )

    print(
        "Model input device:",
        model_input.device
    )

    # ------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------

    with torch.no_grad():

        logits = model(
            model_input
        )

        probabilities = torch.sigmoid(
            logits
        )[0]

    # Move probabilities back to CPU
    probabilities = probabilities.cpu()

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMSUNG ECG AI ANALYSIS")
    print("=" * 70)

    print(
        f"{'CLASS':<8}"
        f"{'PROBABILITY':>14}"
        f"{'THRESHOLD':>13}"
        f"{'RESULT':>12}"
    )

    print("-" * 70)

    detected_classes = []

    for class_name, probability in zip(
        CLASSES,
        probabilities
    ):

        probability_value = (
            probability.item()
        )

        threshold = THRESHOLDS[
            class_name
        ]

        detected = (
            probability_value >= threshold
        )

        result = (
            "YES"
            if detected
            else "NO"
        )

        if detected:
            detected_classes.append(
                class_name
            )

        print(
            f"{class_name:<8}"
            f"{probability_value * 100:>13.2f}%"
            f"{threshold:>13.2f}"
            f"{result:>12}"
        )

    # ------------------------------------------------------------
    # Detected patterns
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DETECTED PATTERNS")
    print("=" * 70)

    if detected_classes:

        for class_name in detected_classes:

            print(
                f"✓ {class_name}"
            )

    else:

        print(
            "No class exceeded its threshold."
        )

    print("=" * 70)

    return probabilities, detected_classes


# ================================================================
# Command-line interface
# ================================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            r'python app.py "path\to\ecg.pdf"'
        )
        print()

        raise SystemExit(1)

    pdf_path = sys.argv[1]

    analyze_samsung_ecg(
        pdf_path
    )

    print()
    print("ECG analysis complete.")