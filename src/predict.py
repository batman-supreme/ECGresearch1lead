import sys
from pathlib import Path

import torch

from model import ECGCNN


# ================================================================
# Configuration
# ================================================================

CHECKPOINT_PATH = Path(
    r"D:\ECG-AI-Research-1Lead\checkpoints\best_model.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# Classes
# ================================================================

CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


# ================================================================
# Optimized thresholds
# ================================================================

THRESHOLDS = {
    "NORM": 0.50,
    "MI": 0.30,
    "STTC": 0.30,
    "CD": 0.35,
    "HYP": 0.25,
}


# ================================================================
# Device information
# ================================================================

print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ================================================================
# Load model
# ================================================================

model = ECGCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("=" * 70)
print("1-LEAD ECG MODEL LOADED")
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
# Prediction function
# ================================================================

def predict_ecg(signal):
    """
    Analyze one Lead-I ECG signal.

    Expected input:
        signal -> torch.Tensor with shape (5000,)

    Returns:
        probabilities -> tensor with shape (5,)
        predictions   -> dictionary of class predictions
    """

    # ------------------------------------------------------------
    # Check input shape
    # ------------------------------------------------------------

    if signal.shape != (5000,):

        raise ValueError(
            f"Expected ECG shape (5000,), "
            f"but received {tuple(signal.shape)}"
        )


    # ------------------------------------------------------------
    # Convert to float32
    # ------------------------------------------------------------

    signal = signal.to(
        dtype=torch.float32
    )


    # ------------------------------------------------------------
    # Add channel dimension
    #
    # (5000,)
    #
    # →
    #
    # (1, 5000)
    # ------------------------------------------------------------

    signal = signal.unsqueeze(0)


    # ------------------------------------------------------------
    # Add batch dimension
    #
    # (1, 5000)
    #
    # →
    #
    # (1, 1, 5000)
    # ------------------------------------------------------------

    signal = signal.unsqueeze(0)


    # ------------------------------------------------------------
    # Move to GPU / CPU
    # ------------------------------------------------------------

    signal = signal.to(DEVICE)


    # ------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------

    with torch.no_grad():

        logits = model(signal)

        probabilities = torch.sigmoid(
            logits
        )


    # ------------------------------------------------------------
    # Remove batch dimension
    #
    # (1, 5) → (5,)
    # ------------------------------------------------------------

    probabilities = probabilities.squeeze(0)


    # ------------------------------------------------------------
    # Apply class-specific thresholds
    # ------------------------------------------------------------

    predictions = {}

    for index, class_name in enumerate(CLASSES):

        probability = probabilities[index].item()

        threshold = THRESHOLDS[class_name]

        predictions[class_name] = (
            probability >= threshold
        )


    return probabilities, predictions


# ================================================================
# Test with a real ECG from the dataset
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RUNNING TEST ECG")
    print("=" * 70)


    # ------------------------------------------------------------
    # Import dataset
    # ------------------------------------------------------------

    from dataset import PTBXL_1Lead_Dataset

    DATASET_ROOT = (
        Path(r"D:\ECG-AI-Research\data\raw")
        / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )

    METADATA_PATH = (
        DATASET_ROOT / "ptbxl_database.csv"
    )


    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    dataset = PTBXL_1Lead_Dataset(
        metadata_path=METADATA_PATH,
        data_root=DATASET_ROOT,
    )


    print(
        "Dataset size:",
        len(dataset)
    )


    # ------------------------------------------------------------
    # Select ECG
    # ------------------------------------------------------------

    index = 0

    signal, actual_labels = dataset[index]


    print(
        "ECG index:",
        index
    )

    print(
        "Signal shape:",
        signal.shape
    )

    print(
        "Actual labels:",
        actual_labels.tolist()
    )


    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    probabilities, predictions = predict_ecg(
        signal
    )


    # ------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("ECG ANALYSIS RESULT")
    print("=" * 70)


    for index, class_name in enumerate(CLASSES):

        probability = (
            probabilities[index].item()
        )

        threshold = THRESHOLDS[class_name]

        result = predictions[class_name]

        status = "YES" if result else "NO"

        print(
            f"{class_name:5s} | "
            f"Probability: {probability:.4f} | "
            f"Threshold: {threshold:.2f} | "
            f"{status}"
        )


    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)