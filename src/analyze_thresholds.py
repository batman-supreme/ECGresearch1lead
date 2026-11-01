import torch

from pathlib import Path
from torch.utils.data import DataLoader, random_split

from dataset import PTBXL_1Lead_Dataset
from model import ECGCNN


# ================================================================
# Configuration
# ================================================================

BATCH_SIZE = 32
VAL_FRACTION = 0.10

CHECKPOINT_DIR = Path(
    r"D:\ECG-AI-Research-1Lead\checkpoints"
)

DATASET_ROOT = (
    Path(r"D:\ECG-AI-Research\data\raw")
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

METADATA_PATH = (
    DATASET_ROOT / "ptbxl_database.csv"
)

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
# Dataset
# ================================================================

full_dataset = PTBXL_1Lead_Dataset(
    metadata_path=METADATA_PATH,
    data_root=DATASET_ROOT,
)

print(
    "Full dataset size:",
    len(full_dataset)
)


# ================================================================
# Recreate EXACT validation split
# ================================================================

validation_size = int(
    len(full_dataset) * VAL_FRACTION
)

training_size = (
    len(full_dataset) - validation_size
)

_, val_dataset = random_split(
    full_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ================================================================
# DataLoader
# ================================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(
    "Validation batches:",
    len(val_loader)
)


# ================================================================
# Model
# ================================================================

model = ECGCNN().to(device)

print(
    "Model device:",
    next(model.parameters()).device
)


# ================================================================
# Load BEST MODEL
# ================================================================

best_model_path = (
    CHECKPOINT_DIR / "best_model.pt"
)

if not best_model_path.exists():

    raise FileNotFoundError(
        f"Could not find:\n{best_model_path}"
    )

checkpoint = torch.load(
    best_model_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("=" * 70)
print("BEST MODEL LOADED")
print("=" * 70)

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
# Collect validation predictions
# ================================================================

all_labels = []
all_probabilities = []

with torch.no_grad():

    for batch_index, (signals, labels) in enumerate(
        val_loader,
        start=1
    ):

        # Dataset:
        # [batch, 5000]
        #
        # CNN:
        # [batch, 1, 5000]

        signals = signals.unsqueeze(1)

        signals = signals.to(device)

        outputs = model(signals)

        probabilities = torch.sigmoid(
            outputs
        )

        all_labels.append(
            labels.cpu()
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        if batch_index % 20 == 0:

            print(
                f"Evaluated batch "
                f"{batch_index}/{len(val_loader)}"
            )


# ================================================================
# Combine predictions
# ================================================================

all_labels = torch.cat(
    all_labels,
    dim=0
)

all_probabilities = torch.cat(
    all_probabilities,
    dim=0
)

print()
print(
    "Collected predictions:",
    all_probabilities.shape
)


# ================================================================
# Metric calculation function
# ================================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).float()

    true_positive = (
        (
            (predictions == 1)
            &
            (labels == 1)
        )
        .sum()
        .item()
    )

    false_positive = (
        (
            (predictions == 1)
            &
            (labels == 0)
        )
        .sum()
        .item()
    )

    false_negative = (
        (
            (predictions == 0)
            &
            (labels == 1)
        )
        .sum()
        .item()
    )

    # ------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------

    if true_positive + false_positive > 0:

        precision = (
            true_positive
            /
            (true_positive + false_positive)
        )

    else:

        precision = 0.0


    # ------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------

    if true_positive + false_negative > 0:

        recall = (
            true_positive
            /
            (true_positive + false_negative)
        )

    else:

        recall = 0.0


    # ------------------------------------------------------------
    # F1
    # ------------------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            /
            (precision + recall)
        )

    else:

        f1 = 0.0


    return precision, recall, f1


# ================================================================
# Threshold search
# ================================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ================================================================
# Find best threshold for each class
# ================================================================

best_thresholds = {}

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

for class_index, class_name in enumerate(CLASSES):

    labels = all_labels[:, class_index]

    probabilities = (
        all_probabilities[:, class_index]
    )

    best_threshold = None
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0

    print()
    print(
        f"CLASS: {class_name}"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print(
        "-" * 48
    )

    for threshold in thresholds:

        precision, recall, f1 = calculate_metrics(
            labels,
            probabilities,
            threshold
        )

        print(
            f"{threshold:>10.2f}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{f1:>12.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall


    best_thresholds[class_name] = (
        best_threshold
    )

    print()
    print(
        f"BEST THRESHOLD FOR {class_name}: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Precision: {best_precision:.4f}"
    )

    print(
        f"Recall:    {best_recall:.4f}"
    )

    print(
        f"F1:        {best_f1:.4f}"
    )


# ================================================================
# Summary
# ================================================================

print()
print("=" * 70)
print("OPTIMAL THRESHOLD SUMMARY")
print("=" * 70)

for class_name in CLASSES:

    print(
        f"{class_name}: "
        f"{best_thresholds[class_name]:.2f}"
    )


# ================================================================
# Compare default vs optimized thresholds
# ================================================================

print()
print("=" * 70)
print("DEFAULT 0.50 VS OPTIMIZED THRESHOLDS")
print("=" * 70)

print(
    f"{'Class':<8}"
    f"{'F1 @ 0.50':>15}"
    f"{'Best threshold':>18}"
    f"{'Best F1':>12}"
)

print(
    "-" * 55
)

for class_index, class_name in enumerate(CLASSES):

    labels = all_labels[:, class_index]

    probabilities = (
        all_probabilities[:, class_index]
    )

    _, _, default_f1 = calculate_metrics(
        labels,
        probabilities,
        0.50
    )

    best_threshold = (
        best_thresholds[class_name]
    )

    _, _, best_f1 = calculate_metrics(
        labels,
        probabilities,
        best_threshold
    )

    print(
        f"{class_name:<8}"
        f"{default_f1:>15.4f}"
        f"{best_threshold:>18.2f}"
        f"{best_f1:>12.4f}"
    )


# ================================================================
# Complete
# ================================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)