import torch
import torch.nn as nn

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

# Probability threshold used to convert
# sigmoid probabilities into binary predictions.
THRESHOLD = 0.5


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
# Validation DataLoader
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
# Load BEST checkpoint
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

best_epoch = checkpoint["epoch"]

saved_validation_loss = (
    checkpoint["validation_loss"]
)

print()
print("=" * 70)
print("BEST MODEL LOADED")
print("=" * 70)

print(
    "Checkpoint:",
    best_model_path
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Saved validation loss:",
    f"{saved_validation_loss:.6f}"
)

print("=" * 70)


# ================================================================
# Loss function
# ================================================================

criterion = nn.BCEWithLogitsLoss()


# ================================================================
# Evaluation
# ================================================================

model.eval()

total_loss = 0.0

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
        labels = labels.to(device)

        # Forward pass
        outputs = model(signals)

        # Validation loss
        loss = criterion(
            outputs,
            labels
        )

        total_loss += loss.item()

        # Convert logits → probabilities
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
# Combine results
# ================================================================

all_labels = torch.cat(
    all_labels,
    dim=0
)

all_probabilities = torch.cat(
    all_probabilities,
    dim=0
)

average_loss = (
    total_loss / len(val_loader)
)


# ================================================================
# Convert probabilities → predictions
# ================================================================

all_predictions = (
    all_probabilities >= THRESHOLD
).float()


# ================================================================
# Calculate metrics
# ================================================================

print()
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(
    f"Validation loss: "
    f"{average_loss:.6f}"
)

print(
    f"Number of validation ECGs: "
    f"{len(all_labels)}"
)

print(
    f"Threshold: "
    f"{THRESHOLD}"
)

print()


# ---------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------

f1_scores = []

print(
    f"{'Class':<8}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print("-" * 44)

for class_index, class_name in enumerate(CLASSES):

    true_positive = (
        (
            (all_predictions[:, class_index] == 1)
            &
            (all_labels[:, class_index] == 1)
        )
        .sum()
        .item()
    )

    false_positive = (
        (
            (all_predictions[:, class_index] == 1)
            &
            (all_labels[:, class_index] == 0)
        )
        .sum()
        .item()
    )

    false_negative = (
        (
            (all_predictions[:, class_index] == 0)
            &
            (all_labels[:, class_index] == 1)
        )
        .sum()
        .item()
    )

    # Precision
    if true_positive + false_positive > 0:

        precision = (
            true_positive
            /
            (true_positive + false_positive)
        )

    else:

        precision = 0.0


    # Recall
    if true_positive + false_negative > 0:

        recall = (
            true_positive
            /
            (true_positive + false_negative)
        )

    else:

        recall = 0.0


    # F1
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


    f1_scores.append(f1)

    print(
        f"{class_name:<8}"
        f"{precision:>12.4f}"
        f"{recall:>12.4f}"
        f"{f1:>12.4f}"
    )


# ================================================================
# Macro F1
# ================================================================

macro_f1 = (
    sum(f1_scores)
    /
    len(f1_scores)
)

print()
print(
    f"Macro F1: {macro_f1:.4f}"
)


# ================================================================
# Example predictions
# ================================================================

print()
print("=" * 70)
print("EXAMPLE PREDICTIONS")
print("=" * 70)

num_examples = min(
    10,
    len(all_labels)
)

for index in range(num_examples):

    print()
    print(
        f"ECG {index + 1}"
    )

    actual = (
        all_labels[index]
        .tolist()
    )

    print(
        "Actual:",
        actual
    )

    print(
        "Predicted probabilities:"
    )

    for class_index, class_name in enumerate(
        CLASSES
    ):

        probability = (
            all_probabilities[
                index,
                class_index
            ]
            .item()
        )

        prediction = (
            "YES"
            if probability >= THRESHOLD
            else "NO"
        )

        print(
            f"  {class_name}: "
            f"{probability:.4f} "
            f"({prediction})"
        )


# ================================================================
# Complete
# ================================================================

print()
print("=" * 70)
print("1-LEAD MODEL EVALUATION COMPLETE")
print("=" * 70)