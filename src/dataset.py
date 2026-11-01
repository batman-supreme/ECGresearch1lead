import ast
from pathlib import Path

import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset, DataLoader


# ================================================================
# Dataset class
# ================================================================

class PTBXL_1Lead_Dataset(Dataset):
    """
    PyTorch Dataset for the 1-lead PTB-XL ECG project.

    We use ONLY Lead I from the original 12-lead ECG.

    Each item contains:

        ECG signal: shape (5000,)
        Labels:     shape (5,)

    Classes:

        0 = NORM
        1 = MI
        2 = STTC
        3 = CD
        4 = HYP
    """

    def __init__(self, metadata_path, data_root):

        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)

        # --------------------------------------------------------
        # Load metadata
        # --------------------------------------------------------

        self.df = pd.read_csv(self.metadata_path)

        # Convert SCP-code strings into Python dictionaries
        self.df["scp_codes"] = self.df["scp_codes"].apply(
            ast.literal_eval
        )

        # --------------------------------------------------------
        # Load SCP statements
        # --------------------------------------------------------

        scp_path = self.metadata_path.parent / "scp_statements.csv"

        self.scp_df = pd.read_csv(
            scp_path,
            index_col=0
        )

        # --------------------------------------------------------
        # Five major diagnostic classes
        # --------------------------------------------------------

        self.classes = [
            "NORM",
            "MI",
            "STTC",
            "CD",
            "HYP",
        ]

        # --------------------------------------------------------
        # Map SCP codes → diagnostic classes
        # --------------------------------------------------------

        diagnostic_codes = self.scp_df[
            self.scp_df["diagnostic_class"].notna()
        ]

        self.diagnostic_class_map = (
            diagnostic_codes["diagnostic_class"].to_dict()
        )

    # ============================================================
    # Dataset length
    # ============================================================

    def __len__(self):

        return len(self.df)

    # ============================================================
    # Get one ECG
    # ============================================================

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # --------------------------------------------------------
        # 1. Locate ECG file
        # --------------------------------------------------------

        record_path = (
            self.data_root / row["filename_hr"]
        )

        # WFDB reads the .dat/.hea pair
        record = wfdb.rdrecord(
            str(record_path)
        )

        # Original shape:
        #
        # (5000, 12)
        #
        # 5000 samples
        # 12 leads

        signal = record.p_signal

        # --------------------------------------------------------
        # 2. Extract ONLY Lead I
        # --------------------------------------------------------

        # Lead ordering:
        #
        # 0 = I
        # 1 = II
        # 2 = III
        # 3 = AVR
        # 4 = AVL
        # 5 = AVF
        # 6 = V1
        # 7 = V2
        # 8 = V3
        # 9 = V4
        # 10 = V5
        # 11 = V6

        lead_i = signal[:, 0]

        # --------------------------------------------------------
        # 3. Convert to PyTorch tensor
        # --------------------------------------------------------

        lead_i = torch.tensor(
            lead_i,
            dtype=torch.float32
        )

        # --------------------------------------------------------
        # 4. Convert SCP codes → diagnostic classes
        # --------------------------------------------------------

        scp_codes = row["scp_codes"]

        diagnostic_classes = set()

        for code in scp_codes:

            if code in self.diagnostic_class_map:

                diagnostic_class = (
                    self.diagnostic_class_map[code]
                )

                diagnostic_classes.add(
                    diagnostic_class
                )

        # --------------------------------------------------------
        # 5. Create multi-label target
        # --------------------------------------------------------

        labels = torch.zeros(
            len(self.classes),
            dtype=torch.float32
        )

        for diagnostic_class in diagnostic_classes:

            if diagnostic_class in self.classes:

                class_index = self.classes.index(
                    diagnostic_class
                )

                labels[class_index] = 1.0

        # --------------------------------------------------------
        # Return
        # --------------------------------------------------------

        return lead_i, labels


# ================================================================
# Dataset configuration
# ================================================================

DATASET_ROOT = (
    Path(r"D:\ECG-AI-Research\data\raw")
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

METADATA_PATH = (
    DATASET_ROOT / "ptbxl_database.csv"
)

BATCH_SIZE = 32


# ================================================================
# Create dataset
# ================================================================

dataset = PTBXL_1Lead_Dataset(
    metadata_path=METADATA_PATH,
    data_root=DATASET_ROOT,
)


# ================================================================
# DataLoader
# ================================================================

def create_loader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ================================================================
# Test dataset
# ================================================================

if __name__ == "__main__":

    print("Dataset size:", len(dataset))

    # ------------------------------------------------------------
    # Test one ECG
    # ------------------------------------------------------------

    signal, labels = dataset[0]

    print("Signal shape:", signal.shape)
    print("Signal dtype:", signal.dtype)
    print("Signal device:", signal.device)

    print("Labels:", labels)
    print("Labels shape:", labels.shape)

    # ------------------------------------------------------------
    # Test batch
    # ------------------------------------------------------------

    loader = create_loader(dataset)

    signals, labels = next(iter(loader))

    print(
        "Batch signal shape:",
        signals.shape
    )

    print(
        "Batch label shape:",
        labels.shape
    )

    print(
        "Signal dtype:",
        signals.dtype
    )

    print(
        "Label dtype:",
        labels.dtype
    )