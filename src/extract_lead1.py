from pathlib import Path

import torch
import wfdb


# ================================================================
# Configuration
# ================================================================

DATASET_ROOT = (
    Path(r"D:\ECG-AI-Research\data\raw")
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# We'll test with ECG #1 first.
ECG_ID = 1


# ================================================================
# Find the ECG
# ================================================================

record_path = (
    DATASET_ROOT
    / "records500"
    / "00000"
    / "00001_hr"
)


# ================================================================
# Read ECG
# ================================================================

record = wfdb.rdrecord(str(record_path))

signal = record.p_signal


# ================================================================
# Display original ECG information
# ================================================================

print("Original ECG shape:", signal.shape)
print("Sampling frequency:", record.fs)
print("Number of leads:", signal.shape[1])
print("Lead names:", record.sig_name)


# ================================================================
# Extract Lead I
# ================================================================

lead_i = signal[:, 0]


# ================================================================
# Convert to PyTorch tensor
# ================================================================

lead_i = torch.tensor(
    lead_i,
    dtype=torch.float32
)


# ================================================================
# Display Lead I information
# ================================================================

print()
print("Lead I shape:", lead_i.shape)
print("Lead I dtype:", lead_i.dtype)
print("First 10 Lead I samples:")
print(lead_i[:10])