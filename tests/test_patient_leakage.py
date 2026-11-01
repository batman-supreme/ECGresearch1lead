"""
Zero Patient Leakage Verification Tests.
Compliance Notice: Research use only, not for clinical decisions.

Rule: "Patient-wise splits only; no leakage."
Verifies that training, validation, and test cohorts share ZERO overlapping patients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def create_patient_wise_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition dataframe strictly by unique patient IDs to prevent data leakage.
    """
    rng = np.random.RandomState(random_seed)
    unique_patients = np.array(list(df[patient_col].unique()))
    rng.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)

    val_patients = set(unique_patients[:n_val])
    test_patients = set(unique_patients[n_val : n_val + n_test])
    train_patients = set(unique_patients[n_val + n_test :])

    train_df = df[df[patient_col].isin(train_patients)].copy()
    val_df = df[df[patient_col].isin(val_patients)].copy()
    test_df = df[df[patient_col].isin(test_patients)].copy()

    return train_df, val_df, test_df


def test_zero_patient_leakage():
    # Simulate a cohort with multiple ECG recordings per patient
    np.random.seed(42)
    n_records = 1000
    n_patients = 200  # Multiple records per patient
    patient_ids = np.random.choice([f"PT_{i:04d}" for i in range(n_patients)], size=n_records)
    ecg_ids = [f"ECG_{i:06d}" for i in range(n_records)]

    mock_df = pd.DataFrame({
        "ecg_id": ecg_ids,
        "patient_id": patient_ids,
        "heart_rate": np.random.randint(55, 110, size=n_records),
    })

    train_df, val_df, test_df = create_patient_wise_splits(mock_df, patient_col="patient_id")

    train_patients = set(train_df["patient_id"].unique())
    val_patients = set(val_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())

    # 1. Total partitions sum to all actual unique patients
    actual_unique = len(mock_df["patient_id"].unique())
    assert len(train_patients | val_patients | test_patients) == actual_unique

    # 2. STRICT ZERO LEAKAGE ASSERTIONS
    train_val_overlap = train_patients.intersection(val_patients)
    assert len(train_val_overlap) == 0, f"Patient leakage between train and val: {train_val_overlap}"

    train_test_overlap = train_patients.intersection(test_patients)
    assert len(train_test_overlap) == 0, f"Patient leakage between train and test: {train_test_overlap}"

    val_test_overlap = val_patients.intersection(test_patients)
    assert len(val_test_overlap) == 0, f"Patient leakage between val and test: {val_test_overlap}"


def test_ptbxl_strat_fold_patient_wise_leakage():
    # If ptbxl_database.csv is available, verify that 10-fold strat_fold has no patient leakage
    possible_paths = [
        Path(r"D:\ECG-AI-Research-1Lead\data\raw\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptbxl_database.csv"),
        Path(r"D:\ECG-AI-Research\data\raw\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptbxl_database.csv"),
    ]
    db_path = None
    for p in possible_paths:
        if p.exists():
            db_path = p
            break

    if db_path is not None:
        df = pd.read_csv(db_path)
        if "patient_id" in df.columns and "strat_fold" in df.columns:
            # Check fold 10 (test) vs folds 1-8 (train) vs fold 9 (val)
            test_pts = set(df[df["strat_fold"] == 10]["patient_id"])
            val_pts = set(df[df["strat_fold"] == 9]["patient_id"])
            train_pts = set(df[df["strat_fold"] <= 8]["patient_id"])

            assert len(test_pts.intersection(train_pts)) == 0
            assert len(val_pts.intersection(train_pts)) == 0
            assert len(test_pts.intersection(val_pts)) == 0
