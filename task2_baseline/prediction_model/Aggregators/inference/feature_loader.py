# Aggregators/inference/feature_loader.py

import torch
import pandas as pd
from pathlib import Path


def load_features(pathology_features_dir: Path, clinical_df: pd.DataFrame, case_id: str):
    """
    Load pathology features (.pt) and clinical features for a given case.

    Parameters:
    - pathology_features_dir: Path to directory with .pt files (e.g., /output/pathology_features/)
    - clinical_df: Preprocessed DataFrame with rows for each case (same order and normalization as training)
    - case_id: Sample ID to fetch features for

    Returns:
    - pathology_features: Tensor of shape (num_tiles, feature_dim)
    - clinical_feats: Tensor of shape (num_clinical_features,)
    """
    # --- Pathology features ---
    feat_path = pathology_features_dir / f"{case_id}.pt"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    pathology_features = torch.load(feat_path, map_location="cpu").float()

    # --- Clinical features ---
    matching_rows = clinical_df[clinical_df["case_id"] == case_id]
    if matching_rows.empty:
        raise ValueError(f"Case ID '{case_id}' not found in clinical DataFrame")
    clinical_row = matching_rows.iloc[0]
    clinical_feats = torch.tensor(
        clinical_row.drop(["case_id", "label"], errors="ignore").values,
        dtype=torch.float32
    )

    return pathology_features, clinical_feats


