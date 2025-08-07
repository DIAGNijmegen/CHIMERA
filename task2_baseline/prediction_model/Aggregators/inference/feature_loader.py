# Aggregators/inference/feature_loader.py

import torch
import pandas as pd
from pathlib import Path
from prediction_model.Aggregators.inference.feature_extractor import extract_pathology_features

# Grand Challenge input slugs
WSI_SLUG = "prostatectomy-tissue-whole-slide-image"
MASK_SLUG = "tissue-mask"

def load_features(
    input_dir: Path,
    model_dir: Path,
    clinical_df: pd.DataFrame,
    case_id: str
):
    """
    Dynamically extract pathology features + load clinical features for a given case.

    Parameters:
    - input_dir: Path to mounted /input folder with images/<slug>/<filename>.tiff
    - model_dir: Path to mounted /opt/ml/model folder
    - clinical_df: Pre-loaded clinical dataframe
    - case_id: Current sample ID (filename without .tiff)

    Returns:
    - pathology_features: Tensor (N_patches, feature_dim)
    - clinical_feats: Tensor (num_clinical_features,)
    """
    # --- Get WSI + mask paths from expected locations ---
    wsi_path = input_dir / "images" / WSI_SLUG / f"{case_id}.tiff"
    mask_path = input_dir / "images" / MASK_SLUG / f"{case_id}.tiff"

    if not wsi_path.exists():
        raise FileNotFoundError(f"Missing WSI file: {wsi_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask file: {mask_path}")

    # --- Extract pathology features in memory ---
    pathology_features = extract_pathology_features(
        wsi_path=wsi_path,
        mask_path=mask_path,
        model_dir=model_dir
    )

    # --- Extract clinical features ---
    matching_rows = clinical_df[clinical_df["case_id"] == case_id]
    if matching_rows.empty:
        raise ValueError(f"Case ID '{case_id}' not found in clinical DataFrame")
    clinical_row = matching_rows.iloc[0]
    clinical_feats = torch.tensor(
        clinical_row.drop(["case_id", "label"], errors="ignore").values,
        dtype=torch.float32
    )

    return pathology_features, clinical_feats
