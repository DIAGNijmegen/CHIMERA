# Aggregators/inference/feature_loader.py
import torch
import pandas as pd
from pathlib import Path

def load_features(pathology_features_dir: Path, clinical_df: pd.DataFrame, case_id: str):
    """
    Load pathology features (.pt) and clinical features for a given case.
    """
    # Pathology
    feat_path = pathology_features_dir / f"{case_id}.pt"
    pathology_features = torch.load(feat_path, map_location="cpu").float()

    # Clinical (ensure same order as training)
    clinical_row = clinical_df[clinical_df["case_id"] == case_id].iloc[0]
    clinical_feats = torch.tensor(clinical_row.drop(["case_id", "label"], errors="ignore").values, dtype=torch.float32)

    return pathology_features, clinical_feats

