# Aggregators/inference/feature_extractor.py

from pathlib import Path
from common.src.features.pathology.main import run_pathology_vision_task
import torch


def extract_pathology_features(wsi_path: Path, mask_path: Path, model_dir: Path) -> torch.Tensor:
    """
    Extract pathology features for a single case using Slide2Vec (UNI) on-the-fly.

    Parameters:
    - wsi_path: Path to the whole-slide image (.tiff)
    - mask_path: Path to the corresponding tissue mask (.tiff)
    - model_dir: Path to directory containing model.bin and config.json (UNI extractor)

    Returns:
    - A torch.Tensor of shape (num_patches, feature_dim)
    """
    print(f"🧠 Extracting pathology features dynamically...")
    features = run_pathology_vision_task(
        wsi_path=wsi_path,
        tissue_mask_path=mask_path,
        model_dir=model_dir,
        output_path=None  # don't save
    )

    print(f"✅ Feature extraction complete. Shape: {features.shape}")
    return features
