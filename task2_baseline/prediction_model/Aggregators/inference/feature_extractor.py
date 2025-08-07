# Aggregators/inference/feature_extractor.py

from pathlib import Path
from common.src.features.pathology.main import run_pathology_vision_task


def extract_pathology_features(wsi_path: Path, mask_path: Path, case_id: str, output_dir: Path, model_dir: Path):
    """
    Extract pathology features for a single case using Slide2Vec (UNI).
    
    Parameters:
    - wsi_path: Path to the whole-slide image (.tiff)
    - mask_path: Path to the corresponding tissue mask (.tiff)
    - case_id: ID used to save the extracted features (e.g., "1001")
    - output_dir: Directory to save extracted features
    - model_dir: Path to directory containing model.bin and config.json (UNI extractor)
    
    Returns:
    - Path to the saved .pt file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}.pt"

    print(f"🧠 Extracting pathology features for case: {case_id}")
    run_pathology_vision_task(
        wsi_path=wsi_path,
        tissue_mask_path=mask_path,
        model_dir=model_dir,
        output_path=output_path
    )

    print(f"✅ Saved pathology features to: {output_path}")
    return output_path


