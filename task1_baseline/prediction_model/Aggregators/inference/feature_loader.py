
import torch
import glob
from pathlib import Path

def load_features(case_id: str, pathology_features_dir: Path, radiology_features_dir: Path, clinical_processor):
    """
    Loads all features for a given case.

    Args:
        case_id: The ID of the case to load features for.
        pathology_features_dir: Path to the pathology features directory.
        radiology_features_dir: Path to the radiology features directory.
        clinical_processor: The fitted clinical data processor.

    Returns:
        A tuple containing:
        - Pathology features tensor.
        - Radiology features tensor.
        - Clinical features tensor.
    """
    # --- Load Pathology Features ---
    # In this pipeline, the feature extractor saves a generic 'features.pt' file.
    pathology_feature_files = glob.glob(str(pathology_features_dir / "*.pt"))

    if not pathology_feature_files:
        raise FileNotFoundError(f"CRITICAL: No pathology feature files found in {pathology_features_dir}")
    
    all_pathology_features = [torch.load(fp) for fp in sorted(pathology_feature_files)]
    pathology_features_tensor = torch.cat(all_pathology_features, dim=0).unsqueeze(0)
    print(f"Loaded and concatenated {len(all_pathology_features)} pathology feature file(s).")
    
    radiology_features_tensor = None
    radiology_files = glob.glob(str(radiology_features_dir / "*.pt"))
    if radiology_files:
        radiology_features_tensor = torch.load(radiology_files[0])
        if len(radiology_features_tensor.shape) == 1:
            radiology_features_tensor = radiology_features_tensor.unsqueeze(0)
        print("Loaded radiology features.")
    else:
        print("Warning: No radiology feature file found.")

    clinical_features_tensor = clinical_processor.transform(case_id).unsqueeze(0)
    print("Processed clinical data.")
        
    return pathology_features_tensor, radiology_features_tensor, clinical_features_tensor