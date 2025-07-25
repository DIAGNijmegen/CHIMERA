
import sys
from pathlib import Path

from common.src.features.radiology.main import run_radiology_feature_extraction

def run_feature_extraction(input_dir: Path, output_dir: Path, radiology_model_dir: Path):
    """Runs the full feature extraction pipeline for both modalities."""
    # --- Create a temporary directory for features ---
    feature_output_dir = output_dir / "features"
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    radiology_output_dir = feature_output_dir / "radiology"
    radiology_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run Radiology Feature Extraction ---
    print("🚀 Starting Feature Extraction for Radiology")
    run_radiology_feature_extraction(
        input_dir=input_dir,
        output_dir=radiology_output_dir,
        model_dir=radiology_model_dir
    )
    print("✅ Radiology feature extraction complete!")

    return feature_output_dir
