
import sys
from pathlib import Path

# --- Add Project Root to Python Path ---
project_root = Path(__file__).resolve().parents[5]
sys.path.append(str(project_root))

from common.src.features.pathology.main import run_pathology_vision_task
from common.src.features.radiology.main import run_radiology_feature_extraction
from common.src.io import load_inputs

def run_feature_extraction(input_dir: Path, output_dir: Path, pathology_model_dir: Path, radiology_model_dir: Path):
    """Runs the full feature extraction pipeline for both modalities."""
    # --- Create a temporary directory for features ---
    feature_output_dir = output_dir / "features"
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    pathology_output_dir = feature_output_dir / "pathology"
    pathology_output_dir.mkdir(parents=True, exist_ok=True)

    radiology_output_dir = feature_output_dir / "radiology"
    radiology_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run Pathology Feature Extraction ---
    print("---*10")
    print("🚀 Starting Feature Extraction for Pathology")
    inputs_json_path = input_dir / "inputs.json"
    input_information = load_inputs(input_path=inputs_json_path)
    run_pathology_vision_task(
        input_information=input_information,
        model_dir=pathology_model_dir,
        output_dir=pathology_output_dir
    )
    print("✅ Pathology feature extraction complete!")

    # --- Run Radiology Feature Extraction ---
    print("---*10")
    print("🚀 Starting Feature Extraction for Radiology")
    run_radiology_feature_extraction(
        input_dir=input_dir,
        output_dir=radiology_output_dir,
        model_dir=radiology_model_dir
    )
    print("✅ Radiology feature extraction complete!")

    return feature_output_dir
