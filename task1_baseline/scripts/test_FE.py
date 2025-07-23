import argparse
import sys
import time
from pathlib import Path

# --- FIX ---
# Add the project root to the Python path to allow for package imports
# This assumes 'common' is located two directories above the script's location
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# --- UPDATED --- Corrected the import path to match the file structure
from common.src.features.pathology.main import run_pathology_vision_task
from common.src.features.radiology.main import run_radiology_feature_extraction
from common.src.io import load_inputs

def main():
    """
    Main execution function to run the feature extraction pipeline for both pathology and radiology.
    """
    start_time = time.time()
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Run the feature extraction pipeline for both pathology and radiology.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the input directory (containing inputs.json for pathology)."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the base directory where the output features will be saved."
    )
    parser.add_argument(
        "--pathology_model_dir",
        type=Path,
        required=True,
        help="Path to the directory containing the pathology model files."
    )
    parser.add_argument(
        "--radiology_model_dir",
        type=Path,
        required=True,
        help="Path to the directory containing the radiology model files."
    )
    args = parser.parse_args()

    # --- Pathology Feature Extraction ---
    pathology_output_dir = args.output_dir / "pathology"
    pathology_output_dir.mkdir(parents=True, exist_ok=True)

    print("---" * 10)
    print("🚀 Starting Feature Extraction for Pathology")
    print(f"📂 Input Directory: {args.input_dir}")
    print(f"🧠 Model Directory: {args.pathology_model_dir}")
    print(f"💾 Output Directory: {pathology_output_dir}")
    print("---" * 10)

    inputs_json_path = args.input_dir / "inputs.json"
    input_information = load_inputs(
        input_path=inputs_json_path
    )
    run_pathology_vision_task(
        input_information=input_information,
        model_dir=args.pathology_model_dir,
        output_dir=pathology_output_dir
    )
    print("---" * 10)
    print("✅ Pathology feature extraction complete!")
    print(f"Output saved to {pathology_output_dir}")
    print("---" * 10)

    # --- Radiology Feature Extraction ---
    radiology_output_dir = args.output_dir / "radiology"
    radiology_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "---" * 10)
    print("🚀 Starting Feature Extraction for Radiology")
    print(f"📂 Input Directory: {args.input_dir}")
    print(f"🧠 Model Directory: {args.radiology_model_dir}")
    print(f"💾 Output Directory: {radiology_output_dir}")
    print("---" * 10)

    run_radiology_feature_extraction(
        input_dir=args.input_dir,
        output_dir=radiology_output_dir,
        model_dir=args.radiology_model_dir
    )

    print("---" * 10)
    print("✅ Radiology feature extraction complete!")
    print(f"Output saved to {radiology_output_dir}")
    print("---" * 10)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")


if __name__ == "__main__":
    main()