import argparse
import sys
from pathlib import Path

# --- FIX ---
# Add the project root to the Python path to allow for package imports
# This assumes 'common' is located two directories above the script's location
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# --- UPDATED --- Corrected the import path to match the file structure
from common.src.features.pathology.main import run_pathology_vision_task
from common.src.io import load_inputs

def main():
    """
    Main execution function to run the feature extraction pipeline.
    """
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Run the UNI pathology feature extraction pipeline.")
    parser.add_argument(
        "--input_dir", 
        type=Path, 
        required=True, 
        help="Path to the directory containing 'inputs.json' (e.g., 'task1_baseline/test/input/interf0')."
    )
    parser.add_argument(
        "--model_dir", 
        type=Path, 
        required=True, 
        help="Path to the directory containing the UNI model files ('pytorch_model.bin', 'uni-config.json')."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        required=True, 
        help="Path to the directory where the output 'features.pt' will be saved."
    )
    args = parser.parse_args()

    print("---" * 10)
    print(f"🚀 Starting Feature Extraction")
    print(f"📂 Input Directory: {args.input_dir}")
    print(f"🧠 Model Directory: {args.model_dir}")
    print(f"💾 Output Directory: {args.output_dir}")
    print("---" * 10)

    inputs_json_path = args.input_dir / "inputs.json"
    input_information = load_inputs(
        input_path=inputs_json_path, 
        base_dir=args.input_dir
    )

    run_pathology_vision_task(
        input_information=input_information,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )

    print("---" * 10)
    print("✅ Feature extraction complete!")
    print(f"Output saved to {args.output_dir / 'features.pt'}")
    print("---" * 10)


if __name__ == "__main__":
    main()
