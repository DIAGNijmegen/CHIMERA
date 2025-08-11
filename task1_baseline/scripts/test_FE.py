import argparse
import sys
import time
from pathlib import Path

# --- FIX ---
# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# --- UPDATED --- Corrected the import path
from common.src.features.pathology.main import run_pathology_vision_task
from common.src.features.radiology.main import run_radiology_feature_extraction
from common.src.io import load_inputs

def run_pathology_pipeline(args):
    """Encapsulated logic for the pathology feature extraction task."""
    pathology_output_dir = args.output_dir / "pathology"
    pathology_output_dir.mkdir(parents=True, exist_ok=True)

    print("---" * 10)
    print("🚀 Starting Feature Extraction for Pathology")
    print(f"📂 Input Directory: {args.input_dir}")
    print(f"🧠 Model Directory: {args.pathology_model_dir}")
    print(f"💾 Output Directory: {pathology_output_dir}")
    print("---" * 10)

    inputs_json_path = args.input_dir / "inputs.json"
    input_information = load_inputs(input_path=inputs_json_path)
    run_pathology_vision_task(
        input_information=input_information,
        model_dir=args.pathology_model_dir,
        output_dir=pathology_output_dir
    )
    print("---" * 10)
    print("✅ Pathology feature extraction complete!")
    print(f"Output saved to {pathology_output_dir}")
    print("---" * 10)

def run_radiology_pipeline(args):
    """Encapsulated logic for the radiology feature extraction task."""
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


def main():
    """Main execution function to run the feature extraction pipeline."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Run feature extraction for pathology and/or radiology.")
    
    # --- NEW: Positional argument for modality selection ---
    parser.add_argument(
        "modality",
        type=str,
        nargs="?",  # Makes the argument optional
        default="all",  # If not provided, it defaults to 'all'
        choices=["pathology", "radiology", "all"],
        help="Optional: specify a single modality to run ('pathology' or 'radiology'). If omitted, both will run."
    )
    
    # --- Existing named arguments ---
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to input directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to output directory.")
    parser.add_argument("--pathology_model_dir", type=Path, required=True, help="Path to pathology model directory.")
    parser.add_argument("--radiology_model_dir", type=Path, required=True, help="Path to radiology model directory.")
    
    args = parser.parse_args()

    # --- UPDATED: Conditional execution based on the 'modality' argument ---
    if args.modality in ["pathology", "all"]:
        run_pathology_pipeline(args)
    
    if args.modality in ["radiology", "all"]:
        run_radiology_pipeline(args)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")


if __name__ == "__main__":
    main()