import argparse
import sys
import time
from pathlib import Path

# --- FIX --- Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import pathology feature extraction logic
from common.src.features.pathology.main import run_pathology_vision_task
from common.src.io import load_inputs


def run_pathology_pipeline(args):
    """Encapsulated logic for the pathology feature extraction task."""
    pathology_output_dir = args.output_dir / "pathology"
    pathology_output_dir.mkdir(parents=True, exist_ok=True)

    print("---" * 10)
    print("🚀 Starting Feature Extraction for Pathology (Task 2)")
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


def main():
    """Main execution function to run pathology feature extraction."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Run pathology feature extraction for Task 2 classification.")

    parser.add_argument("--input_dir", type=Path, required=True, help="Path to input directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to output directory.")
    parser.add_argument("--pathology_model_dir", type=Path, required=True, help="Path to pathology model directory.")

    args = parser.parse_args()

    # Run pathology extraction
    run_pathology_pipeline(args)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")


if __name__ == "__main__":
    main()
