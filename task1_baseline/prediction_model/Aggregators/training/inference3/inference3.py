import os
import json
import glob
from pathlib import Path
import traceback
import sys
import tempfile
import shutil
import argparse

# --- Add Project Root to Python Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
aggregators_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, aggregators_path)

# --- Import from our new helper modules ---
from Aggregators.training.inference3.model_loader import load_model_and_assets
from Aggregators.training.inference3.feature_loader import load_features
from Aggregators.training.inference3.prediction import run_inference_and_calibrate
from Aggregators.training.inference3.feature_extractor import run_feature_extraction

# --- Define Grand Challenge paths (will be used by handlers) ---
GC_INPUT_PATH = Path("/input")
GC_OUTPUT_PATH = Path("/output")
GC_MODEL_PATH = Path("/opt/ml/model/")

def run_pipeline(input_dir: Path, output_dir: Path, model_dir: Path):
    """Core end-to-end pipeline for feature extraction and prediction."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Define specific model paths based on the new structure ---
    pathology_model_path = model_dir / "pathology"
    radiology_model_path = model_dir / "radiology"
    prediction_model_path = model_dir / "ABMIL_task1"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Created temporary directory: {temp_path}")

        try:
            # --- Step 1: Run Feature Extraction ---
            print("--- Step 1: Running Feature Extraction ---")
            feature_output_dir = run_feature_extraction(
                input_dir=input_dir,
                output_dir=temp_path,
                pathology_model_dir=pathology_model_path,
                radiology_model_dir=radiology_model_path
            )
            print(f"Feature extraction complete. Features stored in: {feature_output_dir}")

            # --- Step 2: Load Model and Prediction Assets ---
            print("\n--- Step 2: Loading Model and Prediction Assets ---")
            model, clinical_processor, calibration_data = load_model_and_assets(prediction_model_path, input_dir)

            # --- Step 3: Identify Case and Load Features ---
            clinical_file_path = input_dir / "chimera-clinical-data-of-prostate-cancer-patients.json"
            if not clinical_file_path.exists():
                raise FileNotFoundError(f"Clinical JSON file not found at {clinical_file_path}")
            
            with open(clinical_file_path, 'r') as f:
                clinical_data_json = json.load(f)
                if isinstance(clinical_data_json, list):
                    clinical_data_json = clinical_data_json[0]

            clin_file_name = "chimera-clinical-data-of-prostate-cancer-patients"
            print(f"\n--- Step 3: Loading Features---")

            pathology_features, radiology_features, clinical_features = load_features(
                case_id=clin_file_name,
                pathology_features_dir=feature_output_dir / "pathology",
                radiology_features_dir=feature_output_dir / "radiology",
                clinical_processor=clinical_processor
            )

            # --- Step 4 & 5: Run Inference and Calibrate ---
            predicted_time_months = run_inference_and_calibrate(
                model=model,
                pathology_features=pathology_features,
                radiology_features=radiology_features,
                clinical_features=clinical_features,
                calibration_data=calibration_data
            )

            # --- Step 6: Save Final Prediction ---
            print("\n--- Step 6: Saving Final Prediction ---")
            final_output_path = output_dir / "time-to-biochemical-recurrence-for-prostate-cancer-months.json"
            with open(final_output_path, "w") as f:
                json.dump(predicted_time_months, f, indent=4)
            print(f"Successfully saved prediction to {final_output_path}")
            return 0

        except Exception as e:
            print(f"An error occurred during the pipeline: {e}")
            traceback.print_exc()
            return 1

# =============================================================================
# Grand Challenge I/O and Interface Handling Boilerplate
# =============================================================================

def interf_handler_template(wsi_slugs):
    """A template handler that calls the main pipeline with GC paths."""
    print(f"Handling interface with {len(wsi_slugs)} WSI file(s).")
    # The number of WSIs is handled by the feature extractor, so we just run the pipeline.
    return run_pipeline(input_dir=GC_INPUT_PATH, output_dir=GC_OUTPUT_PATH, model_dir=GC_MODEL_PATH)

def interf0_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image"])
def interf1_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1"])
def interf2_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2"])
def interf3_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3"])
def interf4_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4"])
def interf5_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5"])
def interf6_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6"])
def interf7_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6", "prostatectomy-tissue-whole-slide-image-7"])
def interf8_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6", "prostatectomy-tissue-whole-slide-image-7", "prostatectomy-tissue-whole-slide-image-8"])
def interf9_handler(): return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6", "prostatectomy-tissue-whole-slide-image-7", "prostatectomy-tissue-whole-slide-image-8", "prostatectomy-tissue-whole-slide-image-9"])

def get_interface_key():
    with open(GC_INPUT_PATH / "inputs.json", "r") as f:
        inputs = json.load(f)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))

def run_gc_mode():
    """Entry point for the Grand Challenge container."""
    interface_key = get_interface_key()
    handler = {
        # ... (handler mapping as in the original file)
    }.get(interface_key)

    if handler is None:
        raise RuntimeError(f"No handler found for interface key: {interface_key}")

    return handler()

if __name__ == "__main__":
    # This script can be run in two ways:
    # 1. With command-line arguments for local testing:
    #    python3 inference3.py --input_dir /path/to/input --output_dir /path/to/output --model_dir /path/to/model
    # 2. Without arguments, to simulate the Grand Challenge environment:
    #    python3 inference3.py

    if len(sys.argv) > 1 and '--input_dir' in sys.argv:
        # --- Local Command-Line Execution ---
        parser = argparse.ArgumentParser(description="Run full inference pipeline locally.")
        parser.add_argument("--input_dir", type=Path, required=True, help="Path to input directory.")
        parser.add_argument("--output_dir", type=Path, required=True, help="Path to output directory.")
        parser.add_argument("--model_dir", type=Path, required=True, help="Path to model directory.")
        args = parser.parse_args()

        ret_code = run_pipeline(input_dir=args.input_dir, output_dir=args.output_dir, model_dir=args.model_dir)
        sys.exit(ret_code)
    else:
        # --- Grand Challenge Execution ---
        # This will read from the standard GC paths
        try:
            raise SystemExit(run_gc_mode())
        except Exception as e:
            print(f"An error occurred during GC execution: {e}")
            traceback.print_exc()
            raise SystemExit(1)