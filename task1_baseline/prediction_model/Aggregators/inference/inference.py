# inference.py

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

# --- Add Project Root to Python Path ---
# (Assumes this script is in a subdirectory of the project)
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
prediction_model_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(prediction_model_root))

# --- Import from our helper modules ---
from task1_baseline.prediction_model.Aggregators.inference.feature_extractor import run_feature_extraction
from task1_baseline.prediction_model.Aggregators.inference.model_loader import load_model_and_assets
from task1_baseline.prediction_model.Aggregators.inference.feature_loader import load_features
from task1_baseline.prediction_model.Aggregators.inference.prediction import run_inference_and_calibrate

# --- Define Grand Challenge paths ---
GC_INPUT_PATH = Path("/input")
GC_OUTPUT_PATH = Path("/output")
GC_MODEL_PATH = Path("/opt/ml/model/")


def run_complete_pipeline(input_dir: Path, output_dir: Path, model_dir: Path):
    """
    This is the core, end-to-end pipeline. It runs feature extraction from
    raw data and then predicts time-to-event using the trained models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pathology_model_path = model_dir / "pathology"
    radiology_model_path = model_dir / "radiology"
    prediction_model_path = model_dir / "ABMIL_task1"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Created temporary directory for features: {temp_path}")
        try:
            print("--- Step 1: Running Feature Extraction ---")
            feature_output_dir = run_feature_extraction(input_dir=input_dir, output_dir=temp_path, pathology_model_dir=pathology_model_path, radiology_model_dir=radiology_model_path)
            print("✅ Feature extraction complete.")
            print("\n--- Step 2: Loading Prediction Model and Assets ---")
            model, clinical_processor, calibration_data = load_model_and_assets(prediction_model_path, input_dir)
            print("\n--- Step 3: Loading All Extracted Features ---")
            path_feats, rad_feats, clin_feats = load_features(case_id="chimera-clinical-data-of-prostate-cancer-patients", pathology_features_dir=feature_output_dir / "pathology", radiology_features_dir=feature_output_dir / "radiology", clinical_processor=clinical_processor)
            print("✅ All features loaded.")
            predicted_time_months = run_inference_and_calibrate(model=model, pathology_features=path_feats, radiology_features=rad_feats, clinical_features=clin_feats, calibration_data=calibration_data)
            print("\n--- Step 6: Saving Final Prediction ---")
            final_output_path = output_dir / "time-to-biochemical-recurrence-for-prostate-cancer-months.json"
            with open(final_output_path, "w") as f:
                json.dump(float(predicted_time_months), f, indent=4)
            print(f"✅ Successfully saved prediction to {final_output_path}")
            return 0
        except Exception as e:
            print(f"❌ An error occurred during the pipeline: {e}")
            traceback.print_exc()
            return 1


def get_interface_key():
    """Reads inputs.json and returns a sorted tuple of interface slugs."""
    with open(GC_INPUT_PATH / "inputs.json", "r") as f:
        inputs = json.load(f)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def run_grand_challenge():
    """
    The main entrypoint for the Grand Challenge environment.
    This function now dynamically builds the dispatch table to be robust.
    """
    print("🚀 Starting Grand Challenge prediction pipeline...")

    # 1. Define the true base slugs, present in ALL interfaces
    base_slugs = [
        'axial-adc-prostate-mri', 'axial-t2-prostate-mri',
        'chimera-clinical-data-of-prostate-cancer-patients',
        'prostate-tissue-mask-for-axial-t2-prostate-mri',
        'prostatectomy-tissue-mask', 
        'prostatectomy-tissue-whole-slide-image',
        'transverse-hbv-prostate-mri'
    ]

    # 2. Define the additional slugs that come in pairs
    additional_slug_pairs = [
        (f'prostatectomy-tissue-whole-slide-image-{i}', f'prostatectomy-tissue-whole-slide-image-{i}-2')
        for i in range(1, 10)
    ]

    dispatch_table = {}
    unified_handler = lambda: run_complete_pipeline(
        input_dir=GC_INPUT_PATH,
        output_dir=GC_OUTPUT_PATH,
        model_dir=GC_MODEL_PATH
    )

    # 3. Create the key for the base case (interf0)
    current_slugs = list(base_slugs)
    interface_key = tuple(sorted(current_slugs))
    dispatch_table[interface_key] = unified_handler

    # 4. Loop to create keys for interf1 through interf9 by adding pairs
    for i in range(9): 
        current_slugs.extend(additional_slug_pairs[i])
        interface_key = tuple(sorted(current_slugs))
        dispatch_table[interface_key] = unified_handler

    # --- Dispatch to the correct handler ---
    current_key = get_interface_key()
    handler = dispatch_table.get(current_key)

    if handler is None:
        raise RuntimeError(f"No handler found for the provided interface key: {current_key}")

    return handler()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # --- LOCAL EXECUTION MODE ---
        print("🚀 Starting Local prediction pipeline...")
        parser = argparse.ArgumentParser(description="Run full inference pipeline locally.")
        parser.add_argument("--input_dir", type=Path, required=True)
        parser.add_argument("--output_dir", type=Path, required=True)
        parser.add_argument("--model_dir", type=Path, required=True)
        args = parser.parse_args()
        
        ret_code = run_complete_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir
        )
        sys.exit(ret_code)
    else:
        # --- GRAND CHALLENGE EXECUTION MODE ---
        sys.exit(run_grand_challenge())