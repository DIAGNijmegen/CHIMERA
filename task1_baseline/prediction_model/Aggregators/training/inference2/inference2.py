import os
import json
import glob
from pathlib import Path
import traceback
import sys

# --- Add Project Root to Python Path ---
# This is a robust fix for the ModuleNotFoundError. It adds both the project root
# ('task1_baseline') and the 'Aggregators' directory to Python's path.
# This ensures that top-level imports ('from Aggregators...'), internal relative
# imports ('from utils...'), and unpickling all work correctly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
aggregators_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, aggregators_path)


# --- Import from our new helper modules ---
from Aggregators.training.inference2.model_loader import load_model_and_assets
from Aggregators.training.inference2.feature_loader import load_features
from Aggregators.training.inference2.prediction import run_inference_and_calibrate

# --- Grand Challenge Standard Paths ---
# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
# MODEL_PATH = Path("/opt/ml/model/")

# --- Paths for Local Testing ---
INPUT_PATH = Path("/data/temporary/chimera/Baseline_models/Task1_ABMIL/inference_test_input/input/1026")
OUTPUT_PATH = Path("/data/temporary/chimera/Baseline_models/Task1_ABMIL/inference_test_input/output")
MODEL_PATH = Path("/data/temporary/chimera/Baseline_models/Task1_ABMIL/inference_test_input/model")

# --- Define expected feature directories ---
PATHOLOGY_FEATURES_DIR = INPUT_PATH / "features/pathology"
RADIOLOGY_FEATURES_DIR = INPUT_PATH / "features/radiology"
CLINICAL_DIR = INPUT_PATH

def run_prediction(clinical_data_json: dict, all_wsi_paths: list, mri_paths: dict):
    """
    Core function to run the multi-modal prediction pipeline.
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # --- Step 1: Load Model and Assets ---
    print("--- Step 1: Loading Model and All Assets ---")
    model, clinical_processor, calibration_data = load_model_and_assets(MODEL_PATH)

    # --- Step 2: Load Features ---
    print("\n--- Step 2: Loading All Features ---")
    case_id = clinical_data_json['case_id']
    pathology_features, radiology_features, clinical_features = load_features(
        case_id=case_id,
        pathology_features_dir=PATHOLOGY_FEATURES_DIR,
        radiology_features_dir=RADIOLOGY_FEATURES_DIR,
        clinical_processor=clinical_processor,
        all_wsi_paths=all_wsi_paths
    )

    # --- Step 3 & 4: Run Inference and Calibrate ---
    predicted_time_months = run_inference_and_calibrate(
        model=model,
        pathology_features=pathology_features,
        radiology_features=radiology_features,
        clinical_features=clinical_features,
        calibration_data=calibration_data
    )

    # --- Step 5: Save Final Prediction ---
    print("\n--- Step 5: Saving Prediction ---")
    output_path = OUTPUT_PATH / "time-to-biochemical-recurrence-for-prostate-cancer-months.json"
    write_json_file(location=output_path, content=predicted_time_months)
    print(f"Successfully saved prediction to {output_path}")

# =============================================================================
# Grand Challenge I/O and Interface Handling Boilerplate (Identical to original)
# =============================================================================

def interf_handler_template(wsi_slugs):
    """A template handler to process inputs for a given number of WSIs."""
    print(f"Handling interface with {len(wsi_slugs)} WSI file(s).")
    
    mri_paths = {
        't2': INPUT_PATH / "images/axial-t2-prostate-mri",
        'adc': INPUT_PATH / "images/axial-adc-prostate-mri",
        'hbv': INPUT_PATH / "images/transverse-hbv-prostate-mri",
    }
    
    all_wsi_paths = [INPUT_PATH / f"images/{slug}" for slug in wsi_slugs]
    
    clinical_files = [f for f in glob.glob(str(INPUT_PATH / "*.json")) if Path(f).name != 'inputs.json']
    if not clinical_files:
        raise FileNotFoundError(f"No clinical JSON file found in '{INPUT_PATH}'")
    
    clinical_file_path = Path(clinical_files[0])
    loaded_json = load_json_file(location=clinical_file_path)
    
    if isinstance(loaded_json, list):
        if len(loaded_json) == 0:
            raise ValueError(f"Clinical JSON file at {clinical_file_path} is an empty list.")
        clinical_data_dict = loaded_json[0]
    else:
        clinical_data_dict = loaded_json
    
    case_id = clinical_file_path.stem.replace("-clinical-data-of-prostate-cancer-patients", "")
    clinical_data_dict['case_id'] = case_id

    print("Placeholder: MRI Feature Extraction would run here.")
    print("Placeholder: WSI Feature Extraction would run here.")

    run_prediction(clinical_data_dict, all_wsi_paths, mri_paths)
    return 0

def interf0_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image"])
def interf1_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1"])
def interf2_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2"])
def interf3_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3"])
def interf4_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4"])
def interf5_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5"])
def interf6_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6"])
def interf7_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6", "prostatectomy-tissue-whole-slide-image-7"])
def interf8_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6", "prostatectomy-tissue-whole-slide-image-7", "prostatectomy-tissue-whole-slide-image-8"])
def interf9_handler():
    return interf_handler_template(wsi_slugs=["prostatectomy-tissue-whole-slide-image", "prostatectomy-tissue-whole-slide-image-1", "prostatectomy-tissue-whole-slide-image-2", "prostatectomy-tissue-whole-slide-image-3", "prostatectomy-tissue-whole-slide-image-4", "prostatectomy-tissue-whole-slide-image-5", "prostatectomy-tissue-whole-slide-image-6", "prostatectomy-tissue-whole-slide-image-7", "prostatectomy-tissue-whole-slide-image-8", "prostatectomy-tissue-whole-slide-image-9"])

def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))

def load_json_file(*, location):
    with open(location, "r") as f:
        return json.load(f)

def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def run():
    """Main entry point for the container."""
    interface_key = get_interface_key()
    handler = {
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'transverse-hbv-prostate-mri'): interf0_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'transverse-hbv-prostate-mri'): interf1_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'transverse-hbv-prostate-mri'): interf2_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'prostatectomy-tissue-whole-slide-image-3', 'prostatectomy-tissue-whole-slide-image-3-2', 'transverse-hbv-prostate-mri'): interf3_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'prostatectomy-tissue-whole-slide-image-3', 'prostatectomy-tissue-whole-slide-image-3-2', 'prostatectomy-tissue-whole-slide-image-4', 'prostatectomy-tissue-whole-slide-image-4-2', 'transverse-hbv-prostate-mri'): interf4_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'prostatectomy-tissue-whole-slide-image-3', 'prostatectomy-tissue-whole-slide-image-3-2', 'prostatectomy-tissue-whole-slide-image-4', 'prostatectomy-tissue-whole-slide-image-4-2', 'prostatectomy-tissue-whole-slide-image-5', 'prostatectomy-tissue-whole-slide-image-5-2', 'transverse-hbv-prostate-mri'): interf5_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'prostatectomy-tissue-whole-slide-image-3', 'prostatectomy-tissue-whole-slide-image-3-2', 'prostatectomy-tissue-whole-slide-image-4', 'prostatectomy-tissue-whole-slide-image-4-2', 'prostatectomy-tissue-whole-slide-image-5', 'prostatectomy-tissue-whole-slide-image-5-2', 'prostatectomy-tissue-whole-slide-image-6', 'prostatectomy-tissue-whole-slide-image-6-2', 'transverse-hbv-prostate-mri'): interf6_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'prostatectomy-tissue-whole-slide-image-3', 'prostatectomy-tissue-whole-slide-image-3-2', 'prostatectomy-tissue-whole-slide-image-4', 'prostatectomy-tissue-whole-slide-image-4-2', 'prostatectomy-tissue-whole-slide-image-5', 'prostatectomy-tissue-whole-slide-image-5-2', 'prostatectomy-tissue-whole-slide-image-6', 'prostatectomy-tissue-whole-slide-image-6-2', 'prostatectomy-tissue-whole-slide-image-7', 'prostatectomy-tissue-whole-slide-image-7-2', 'transverse-hbv-prostate-mri'): interf7_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'prostatectomy-tissue-whole-slide-image-3', 'prostatectomy-tissue-whole-slide-image-3-2', 'prostatectomy-tissue-whole-slide-image-4', 'prostatectomy-tissue-whole-slide-image-4-2', 'prostatectomy-tissue-whole-slide-image-5', 'prostatectomy-tissue-whole-slide-image-5-2', 'prostatectomy-tissue-whole-slide-image-6', 'prostatectomy-tissue-whole-slide-image-6-2', 'prostatectomy-tissue-whole-slide-image-7', 'prostatectomy-tissue-whole-slide-image-7-2', 'prostatectomy-tissue-whole-slide-image-8', 'prostatectomy-tissue-whole-slide-image-8-2', 'transverse-hbv-prostate-mri'): interf8_handler,
        ('axial-adc-prostate-mri', 'axial-t2-prostate-mri', 'chimera-clinical-data-of-prostate-cancer-patients', 'prostate-tissue-mask-for-axial-t2-prostate-mri', 'prostatectomy-tissue-mask', 'prostatectomy-tissue-whole-slide-image', 'prostatectomy-tissue-whole-slide-image-1', 'prostatectomy-tissue-whole-slide-image-1-2', 'prostatectomy-tissue-whole-slide-image-2', 'prostatectomy-tissue-whole-slide-image-2-2', 'prostatectomy-tissue-whole-slide-image-3', 'prostatectomy-tissue-whole-slide-image-3-2', 'prostatectomy-tissue-whole-slide-image-4', 'prostatectomy-tissue-whole-slide-image-4-2', 'prostatectomy-tissue-whole-slide-image-5', 'prostatectomy-tissue-whole-slide-image-5-2', 'prostatectomy-tissue-whole-slide-image-6', 'prostatectomy-tissue-whole-slide-image-6-2', 'prostatectomy-tissue-whole-slide-image-7', 'prostatectomy-tissue-whole-slide-image-7-2', 'prostatectomy-tissue-whole-slide-image-8', 'prostatectomy-tissue-whole-slide-image-8-2', 'prostatectomy-tissue-whole-slide-image-9', 'prostatectomy-tissue-whole-slide-image-9-2', 'transverse-hbv-prostate-mri'): interf9_handler,
    }.get(interface_key)

    if handler is None:
        raise RuntimeError(f"No handler found for interface key: {interface_key}")

    return handler()

if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
        raise SystemExit(1)
