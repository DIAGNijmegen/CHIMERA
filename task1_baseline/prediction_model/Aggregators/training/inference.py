import torch
import torch.nn.functional as F
import argparse
import os
import json
import pandas as pd
import glob
import numpy as np
import SimpleITK
# import pyvips
from pathlib import Path
from tqdm import tqdm
import traceback
import sys
import pickle

# --- Add Project Root to Python Path ---
# This is a robust fix for the ModuleNotFoundError. It adds both the project root
# ('task1_baseline') and the 'Aggregators' directory to Python's path.
# This ensures that top-level imports ('from Aggregators...'), internal relative
# imports ('from utils...'), and unpickling all work correctly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
aggregators_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, aggregators_path)


# --- Import model-specific components ---
# These imports will now work correctly.
from Aggregators.mil_models import create_downstream_model
from Aggregators.wsi_datasets.clinical_processor import ClinicalDataProcessor


# --- Grand Challenge Standard Paths ---
# Uncomment these lines for the final container
# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
# MODEL_PATH = Path("/opt/ml/model/")

# --- Paths for Local Testing ---
# Use these paths when running the script on your local machine or cluster
INPUT_PATH = Path("/data/temporary/chimera/Baseline_models/Task1_ABMIL/inference_test_input/input/1026")
OUTPUT_PATH = Path("/data/temporary/chimera/Baseline_models/Task1_ABMIL/inference_test_input/output")
MODEL_PATH = Path("/data/temporary/chimera/Baseline_models/Task1_ABMIL/inference_test_input/model")

# --- Define expected feature directories (placeholders until feature extraction is integrated) ---
# In a real run, these features would be generated from the raw data.
# For local testing, you would place your pre-computed features in these subdirectories inside ./test/input/
PATHOLOGY_FEATURES_DIR = INPUT_PATH / "features/pathology"
RADIOLOGY_FEATURES_DIR = INPUT_PATH / "features/radiology"
CLINICAL_DIR = INPUT_PATH # Clinical JSON is at the root of /input/

def run_prediction(clinical_data_json: dict, all_wsi_paths: list, mri_paths: dict):
    """
    Core function to run the multi-modal prediction pipeline.
    This function is called by the specific interface handlers.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # --- Step 1: Load Pre-trained Model, Processor, and CALIBRATION DATA ---
    print("--- Step 1: Loading Model and All Assets ---")
    
    checkpoint_path = MODEL_PATH / "s_checkpoint.pth"
    clinical_processor_path = MODEL_PATH / "clinical_processor.pkl"
    config_json_path = MODEL_PATH / "config.json"
    calibration_data_path = MODEL_PATH / "calibration_data.pkl" # NEW

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    if not clinical_processor_path.exists():
        raise FileNotFoundError(f"Fitted clinical processor not found at {clinical_processor_path}")
    if not calibration_data_path.exists():
        raise FileNotFoundError(f"Calibration data not found at {calibration_data_path}. Please run create_calibration_data.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    with open(clinical_processor_path, 'rb') as f:
        clinical_processor = pickle.load(f)
    with open(calibration_data_path, 'rb') as f:
        calibration_data = pickle.load(f) # NEW

    # --- Robustly load model configuration and weights ---
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        saved_weights = checkpoint['model']
    else:
        if not config_json_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_json_path}.")
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)
        saved_weights = checkpoint
    
    config_args = argparse.Namespace(**config_dict)
    config_args.clinical_processor = clinical_processor
    
    model = create_downstream_model(args=config_args, mode='survival')
    model.load_state_dict(saved_weights)
    model.eval()
    print("Model, processor, and calibration data loaded successfully.")

    # --- Step 2: Identify Case ID and Process Inputs ---
    print("\n--- Step 2: Identifying Case and Processing Inputs ---")
    
    case_id = clinical_data_json['case_id']
    print(f"Processing Case ID: {case_id}")

    # --- Load Pathology Features ---
    pathology_feature_files = [fp for wsi_path in all_wsi_paths for fp in glob.glob(str(PATHOLOGY_FEATURES_DIR / f"{case_id}_*.pt"))]
    if not pathology_feature_files:
        raise FileNotFoundError(f"CRITICAL: No pathology feature files found for case {case_id} in {PATHOLOGY_FEATURES_DIR}")
    
    all_pathology_features = [torch.load(fp) for fp in sorted(list(set(pathology_feature_files)))]
    pathology_features_tensor = torch.cat(all_pathology_features, dim=0).unsqueeze(0)
    print(f"Loaded {len(all_pathology_features)} pathology feature file(s).")

    # --- Load Radiology Features ---
    radiology_features_tensor = None
    radiology_files = glob.glob(str(RADIOLOGY_FEATURES_DIR / f"{case_id}_*.pt"))
    if radiology_files:
        radiology_features_tensor = torch.load(radiology_files[0])
        if len(radiology_features_tensor.shape) == 1:
            radiology_features_tensor = radiology_features_tensor.unsqueeze(0)
        print(f"Loaded radiology features.")
    else:
        print("Warning: No radiology feature file found.")

    # --- Process Clinical Data ---
    clinical_features_tensor = None
    if clinical_processor:
        clinical_features_tensor = clinical_processor.transform(case_id).unsqueeze(0)
        print(f"Processed clinical data.")
    else:
        print("Warning: Clinical processor not available.")

    # --- Step 3: Perform Inference ---
    print("\n--- Step 3: Running Model Inference ---")
    with torch.no_grad():
        output_dict = model.forward_no_loss(
            h=pathology_features_tensor,
            additional_embeddings=radiology_features_tensor,
            clinical_features=clinical_features_tensor
        )
    
    logits = output_dict['logits']
    risk_score = torch.exp(logits).item()
    print(f"Predicted Risk Score: {risk_score:.4f}")

    # --- Step 4: Calibrating Risk to Time-to-Event ---
    print("\n--- Step 4: Calibrating Risk to Time-to-Event ---")
    
    try:
        # Calculate patient's personalized survival curve from the baseline
        hazard_ratio = risk_score
        patient_survival_probs = calibration_data['baseline_survival'] ** hazard_ratio
        time_points = calibration_data['time_points']

        # Flip arrays for interpolation, as np.interp needs increasing x-values
        time_points_interp = np.flip(time_points)
        patient_survival_probs_interp = np.flip(patient_survival_probs)
        
        min_survival_prob = patient_survival_probs_interp[0]

        if 0.5 > min_survival_prob:
            # --- HIGH-RISK CASE ---
            # The curve drops below 50%. Find the median time directly.
            predicted_time_months = np.interp(0.5, patient_survival_probs_interp, time_points_interp)
            print(f"Predicted Median Time to Recurrence: {predicted_time_months:.2f} months")
        else:
            # --- LOW-RISK CASE (Extrapolation) ---
            # The curve bottoms out above 50%. Extrapolate to get an uncapped time.
            print("Patient is low risk. Extrapolating time beyond max follow-up.")
            # Use the last two points of the curve to find the slope
            y2, y1 = patient_survival_probs[-1], patient_survival_probs[-2]
            x2, x1 = time_points[-1], time_points[-2]
            
            # Avoid division by zero if the curve is flat at the end
            if (x2 - x1) == 0 or (y2 - y1) == 0:
                 predicted_time_months = x2 * 1.1
            else:
                # Extrapolate using the point-slope formula to find time (x) at 50% prob (y)
                m = (y2 - y1) / (x2 - x1)
                predicted_time_months = x2 + (0.5 - y2) / m

            # Ensure prediction is not negative if slope is positive
            if predicted_time_months < x2:
                 predicted_time_months = x2
            print(f"Extrapolated Time to Recurrence: {predicted_time_months:.2f} months")

    except Exception as e:
        print(f"Could not calculate median survival time due to: {e}. Defaulting to -1.")
        predicted_time_months = -1.0

    # --- Step 5: Save Final Prediction ---
    print("\n--- Step 5: Saving Prediction ---")
    output_path = OUTPUT_PATH / "time-to-biochemical-recurrence-for-prostate-cancer-months.json"
    write_json_file(location=output_path, content=predicted_time_months)
    print(f"Successfully saved prediction to {output_path}")

# =============================================================================
# Grand Challenge I/O and Interface Handling Boilerplate
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
    # This is the main entry point for the container.
    # It will call the `run` function, which in turn will select the correct
    # handler based on the inputs provided to the container.
    try:
        raise SystemExit(run())
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
        raise SystemExit(1)
