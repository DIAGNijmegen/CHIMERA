import torch
import torch.nn.functional as F
import argparse
import os
import json
import pandas as pd
import glob
import numpy as np
import SimpleITK
import pyvips
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
INPUT_PATH = Path("/data/temporary/chimera/Baseline_models/Task1_ABMIL/inference_test_input/input")
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
    
    # --- Step 1: Load Pre-trained Model and Fitted Clinical Processor ---
    print("--- Step 1: Loading Model and Pre-fitted Clinical Processor ---")
    
    checkpoint_path = MODEL_PATH / "s_checkpoint.pth"
    clinical_processor_path = MODEL_PATH / "clinical_processor.pkl"
    config_json_path = MODEL_PATH / "config.json" # Path to the standalone config file

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    if not clinical_processor_path.exists():
        raise FileNotFoundError(f"Fitted clinical processor not found at {clinical_processor_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    with open(clinical_processor_path, 'rb') as f:
        clinical_processor = pickle.load(f)

    # --- Robustly load model configuration and weights ---
    if 'config' in checkpoint:
        # Case 1: Config is saved inside the checkpoint (trained with early stopping)
        print("Loading configuration from checkpoint file.")
        config_dict = checkpoint['config']
        saved_weights = checkpoint['model']
    else:
        # Case 2: Config is not in checkpoint (trained without early stopping)
        # Load config from the separate config.json and assume checkpoint is just the state_dict
        print("Configuration not in checkpoint. Loading from separate config.json.")
        if not config_json_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_json_path}. It is required when config is not in the checkpoint.")
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)
        saved_weights = checkpoint
    
    config_args = argparse.Namespace(**config_dict)
    
    # --- FIX: Attach the loaded clinical processor to the config arguments ---
    # The model factory (`create_downstream_model`) uses this object to get the
    # correct clinical feature dimension, ensuring the model architecture matches.
    config_args.clinical_processor = clinical_processor
    
    model = create_downstream_model(args=config_args, mode='survival')
    model.load_state_dict(saved_weights)
    model.eval()
    print("Model and clinical processor loaded successfully.")

    # --- Step 2: Identify Case ID and Process Inputs ---
    print("\n--- Step 2: Identifying Case and Processing Inputs ---")
    
    # The case_id is derived from the clinical JSON filename.
    case_id = clinical_data_json['case_id']
    print(f"Processing Case ID: {case_id}")

    # --- Load Pathology Features ---
    # TODO: This section will be replaced by the WSI feature extractor.
    # For now, it loads pre-computed features.
    pathology_feature_files = [fp for wsi_path in all_wsi_paths for fp in glob.glob(str(PATHOLOGY_FEATURES_DIR / f"{case_id}_*.pt"))]
    if not pathology_feature_files:
        raise FileNotFoundError(f"CRITICAL: No pathology feature files found for case {case_id} in {PATHOLOGY_FEATURES_DIR}")
    
    all_pathology_features = [torch.load(fp) for fp in sorted(list(set(pathology_feature_files)))] # Use set to avoid duplicates
    pathology_features_tensor = torch.cat(all_pathology_features, dim=0).unsqueeze(0)
    print(f"Loaded and concatenated {len(all_pathology_features)} pathology feature file(s). Final shape: {pathology_features_tensor.shape}")
    # DEBUG: Save the processed tensor
    torch.save(pathology_features_tensor, OUTPUT_PATH / f"{case_id}_pathology_features.pt")
    print(f"DEBUG: Saved pathology tensor to {OUTPUT_PATH / f'{case_id}_pathology_features.pt'}")


    # --- Load Radiology Features ---
    # TODO: This section will be replaced by the MRI feature extractor.
    radiology_features_tensor = None
    radiology_files = glob.glob(str(RADIOLOGY_FEATURES_DIR / f"{case_id}_*.pt"))
    if radiology_files:
        radiology_features_tensor = torch.load(radiology_files[0])
        if len(radiology_features_tensor.shape) == 1:
            radiology_features_tensor = radiology_features_tensor.unsqueeze(0)
        print(f"Loaded radiology features. Shape: {radiology_features_tensor.shape}")
        # DEBUG: Save the processed tensor
        torch.save(radiology_features_tensor, OUTPUT_PATH / f"{case_id}_radiology_features.pt")
        print(f"DEBUG: Saved radiology tensor to {OUTPUT_PATH / f'{case_id}_radiology_features.pt'}")
    else:
        print("Warning: No radiology feature file found. Proceeding without it.")

    # --- Process Clinical Data ---
    clinical_features_tensor = None
    if clinical_processor:
        clinical_features_tensor = clinical_processor.transform(case_id).unsqueeze(0)
        print(f"Processed clinical data. Shape: {clinical_features_tensor.shape}")
        # DEBUG: Save the processed tensor
        torch.save(clinical_features_tensor, OUTPUT_PATH / f"{case_id}_clinical_features.pt")
        print(f"DEBUG: Saved clinical tensor to {OUTPUT_PATH / f'{case_id}_clinical_features.pt'}")
    else:
        print("Warning: Clinical processor not available. Proceeding without clinical data.")

    # --- Step 3: Perform Inference ---
    print("\n--- Step 3: Running Model Inference ---")
    with torch.no_grad():
        output_dict = model.forward_no_loss(
            h=pathology_features_tensor,
            additional_embeddings=radiology_features_tensor,
            clinical_features=clinical_features_tensor
        )
    
    # DEBUG: Save the entire model output dictionary (includes logits, attention, etc.)
    torch.save(output_dict, OUTPUT_PATH / f"{case_id}_model_outputs.pt")
    print(f"DEBUG: Saved full model output dictionary to {OUTPUT_PATH / f'{case_id}_model_outputs.pt'}")

    logits = output_dict['logits']
    risk_score = torch.exp(logits).item()
    print(f"Predicted Risk Score: {risk_score:.4f}")

    # --- Step 4: Save Output in Grand Challenge Format ---
    print("\n--- Step 4: Saving Prediction ---")
    output_path = OUTPUT_PATH / "time-to-biochemical-recurrence-for-prostate-cancer-months.json"
    write_json_file(location=output_path, content=risk_score)
    print(f"Successfully saved prediction to {output_path}")

# =============================================================================
# Grand Challenge I/O and Interface Handling Boilerplate
# =============================================================================

def interf_handler_template(wsi_slugs):
    """A template handler to process inputs for a given number of WSIs."""
    print(f"Handling interface with {len(wsi_slugs)} WSI file(s).")
    
    # --- Load Raw Data (as per GC template) ---
    # This part loads the raw images and clinical data.
    mri_paths = {
        't2': INPUT_PATH / "images/axial-t2-prostate-mri",
        'adc': INPUT_PATH / "images/axial-adc-prostate-mri",
        'hbv': INPUT_PATH / "images/transverse-hbv-prostate-mri",
    }
    
    all_wsi_paths = [INPUT_PATH / f"images/{slug}" for slug in wsi_slugs]
    
    # --- FIX: Find the clinical JSON file, explicitly ignoring 'inputs.json' ---
    clinical_files = [f for f in glob.glob(str(INPUT_PATH / "*.json")) if Path(f).name != 'inputs.json']
    if not clinical_files:
        # Grand Challenge provides a specific clinical data file, so this check is for robustness.
        raise FileNotFoundError(f"No clinical JSON file found in '{INPUT_PATH}'")
    
    clinical_file_path = Path(clinical_files[0])
    loaded_json = load_json_file(location=clinical_file_path)
    
    # FIX: Handle cases where the JSON is a list containing a single dictionary
    if isinstance(loaded_json, list):
        if len(loaded_json) == 0:
            raise ValueError(f"Clinical JSON file at {clinical_file_path} is an empty list.")
        clinical_data_dict = loaded_json[0]
    else:
        clinical_data_dict = loaded_json
    
    # Extract the case_id from the filename. This handles both simple names (e.g., "1003.json")
    # and the full Grand Challenge name.
    case_id = clinical_file_path.stem.replace("-clinical-data-of-prostate-cancer-patients", "")
    clinical_data_dict['case_id'] = case_id

    # --- Feature Extraction (Placeholder) ---
    # TODO: Insert MRI feature extraction logic here. It should process raw MRI images
    # from mri_paths and save a single .pt file to RADIOLOGY_FEATURES_DIR.
    print("Placeholder: MRI Feature Extraction would run here.")

    # TODO: Insert WSI feature extraction logic here. It should process raw WSI files
    # from all_wsi_paths and save one or more .pt files to PATHOLOGY_FEATURES_DIR.
    print("Placeholder: WSI Feature Extraction would run here.")

    # --- Run Prediction on Extracted Features ---
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
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()
    
    # This dictionary maps the set of expected input files to the correct handler.
    # It is designed to handle cases with a variable number of WSI files.
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
