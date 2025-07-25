
import torch
import pickle
import json
import argparse
from pathlib import Path

from Aggregators.mil_models import create_downstream_model

def load_model_and_assets(model_path: Path, clinical_data_path: Path):
    """
    Loads the pre-trained model, clinical processor, and calibration data.

    Args:
        model_path: Path to the directory containing model assets.
        clinical_data_path: Path to the directory containing clinical JSON files.

    Returns:
        A tuple containing:
        - The loaded model.
        - The clinical data processor.
        - The calibration data.
    """
    checkpoint_path = model_path / "s_checkpoint.pth"
    clinical_processor_path = model_path / "clinical_processor.pkl"
    config_json_path = model_path / "config.json"
    calibration_data_path = model_path / "calibration_data.pkl"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    if not clinical_processor_path.exists():
        raise FileNotFoundError(f"Fitted clinical processor not found at {clinical_processor_path}")
    if not calibration_data_path.exists():
        raise FileNotFoundError(f"Calibration data not found at {calibration_data_path}. Please run create_calibration_data.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    with open(clinical_processor_path, 'rb') as f:
        clinical_processor = pickle.load(f)
    
    # Set the clinical_data_path for the loaded processor
    clinical_processor.clinical_data_path = clinical_data_path

    with open(calibration_data_path, 'rb') as f:
        calibration_data = pickle.load(f)

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
    return model, clinical_processor, calibration_data
