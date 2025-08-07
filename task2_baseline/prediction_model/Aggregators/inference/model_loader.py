# Aggregators/inference/model_loader.py

import torch
import pickle
import json
from pathlib import Path
from prediction_model.Aggregators.mil_models.model_abmil_fusion import ABMIL_Fusion


def load_model(model_dir: Path, clinical_input_dim: int, pathology_input_dim: int):
    """
    Load Task 2 ABMIL classification model and the clinical processor if available.
    """

    # --- Step 1: Load clinical preprocessor (if exists) ---
    clinical_processor_path = model_dir / "clinical_processor.pkl"
    clinical_processor = None
    if clinical_processor_path.exists():
        with open(clinical_processor_path, "rb") as f:
            clinical_processor = pickle.load(f)
        print("[INFO] Loaded clinical preprocessor.")
    else:
        print("[INFO] No clinical preprocessor found.")

    # --- Step 2: Load model config ---
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # --- Step 3: Initialize model ---
    model = ABMIL_Fusion(
        in_dim=pathology_input_dim,
        clinical_in_dim=clinical_input_dim,
        n_classes=config.get("n_classes", 1),
        gate=config.get("gate", True)
    )
    print("[INFO] Initialized ABMIL_Fusion model.")

    # --- Step 4: Load model weights ---
    ckpt_path = model_dir / "s_checkpoint.pth"  # ✅ match your file name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model weights: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Handle if checkpoint is either plain state_dict or dict with "model" key
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("[INFO] Model weights loaded and set to eval mode.")

    return model, clinical_processor
