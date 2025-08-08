# task2_baseline/prediction_model/Aggregators/inference/model_loader.py

import torch
import pickle
import json
from pathlib import Path
from prediction_model.Aggregators.inference.abmil_infer_model import ABMIL_Fusion


def load_model(model_dir: Path):
    """
    Load Task 2 ABMIL classification model and the clinical processor if available.
    Model dimensions are read from config.json in model_dir.
    """

    # --- Step 1: Load clinical preprocessor (if exists) ---
    clinical_processor = None
    clinical_processor_path = model_dir / "clinical_processor.pkl"
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
    print("[INFO] Loaded model config.")

    # --- Step 3: Initialize model ---
    model = ABMIL_Fusion(
        in_dim=config["wsi_dim"],                # pathology features input dim
        clinical_in_dim=config["clinical_dim"],  # clinical features input dim
        n_classes=config.get("n_classes", 1),
        gate=config.get("gate", True),
        dropout_p=config.get("dropout", 0.5)
    )
    print("[INFO] Initialized ABMIL_Fusion model.")

    # --- Step 4: Load model weights ---
    ckpt_path = model_dir / "s_checkpoint.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model weights: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Handle both plain state_dict and checkpoint dict with "model" key
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print("[INFO] Model weights loaded and set to eval mode.")

    return model, clinical_processor
