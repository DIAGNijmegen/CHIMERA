# Aggregators/inference/model_loader.py
import torch
import pickle
from pathlib import Path
from prediction_model.Aggregators.mil_models.model_abmil_fusion import ABMIL_Fusion

def load_model(model_dir: Path, clinical_input_dim: int, pathology_input_dim: int):
    """
    Load Task 2 ABMIL classification model and optional clinical processor.
    """
    # Load clinical processor (scaler, encoder)
    clinical_processor_path = model_dir / "clinical_processor.pkl"
    clinical_processor = None
    if clinical_processor_path.exists():
        with open(clinical_processor_path, "rb") as f:
            clinical_processor = pickle.load(f)

    # Load model config
    config_path = model_dir / "config.json"
    import json
    with open(config_path, "r") as f:
        config = json.load(f)

    # Init model
    model = ABMIL_Fusion(
        in_dim=pathology_input_dim,
        clinical_in_dim=clinical_input_dim,
        n_classes=config["n_classes"],
        gate=config["gate"]
    )

    # Load weights
    ckpt_path = model_dir / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, clinical_processor

