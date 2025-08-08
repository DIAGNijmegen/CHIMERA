import json
from pathlib import Path
from typing import Optional
import torch
import numpy as np

def load_features_tensor_for_case(
    case_id: str,
    clinical_dir: Path,
    clinical_processor: Optional[dict] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Loads a single case's clinical features and returns as [1, D] tensor.

    Args:
        case_id: e.g., "1001"
        clinical_dir: path to directory with case JSON files
        clinical_processor: dict loaded from clinical_processor.pkl
        device: "cpu" or "cuda"
    """
    # --- Step 1: load JSON file ---
    json_path = Path(clinical_dir) / f"{case_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No clinical JSON found for case {case_id}: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # --- Step 2: get feature order from processor ---
    if clinical_processor is None:
        raise ValueError("clinical_processor is required for inference.")
    categorical_cols = clinical_processor["categorical_cols"]
    numerical_cols = clinical_processor["numerical_cols"]
    missing_code = clinical_processor["missing_code"]

    # --- Step 3: encode categoricals ---
    encoders = clinical_processor["encoders"]
    cat_feats = []
    for col in categorical_cols:
        val = data.get(col, None)
        if val is None or str(val).strip() == "":
            cat_feats.append(missing_code)
        else:
            le = encoders[col]
            if str(val) in le.classes_:
                cat_feats.append(int(np.where(le.classes_ == str(val))[0][0]))
            else:
                cat_feats.append(missing_code)

    # --- Step 4: scale numerics ---
    scaler = clinical_processor["scaler"]
    num_feats = []
    for col in numerical_cols:
        val = data.get(col, None)
        try:
            num_feats.append(float(val))
        except (TypeError, ValueError):
            num_feats.append(np.nan)
    num_feats = np.array(num_feats, dtype=np.float32).reshape(1, -1)
    num_feats = scaler.transform(num_feats).flatten().tolist()

    # --- Step 5: concatenate & tensorize ---
    feats = np.array(cat_feats + num_feats, dtype=np.float32).reshape(1, -1)
    feats_tensor = torch.from_numpy(feats).to(device)
    return feats_tensor
