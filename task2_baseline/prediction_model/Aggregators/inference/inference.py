# Aggregators/inference/inference.py

import json
import pandas as pd
import torch
from pathlib import Path

from prediction_model.Aggregators.inference.feature_loader import load_features
from prediction_model.Aggregators.inference.model_loader import load_model
from prediction_model.Aggregators.inference.prediction import predict_case

# --- Grand Challenge mount points ---
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
MODEL_DIRECTORY = Path("/opt/ml/model")

# --- Interface slugs ---
WSI_SLUG = "bladder-cancer-tissue-biopsy-whole-slide-image"
PROB_SLUG = "brs3-probability"  # Required GC output key


def main():
    print("[INFO] 🚀 Starting CHIMERA Task 2 Inference")

    # --- Step 1: Load preprocessed clinical data (CSV format) ---
    clinical_csv_path = MODEL_DIRECTORY / "clinical_data.csv"
    if not clinical_csv_path.exists():
        raise FileNotFoundError(f"Missing: {clinical_csv_path}")
    clinical_df = pd.read_csv(clinical_csv_path)

    # --- Step 2: Determine input dimensions using first sample ---
    example_case = clinical_df.iloc[0]["case_id"]
    example_feat_path = MODEL_DIRECTORY / "pathology_features" / f"{example_case}.pt"
    if not example_feat_path.exists():
        raise FileNotFoundError(f"Missing: {example_feat_path}")
    pathology_input_dim = torch.load(example_feat_path).shape[1]
    clinical_input_dim = len(clinical_df.columns) - 2  # Exclude 'case_id' and 'label'

    # --- Step 3: Load ABMIL model and fitted clinical processor ---
    model, clinical_processor = load_model(
        model_dir=MODEL_DIRECTORY,
        pathology_input_dim=pathology_input_dim,
        clinical_input_dim=clinical_input_dim
    )

    # --- Step 4: Load jobs from Grand Challenge ---
    jobs_file = INPUT_DIRECTORY / "predictions.json"
    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing predictions.json at {jobs_file}")
    with open(jobs_file, "r") as f:
        jobs = json.load(f)

    # --- Step 5: Run inference for each job ---
    for job in jobs:
        case_id = get_case_id(job)
        print(f"[INFO] 🧪 Processing case: {case_id}")

        # Load pathology + clinical features
        pathology_features, clinical_features = load_features(
            pathology_features_dir=MODEL_DIRECTORY / "pathology_features",
            clinical_df=clinical_df,
            case_id=case_id
        )

        # Apply clinical preprocessor
        if clinical_processor:
            clinical_features = torch.tensor(
                clinical_processor.transform([clinical_features.numpy()])[0],
                dtype=torch.float32
            )

        # Predict probability and binary class
        prob, pred_label = predict_case(
            model=model,
            pathology_features=pathology_features,
            clinical_features=clinical_features
        )

        # Save result to Grand Challenge format
        case_output_dir = OUTPUT_DIRECTORY / str(job["pk"]) / "output"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        with open(case_output_dir / f"{PROB_SLUG}.json", "w") as f:
            json.dump(prob, f, indent=4)

    print("[INFO] ✅ Inference complete.")


def get_case_id(job):
    for value in job["inputs"]:
        if value["interface"]["slug"] == WSI_SLUG:
            return value["image"]["name"]
    raise RuntimeError("❌ Could not find case ID in job inputs")


if __name__ == "__main__":
    main()
