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

    # --- Step 1: Load clinical data ---
    clinical_csv_path = MODEL_DIRECTORY / "clinical_data.csv"
    if not clinical_csv_path.exists():
        raise FileNotFoundError(f"Missing clinical CSV: {clinical_csv_path}")
    clinical_df = pd.read_csv(clinical_csv_path)

    # --- Step 2: Load jobs (cases to run) ---
    jobs_file = INPUT_DIRECTORY / "predictions.json"
    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing predictions.json at: {jobs_file}")
    with open(jobs_file, "r") as f:
        jobs = json.load(f)

    if len(jobs) == 0:
        raise ValueError("No jobs found in predictions.json!")

    # --- Step 3: Determine model input dims from first case ---
    first_case_id = get_case_id(jobs[0])
    wsi_path = INPUT_DIRECTORY / "images" / WSI_SLUG / f"{first_case_id}.tiff"
    if not wsi_path.exists():
        raise FileNotFoundError(f"Missing sample WSI: {wsi_path}")

    from common.src.features.pathology.main import extract_feature_dim
    pathology_input_dim = extract_feature_dim(wsi_path)  # ← your custom extractor should support this
    clinical_input_dim = len(clinical_df.columns) - 2

    # --- Step 4: Load model ---
    model, clinical_processor = load_model(
        model_dir=MODEL_DIRECTORY,
        pathology_input_dim=pathology_input_dim,
        clinical_input_dim=clinical_input_dim
    )

    # --- Step 5: Inference loop ---
    for job in jobs:
        case_id = get_case_id(job)
        print(f"[INFO] 🧪 Running case: {case_id}")

        # Load features from WSI + mask + clinical CSV
        pathology_features, clinical_features = load_features(
            input_dir=INPUT_DIRECTORY,
            model_dir=MODEL_DIRECTORY,
            clinical_df=clinical_df,
            case_id=case_id
        )

        # Apply clinical processor
        if clinical_processor:
            clinical_features = torch.tensor(
                clinical_processor.transform([clinical_features.numpy()])[0],
                dtype=torch.float32
            )

        # Run model
        prob, pred_label = predict_case(model, pathology_features, clinical_features)

        # Save output
        case_output_dir = OUTPUT_DIRECTORY / str(job["pk"]) / "output"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        with open(case_output_dir / f"{PROB_SLUG}.json", "w") as f:
            json.dump(prob, f, indent=4)

    print("[INFO] ✅ Inference completed successfully.")


def get_case_id(job):
    for value in job["inputs"]:
        if value["interface"]["slug"] == WSI_SLUG:
            return value["image"]["name"]
    raise RuntimeError("❌ Could not extract case ID from job.")


if __name__ == "__main__":
    main()
