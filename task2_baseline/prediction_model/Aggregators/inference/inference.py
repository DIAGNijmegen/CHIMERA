# Aggregators/inference/inference.py
import json
import pandas as pd
import torch
from pathlib import Path
from prediction_model.Aggregators.inference.feature_loader import load_features
from prediction_model.Aggregators.inference.model_loader import load_model
from prediction_model.Aggregators.inference.prediction import predict_case

# GC mount points
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
MODEL_DIRECTORY = Path("/opt/ml/model")

# GC interface slugs
WSI_SLUG = "prostatectomy-tissue-whole-slide-image"
PROB_SLUG = "brs3-probability"

def main():
    print("[INFO] Starting Task 2 Classification Inference...")

    # --- Step 1: Load clinical data ---
    clinical_csv_path = MODEL_DIRECTORY / "clinical_data.csv"
    if not clinical_csv_path.exists():
        raise FileNotFoundError(f"Missing clinical_data.csv in {MODEL_DIRECTORY}")
    clinical_df = pd.read_csv(clinical_csv_path)

    # --- Step 2: Detect feature dimensions ---
    example_case = clinical_df.iloc[0]["case_id"]
    example_feature_path = MODEL_DIRECTORY / "pathology_features" / f"{example_case}.pt"
    if not example_feature_path.exists():
        raise FileNotFoundError(f"Missing example pathology feature file: {example_feature_path}")

    pathology_input_dim = torch.load(example_feature_path).shape[1]
    clinical_input_dim = len(clinical_df.columns) - 2  # minus case_id + label

    # --- Step 3: Load model ---
    model, clinical_processor = load_model(
        model_dir=MODEL_DIRECTORY,
        clinical_input_dim=clinical_input_dim,
        pathology_input_dim=pathology_input_dim
    )

    # --- Step 4: Read GC jobs ---
    predictions_file = INPUT_DIRECTORY / "predictions.json"
    if not predictions_file.exists():
        raise FileNotFoundError(f"Missing predictions.json in {INPUT_DIRECTORY}")
    with open(predictions_file, "r") as f:
        jobs = json.load(f)

    # --- Step 5: Process each case ---
    for job in jobs:
        case_id = get_case_id(job)

        # Load features for case
        pathology_features, clinical_features = load_features(
            pathology_features_dir=MODEL_DIRECTORY / "pathology_features",
            clinical_df=clinical_df,
            case_id=case_id
        )

        # Apply clinical processor if exists
        if clinical_processor:
            clinical_features = torch.tensor(
                clinical_processor.transform([clinical_features.numpy()])[0],
                dtype=torch.float32
            )

        # Predict probability and label
        prob, pred_label = predict_case(model, pathology_features, clinical_features)

        # Save GC-compatible probability JSON
        case_output_dir = OUTPUT_DIRECTORY / job["pk"] / "output"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        with open(case_output_dir / f"{PROB_SLUG}.json", "w") as f:
            f.write(json.dumps(prob))

    print("[INFO] Inference complete.")

def get_case_id(job):
    for value in job["inputs"]:
        if value["interface"]["slug"] == WSI_SLUG:
            return value["image"]["name"]
    raise RuntimeError(f"Case ID not found in job: {job}")

if __name__ == "__main__":
    main()
