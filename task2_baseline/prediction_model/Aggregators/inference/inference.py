# Aggregators/inference/inference.py
import json
import pandas as pd
from pathlib import Path
from prediction_model.Aggregators.inference.feature_loader import load_features
from prediction_model.Aggregators.inference.model_loader import load_model
from prediction_model.Aggregators.inference.prediction import predict_case

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
MODEL_DIRECTORY = Path("/opt/ml/model")

# GC interface slugs
WSI_SLUG = "prostatectomy-tissue-whole-slide-image"
PROB_SLUG = "brs3-probability"

def main():
    print("[INFO] Starting Task 2 Classification Inference...")

    # Load clinical table (processed CSV must be inside container or pre-generated)
    clinical_df = pd.read_csv(MODEL_DIRECTORY / "clinical_data.csv")

    # Detect feature dimension
    example_case = clinical_df.iloc[0]["case_id"]
    pathology_input_dim = torch.load(MODEL_DIRECTORY / f"{example_case}.pt").shape[1]
    clinical_input_dim = len(clinical_df.columns) - 2  # minus case_id + label

    # Load model
    model, clinical_processor = load_model(
        model_dir=MODEL_DIRECTORY,
        clinical_input_dim=clinical_input_dim,
        pathology_input_dim=pathology_input_dim
    )

    # Read GC predictions.json
    with open(INPUT_DIRECTORY / "predictions.json", "r") as f:
        jobs = json.load(f)

    # Process each case
    for job in jobs:
        case_id = get_case_id(job)
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

        # Predict
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

