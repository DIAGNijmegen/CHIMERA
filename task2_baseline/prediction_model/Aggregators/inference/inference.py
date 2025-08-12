import json
import torch
from pathlib import Path

from prediction_model.Aggregators.inference.feature_loader import load_features
from prediction_model.Aggregators.inference.model_loader import load_model
from prediction_model.Aggregators.inference.prediction import predict_case
from prediction_model.Aggregators.inference.preprocess_clinical import load_features_tensor_for_case

# --- Grand Challenge mount points ---
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
MODEL_DIRECTORY = Path("/opt/ml/model")

# --- Interface slugs ---
WSI_SLUG = "bladder-cancer-tissue-biopsy-whole-slide-image"
CLINICAL_SLUG = "bladder-cancer-tissue-biopsy-clinical-data"  # adjust to GC interface slug for clinical
PROB_SLUG = "brs3-probability"  # Required GC output key


def main():
    print("[INFO] 🚀 Starting CHIMERA Task 2 Inference")

    # --- Step 1: Load jobs (cases to run) ---
    jobs_file = INPUT_DIRECTORY / "predictions.json"
    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing predictions.json at: {jobs_file}")
    with open(jobs_file, "r") as f:
        jobs = json.load(f)
    if len(jobs) == 0:
        raise ValueError("No jobs found in predictions.json!")

    # --- Step 2: Determine input dims from first case ---
    first_case_id = get_case_id(jobs[0])
    wsi_path = INPUT_DIRECTORY / "images" / WSI_SLUG / f"{first_case_id}.tiff"
    if not wsi_path.exists():
        raise FileNotFoundError(f"Missing sample WSI: {wsi_path}")

    from common.src.features.pathology.main import extract_feature_dim
    pathology_input_dim = extract_feature_dim(wsi_path)

    # We'll let the clinical input dim be determined from the PKL processor
    from prediction_model.Aggregators.inference.preprocess_clinical import load_processor
    proc_info = load_processor(MODEL_DIRECTORY / "clinical_processor.pkl")
    clinical_input_dim = len(proc_info["categorical_cols"]) + len(proc_info["numerical_cols"])

    # --- Step 3: Load model ---
    model, _ = load_model(
        model_dir=MODEL_DIRECTORY,
        pathology_input_dim=pathology_input_dim,
        clinical_input_dim=clinical_input_dim
    )

    # --- Step 4: Inference loop ---
    for job in jobs:
        case_id = get_case_id(job)
        print(f"[INFO] 🧪 Running case: {case_id}")

        # Pathology features
        pathology_features, _ = load_features(
            input_dir=INPUT_DIRECTORY,
            model_dir=MODEL_DIRECTORY,
            clinical_df=None,  # no longer passing prebuilt CSV
            case_id=case_id
        )

        # Clinical JSON path
        clinical_json_path = get_clinical_json_path(job)
        if not clinical_json_path.exists():
            raise FileNotFoundError(f"Missing clinical JSON for case {case_id}: {clinical_json_path}")

        # Apply PKL processor to JSON → clinical tensor
        clinical_tensor, sample_id, _ = load_features_tensor_for_case(
            json_path=clinical_json_path,
            model_root=MODEL_DIRECTORY,
            keep_id=False,
            device=pathology_features.device
        )

        # Run model
        prob, pred_label = predict_case(model, pathology_features, clinical_tensor)

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


def get_clinical_json_path(job) -> Path:
    """Get the Path to the clinical JSON for this case from GC job spec."""
    for value in job["inputs"]:
        if value["interface"]["slug"] == CLINICAL_SLUG:
            return INPUT_DIRECTORY / "clinical-data" / CLINICAL_SLUG / value["filename"]
    raise RuntimeError("❌ Could not find clinical JSON path in job spec.")


if __name__ == "__main__":
    main()
