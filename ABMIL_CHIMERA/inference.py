import json
from pathlib import Path
import torch

from prediction_model.Aggregators.inference.feature_loader import load_features
from prediction_model.Aggregators.inference.model_loader import load_model
from prediction_model.Aggregators.inference.preprocess_clinical import (
    load_features_tensor_for_case,
    load_processor,
)

# --- Grand Challenge mount points ---
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
MODEL_DIRECTORY = Path("/opt/ml/model")

# --- Interface slugs ---
WSI_SLUG = "bladder-cancer-tissue-biopsy-whole-slide-image"
MASK_SLUG = "bladder-cancer-tissue-biopsy-whole-slide-image-mask"
CLINICAL_SLUG = "bladder-cancer-tissue-biopsy-clinical-data"
PROB_SLUG = "brs3-probability"  # Required GC output key

# Where UNI/Slide2Vec extractor weights live (model.bin + config.json)
UNI_WEIGHTS_DIR = Path("/opt/app/common/model/pathology")


def main():
    print("[INFO] 🚀 Starting CHIMERA Task 2 Inference")

    # --- Step 1: Load jobs file ---
    jobs_file = INPUT_DIRECTORY / "predictions.json"
    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing predictions.json at: {jobs_file}")
    jobs = json.loads(jobs_file.read_text())
    if not jobs:
        raise ValueError("No jobs found in predictions.json!")

    # --- Step 2: Clinical input dim from PKL processor ---
    proc_info = load_processor(MODEL_DIRECTORY / "clinical_processor.pkl")
    clinical_input_dim = len(proc_info["categorical_cols"]) + len(proc_info["numerical_cols"])

    # Pre-compute the clinical root dir
    clinical_root = INPUT_DIRECTORY / "clinical-data" / CLINICAL_SLUG

    # --- Step 3: Loop over jobs ---
    for job in jobs:
        case_id, wsi_path, mask_path, clinical_filename = parse_job_paths(job)

        print(f"[INFO] 🧪 Running case: {case_id}")
        if not wsi_path.exists():
            raise FileNotFoundError(f"Missing WSI: {wsi_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing MASK: {mask_path}")

        clinical_json_path = clinical_root / clinical_filename
        if not clinical_json_path.exists():
            raise FileNotFoundError(f"Missing clinical JSON: {clinical_json_path}")

        # Extract WSI features on the fly (UNI)
        pathology_features, pathology_input_dim = load_features(
            case_id=case_id,
            wsi_path=wsi_path,
            mask_path=mask_path,
            model_dir=UNI_WEIGHTS_DIR,
        )

        # Load ABMIL_Fusion model (fusion of pathology + clinical)
        model, _ = load_model(
            model_dir=MODEL_DIRECTORY,
            pathology_input_dim=pathology_input_dim,
            clinical_input_dim=clinical_input_dim,
        )

        # Clinical JSON -> tensor (uses the saved processor)
        clinical_tensor = load_features_tensor_for_case(
            case_id=case_id,
            clinical_dir=clinical_root,
            clinical_processor=proc_info,
            device=str(pathology_features.device),
        )

        # Predict
        prob, pred_label = predict_case(model, pathology_features, clinical_tensor)

        # Save to GC format
        case_output_dir = OUTPUT_DIRECTORY / str(job["pk"]) / "output"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        (case_output_dir / f"{PROB_SLUG}.json").write_text(json.dumps(prob, indent=4))

    print("[INFO] ✅ All cases processed.")


def parse_job_paths(job):
    """Extract (case_id, WSI path, mask path, clinical filename) from a GC job."""
    case_id = None
    wsi_path = None
    mask_path = None
    clinical_filename = None

    for value in job["inputs"]:
        slug = value["interface"]["slug"]
        if slug == WSI_SLUG:
            # Prefer explicit filename if provided, else build from image.name and try common exts
            base = INPUT_DIRECTORY / "images" / WSI_SLUG
            if "filename" in value:
                p = base / value["filename"]
                if p.exists():
                    # derive case_id from filename stem
                    case_id = Path(value["filename"]).stem
                    wsi_path = p
                else:
                    raise FileNotFoundError(f"WSI filename given but not found: {p}")
            else:
                case_id = value["image"]["name"]
                for ext in (".tiff", ".tif", ".svs"):
                    p = base / f"{case_id}{ext}"
                    if p.exists():
                        wsi_path = p
                        break
        elif slug == MASK_SLUG:
            mask_path = INPUT_DIRECTORY / "images" / MASK_SLUG / value["filename"]
        elif slug == CLINICAL_SLUG:
            clinical_filename = value["filename"]

    if case_id is None or wsi_path is None or mask_path is None or clinical_filename is None:
        raise RuntimeError("❌ Missing one or more required inputs (WSI/mask/clinical).")

    return case_id, wsi_path, mask_path, clinical_filename


def predict_case(model, pathology_features, clinical_tensor):
    """Run ABMIL_Fusion and return probability + predicted label."""
    model.eval()
    with torch.no_grad():
        # [N, D] -> batch of 1: [1, N, D] if your model expects MIL bag,
        # adjust if your forward() expects a different shape.
        prob = model(pathology_features.unsqueeze(0), clinical_tensor)
        prob = torch.sigmoid(prob).item()
        pred_label = int(prob >= 0.5)
    return prob, pred_label


if __name__ == "__main__":
    main()
