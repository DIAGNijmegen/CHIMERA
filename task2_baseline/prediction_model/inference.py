import os
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from mil_models.model_abmil_fusion import ABMIL_Fusion
from sklearn.metrics import accuracy_score, roc_auc_score

# === Dataset for WSI features + clinical data ===
class FusionDataset(Dataset):
    def __init__(self, csv_path, feats_dir, clinical_cols, bag_size=None):
        self.df = pd.read_csv(csv_path)
        self.feats_dir = feats_dir
        self.clinical_cols = clinical_cols
        self.bag_size = bag_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']

        feat_path = os.path.join(self.feats_dir, f"{slide_id}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing feature: {feat_path}")

        x_bag = torch.load(feat_path, map_location='cpu').float()

        # Adjust bag size if needed
        if self.bag_size is not None:
            if x_bag.size(0) > self.bag_size:
                x_bag = x_bag[:self.bag_size]
            elif x_bag.size(0) < self.bag_size:
                pad = self.bag_size - x_bag.size(0)
                x_bag = torch.cat([x_bag, x_bag.new_zeros(pad, x_bag.size(1))], dim=0)

        x_clinical = torch.tensor(row[self.clinical_cols].values.astype(np.float32))
        return x_bag, x_clinical, slide_id, row.get("label", None)  # Label may not exist in test

# === Main inference function ===
def run_inference(input_dir: Path, output_dir: Path, debug: bool = False):
    """
    Run inference for Task 2 classification.
    - input_dir: Path to folder containing 'val.csv' and 'features/'
    - output_dir: Where predictions.csv will be saved
    - debug: If True, will print Accuracy/AUC when labels exist
    """

    # Expected input files
    csv_path = input_dir / "val.csv"  # Or test.csv in submission phase
    feats_dir = input_dir / "features"
    checkpoint = Path("/model_weights/best_model.pth")
    config_path = Path("/prediction_model/configs/model_config.json")

    # Determine clinical columns
    df_val = pd.read_csv(csv_path)
    exclude_cols = ['slide_id', 'sample_id', 'SampleID', 'label']
    clinical_cols = [c for c in df_val.columns if c not in exclude_cols]
    print(f"[INFO] Using clinical features: {clinical_cols} ({len(clinical_cols)})")

    # Load model config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Build dataset & loader
    dataset = FusionDataset(csv_path, feats_dir, clinical_cols, bag_size=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Build and load model
    model = ABMIL_Fusion(
        in_dim=config["in_dim"],
        clinical_in_dim=len(clinical_cols),
        n_classes=config["n_classes"],
        gate=config["gate"]
    )
    ckpt = torch.load(checkpoint, map_location='cpu')
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Storage for predictions
    all_labels = []
    all_probs = []
    all_preds = []
    case_ids = []

    with torch.no_grad():
        for x_bag, x_clinical, case_id, label in loader:
            logits = model(x_bag, x_clinical)['logits']
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Class 1 = BRS3
            pred_label = int(probs >= 0.5)

            case_ids.append(case_id[0])
            all_probs.append(float(probs))
            all_preds.append(pred_label)

            if label is not None and not pd.isna(label).any():
                all_labels.append(int(label.item()))

    # Save predictions
    results_df = pd.DataFrame({
        "case_id": case_ids,
        "predicted_label": all_preds,
        "probability": all_probs
    })
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"[INFO] Predictions saved to {output_dir/'predictions.csv'}")

    # Optional debug metrics if labels exist
    if debug and len(all_labels) > 0:
        all_labels = np.array(all_labels)
        preds_np = np.array(all_preds)
        probs_np = np.array(all_probs)

        acc = accuracy_score(all_labels, preds_np)
        try:
            auc = roc_auc_score(all_labels, probs_np)
        except ValueError:
            auc = None

        print(f"[DEBUG] Accuracy: {acc:.4f}")
        if auc is not None:
            print(f"[DEBUG] ROC AUC: {auc:.4f}")
