import os
import argparse
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from mil_models.model_abmil_fusion import ABMIL_Fusion
from sklearn.metrics import accuracy_score, roc_auc_score

# === Dataset class ===
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
        label_str = row['label']
        label = 1 if label_str == 'BRS3' else 0  # BRS3 = 1, BRS1/2 = 0

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
        return x_bag, x_clinical, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--val_feats_dir', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--bag_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Determine clinical columns from CSV
    df_val = pd.read_csv(args.val_csv)
    exclude_cols = ['slide_id', 'sample_id', 'SampleID', 'label']
    clinical_cols = [c for c in df_val.columns if c not in exclude_cols]
    print(f"[INFO] Using clinical features: {clinical_cols} ({len(clinical_cols)})")

    # Load model config
    with open(args.model_config, 'r') as f:
        config = json.load(f)

    # Build dataset & loader
    val_dataset = FusionDataset(args.val_csv, args.val_feats_dir, clinical_cols, bag_size=args.bag_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = ABMIL_Fusion(
        in_dim=config["in_dim"],
        clinical_in_dim=len(clinical_cols),
        n_classes=config["n_classes"],
        gate=config["gate"]
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'] if isinstance(torch.load(args.checkpoint), dict) else torch.load(args.checkpoint))
    model = model.to(args.device)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x_bag, x_clinical, labels in val_loader:
            x_bag = x_bag.to(args.device)
            x_clinical = x_clinical.to(args.device)
            labels = labels.to(args.device)

            out = model(x_bag, x_clinical)
            logits = out['logits']
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    # Compute metrics
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = None

    print(f"[RESULT] Validation Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"[RESULT] Validation ROC AUC: {auc:.4f}")

    # Save predictions
    results_df = pd.DataFrame({
        "label": all_labels,
        "pred": preds,
        "prob": all_probs
    })
    pred_path = os.path.join(os.path.dirname(args.checkpoint), "val_predictions.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"[INFO] Predictions saved to {pred_path}")
