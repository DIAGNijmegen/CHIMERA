import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from mil_models.model_factory import create_downstream_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import collections

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
        slide_id = row['sample_id']
        label = 1 if row['label'] == 'BRS3' else 0

        feat_path = os.path.join(self.feats_dir, f"{slide_id}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing feature: {feat_path}")
        x_bag = torch.load(feat_path, map_location='cpu', weights_only=False).float()

        # Adjust bag size if needed
        if self.bag_size:
            if x_bag.size(0) > self.bag_size:
                x_bag = x_bag[:self.bag_size]
            elif x_bag.size(0) < self.bag_size:
                pad = self.bag_size - x_bag.size(0)
                x_bag = torch.cat([x_bag, torch.zeros(pad, x_bag.size(1))], dim=0)

        x_clinical = torch.tensor(row[self.clinical_cols].values.astype("float32"))
        return {'img': x_bag, 'clinical': x_clinical, 'label': label}


# === Paths ===
val_csv = "/data/temporary/maryammohamm/FusionModelTask2/clinical_val_encoded.csv"
val_feats_dir = "/data/temporary/chimera/bladder/features_validation_task2_uni_maryam/features_validation_task2_uni_maryam/features/"
checkpoint_path = "/data/temporary/maryammohamm/FusionModelTask2/result_TrainValidation/train_val_run/s_checkpoint.pth"

# === Clinical feature columns ===
df_val = pd.read_csv(val_csv)
exclude_cols = ['slide_id', 'sample_id', 'SampleID', 'label']
clinical_cols = [c for c in df_val.columns if c not in exclude_cols]

# === Dataset & Loader ===
val_dataset = FusionDataset(val_csv, val_feats_dir, clinical_cols)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# === Load Model ===
class Args: pass
args = Args()
args.model_type = 'ABMIL_Fusion'
args.in_dim = 1024
args.clinical_in_dim = len(clinical_cols)
args.n_classes = 2
args.gate = True

model = create_downstream_model(args)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

# === Run Predictions ===
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        img, clinical, label = batch['img'], batch['clinical'], batch['label'].item()
        out = model(img, clinical)
        pred = torch.argmax(out['logits'], dim=1).item()
        all_preds.append(pred)
        all_labels.append(label)

# === Results ===
print("Prediction counts:", collections.Counter(all_preds))
print("Label counts:", collections.Counter(all_labels))
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Balanced Accuracy:", balanced_accuracy_score(all_labels, all_preds))
print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))
