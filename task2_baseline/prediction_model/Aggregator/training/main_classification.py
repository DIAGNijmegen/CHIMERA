import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .trainer import train as trainer_train


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
        return {
            'img': x_bag,
            'clinical': x_clinical,
            'label': label
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--train_feats_dir', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--val_feats_dir', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--bag_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--model_type', type=str, default='ABMIL_Fusion')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Optional model/training params
    parser.add_argument('--dropout_p', type=float, default=0.5, help="Dropout probability for ABMIL_Fusion")
    parser.add_argument('--gate', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use gating for clinical features")
    parser.add_argument('--in_dropout', type=float, default=0.0)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['constant', 'cosine', 'linear'],
                        help='Learning rate scheduler type')
    parser.add_argument('--es_min_epochs', type=int, default=5)
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--opt', type=str, default='adamW',
                        choices=['adamW', 'sgd', 'RAdam'],
                        help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps for LR scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs for LR scheduler')

    args = parser.parse_args()

    # Determine clinical feature columns automatically
    df_train = pd.read_csv(args.train_csv)
    exclude_cols = ['slide_id', 'sample_id', 'SampleID', 'label']
    clinical_cols = [c for c in df_train.columns if c not in exclude_cols]
    print(f"[INFO] Using clinical features: {clinical_cols} ({len(clinical_cols)})")

    # Print training and validation class distributions
    print("[INFO] Training class distribution:",
          pd.Series([1 if l == 'BRS3' else 0 for l in df_train['label']]).value_counts().to_dict())

    df_val = pd.read_csv(args.val_csv)
    print("[INFO] Validation class distribution:",
          pd.Series([1 if l == 'BRS3' else 0 for l in df_val['label']]).value_counts().to_dict())

    # Build datasets
    train_dataset = FusionDataset(args.train_csv, args.train_feats_dir, clinical_cols, bag_size=args.bag_size)
    val_dataset = FusionDataset(args.val_csv, args.val_feats_dir, clinical_cols, bag_size=args.bag_size)

    # WeightedRandomSampler for balanced batches
    train_labels = [1 if lbl == "BRS3" else 0 for lbl in df_train["label"]]
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    class_weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Prepare datasets dict for trainer
    datasets = {
        'train': train_loader,
        'val': val_loader
    }

    # Load model config
    with open(args.model_config, 'r') as f:
        config = json.load(f)

    # Add these args for trainer & model creation
    args.in_dim = config["in_dim"]
    args.clinical_in_dim = len(clinical_cols)
    args.n_classes = 2
    args.gate = config.get("gate", args.gate)  # allow override from config or CLI

    # Run training
    trainer_train(datasets, args, mode='classification')
