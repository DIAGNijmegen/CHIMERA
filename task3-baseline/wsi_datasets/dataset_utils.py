import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# === Apply sampling function ===
def apply_sampling(target_bag_size, all_features, all_coords):
    attn_mask = None
    if target_bag_size > 0:
        bag_size = all_features.size(0)
        attn_mask = torch.ones(target_bag_size)

        if bag_size < target_bag_size:
            pad_size = target_bag_size - bag_size
            pad_feats = torch.zeros((pad_size, all_features.shape[1]), dtype=all_features.dtype)
            sampled_features = torch.cat([all_features, pad_feats], dim=0)

            if isinstance(all_coords, np.ndarray) and all_coords.size > 0:
                pad_coords = np.zeros((pad_size, all_coords.shape[1]))
                all_coords = np.concatenate([all_coords, pad_coords], axis=0)
        else:
            sampled_patch_ids = np.random.choice(bag_size, target_bag_size, replace=False)
            sampled_features = all_features[sampled_patch_ids, :]
            attn_mask = torch.ones(target_bag_size)

            if isinstance(all_coords, np.ndarray) and all_coords.size > 0:
                all_coords = all_coords[sampled_patch_ids, :]

        all_features = sampled_features

    return all_features, all_coords, attn_mask


# === Dataset class for survival fusion ===
class SurvivalFusionDataset(Dataset):
    def __init__(self, df, feature_dir, target_col, event_col, bag_size):
        self.df = df
        self.feature_dir = feature_dir
        self.target_col = target_col
        self.event_col = event_col
        self.bag_size = bag_size

        # RNA/clinical features = everything except ID, label, event
        self.rna_cols = [c for c in df.columns if c not in ['sample_id', target_col, event_col]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['sample_id']

        # Load pathology features (.pt file)
        feat_path = os.path.join(self.feature_dir, f"{sample_id}.pt")
        img_feat = torch.load(feat_path).float()  # [N, D]

        # Apply sampling to pathology features
        img_feat, _, _ = apply_sampling(self.bag_size, img_feat, [])

        # Get RNA/clinical features
        rna_feat = torch.tensor(row[self.rna_cols].values.astype(np.float32))

        return {
            'img': img_feat,                  # [bag_size, D]
            'clinical': rna_feat,             # [R]
            'time': torch.tensor(row[self.target_col], dtype=torch.float32),
            'event': torch.tensor(row[self.event_col], dtype=torch.float32),
            'sample_id': sample_id
        }


# === Loader function ===
def get_split_loader(args):
    split_dir = args.split_path
    print(f"[INFO] Loading split from: {split_dir}")

    train_csv = os.path.join(split_dir, "train.csv")
    test_csv  = os.path.join(split_dir, "test.csv")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    train_dataset = SurvivalFusionDataset(train_df, args.data_source, args.target_col, args.event_col, args.bag_size)
    val_dataset   = SurvivalFusionDataset(test_df,  args.data_source, args.target_col, args.event_col, args.bag_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    return {'train': train_loader, 'val': val_loader}
