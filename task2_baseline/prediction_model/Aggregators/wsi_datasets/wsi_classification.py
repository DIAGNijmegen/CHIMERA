from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from .dataset_utils import apply_sampling
from utils.pandas_helper_funcs import df_sdir, series_diff

class WSIClassificationDataset(Dataset):
    """WSI + Clinical dataset for ABMIL_Fusion (no cross-validation assumptions)."""

    def __init__(self,
                 df: pd.DataFrame,
                 data_source: list,
                 sample_col='slide_id',
                 slide_col='slide_id',
                 target_col='label',
                 label_map=None,
                 bag_size=0,
                 clinical_cols=None):
        self.data_source = list(data_source)
        self.sample_col = sample_col
        self.slide_col = slide_col
        self.target_col = target_col
        self.label_map = label_map
        self.bag_size = bag_size

        if clinical_cols is None:
            self.clinical_cols = [
                'age', 'sex', 'smoking', 'tumor', 'stage', 'substage',
                'grade', 'reTUR', 'LVI', 'variant', 'EORTC', 'no_instillations'
            ]
        else:
            self.clinical_cols = clinical_cols

        # Ensure clean DataFrame
        self.data_df = df.copy()
        if 'Unnamed: 0' in self.data_df.columns:
            self.data_df = self.data_df.drop(columns=['Unnamed: 0'])
        self.data_df[self.sample_col] = self.data_df[self.sample_col].astype(str)
        self.data_df[self.slide_col] = self.data_df[self.slide_col].astype(str)

        # Validate labels and attach feature paths
        self._validate_labels()
        self._set_feat_paths()

        # Create list of unique samples
        self.idx2sample_df = pd.DataFrame({'sample_id': self.data_df[self.sample_col].unique()})
        self.data_df.index = self.data_df[self.sample_col]

        # Preload labels
        self.labels = torch.tensor([self._get_label(i) for i in self.idx2sample_df.index], dtype=torch.long)

        print(self.data_df.groupby([self.target_col])[self.sample_col].count().to_string())

    def _validate_labels(self):
        counts = self.data_df.groupby(self.sample_col)[self.target_col].nunique()
        if not (counts == 1).all():
            raise ValueError("Each sample must have exactly one label value.")

    def _set_feat_paths(self):
        feats_df = pd.concat(
            [df_sdir(feats_dir, cols=['fpath', 'fname', self.slide_col]) for feats_dir in self.data_source],
            ignore_index=True
        ).drop(['fname'], axis=1)

        missing = series_diff(self.data_df[self.slide_col], feats_df[self.slide_col])
        if len(missing) > 0:
            raise FileNotFoundError(f"Missing features for slides: {missing.tolist()}")

        self.data_df = self.data_df.merge(feats_df, how='left', on=self.slide_col, validate='1:1')

    def _get_label(self, idx):
        sample_id = self.idx2sample_df.loc[idx, 'sample_id']
        label = self.data_df.loc[sample_id, self.target_col]
        if isinstance(label, pd.Series):
            label = label.iloc[0]
        if self.label_map:
            label = self.label_map[label]
        return label

    def _get_feat_paths(self, idx):
        sample_id = self.idx2sample_df.loc[idx, 'sample_id']
        fpath = self.data_df.loc[sample_id, 'fpath']
        return [fpath] if isinstance(fpath, str) else fpath.tolist()

    def __len__(self):
        return len(self.idx2sample_df)

    def __getitem__(self, idx):
        feat_paths = self._get_feat_paths(idx)
        all_features, all_coords = [], []

        for fp in feat_paths:
            if fp.endswith('.h5'):
                with h5py.File(fp, 'r') as f:
                    feats = f['features'][:]
                    coords = f.get('coords', np.zeros((feats.shape[0], 2)))
            else:
                feats = torch.load(fp, map_location='cpu')
                coords = np.zeros((feats.shape[0], 2))

            if len(feats.shape) > 2:
                assert feats.shape[0] == 1
                feats = np.squeeze(feats, axis=0)

            all_features.append(feats)
            all_coords.append(coords)

        all_features = torch.from_numpy(np.concatenate(all_features, axis=0)).float()
        all_coords = np.concatenate(all_coords, axis=0)

        # Apply bag sampling
        all_features, all_coords, attn_mask = apply_sampling(self.bag_size, all_features, all_coords)

        # Clinical features
        sample_id = self.idx2sample_df.loc[idx, 'sample_id']
        clinical_row = self.data_df.loc[sample_id, self.clinical_cols]
        if isinstance(clinical_row, pd.Series):
            clinical_feats = clinical_row.values
        else:
            clinical_feats = clinical_row.iloc[0].values
        clinical_feats = torch.tensor(clinical_feats, dtype=torch.float32)

        out = {
            'img': all_features,
            'coords': all_coords,
            'label': self.labels[idx],
            'clinical': clinical_feats
        }
        if attn_mask is not None:
            out['attn_mask'] = attn_mask
        return out
