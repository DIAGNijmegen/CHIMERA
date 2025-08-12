from __future__ import print_function, division
import os
from os.path import join as j_
import torch
import numpy as np
import pandas as pd
import math
import sys

from torch.utils.data import Dataset, DataLoader
import h5py
from .dataset_utils import apply_sampling
sys.path.append('../')
from utils.pandas_helper_funcs import df_sdir, series_diff

class WSISurvivalDataset(Dataset):
    def __init__(self,
                 df,
                 data_source,
                 target_transform=None,
                 sample_col='sample_id',
                 survival_time_col='Time_to_prog_or_FUend',
                 censorship_col='progression',
                 n_label_bins=4,
                 label_bins=None,
                 bag_size=0,
                 rna_dim=None,
                 include_surv_t0=True,
                 **kwargs):
        self.data_source = [data_source] if isinstance(data_source, str) else data_source
        self.use_h5 = any(os.path.basename(src) == 'feats_h5' for src in self.data_source)
        self.rna_dim = rna_dim
        self.data_df = df.dropna(subset=[censorship_col, survival_time_col])
        self.sample_col = sample_col
        self.target_col = survival_time_col
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.include_surv_t0 = include_surv_t0

        if (self.data_df[self.survival_time_col] < 0).sum() > 0 and not self.include_surv_t0:
            self.data_df = self.data_df[self.data_df[self.survival_time_col] > 0]

        self.target_transform = target_transform
        self.n_label_bins = n_label_bins
        self.label_bins = label_bins
        self.bag_size = bag_size

        self.validate_survival_dataset()
        self.idx2sample_df = pd.DataFrame({self.sample_col: self.data_df[sample_col].astype(str).unique()})
        self.set_feat_paths_in_df()
        self.data_df.index = self.data_df[sample_col].astype(str)
        self.data_df.index.name = 'sample_id'

        if self.n_label_bins > 0:
            disc_labels, self.label_bins = compute_discretization(
                df=self.data_df,
                survival_time_col=self.survival_time_col,
                censorship_col=self.censorship_col,
                n_label_bins=self.n_label_bins,
                label_bins=self.label_bins
            )
            self.data_df = self.data_df.join(disc_labels)
            self.target_col = disc_labels.name

    def __len__(self):
        return len(self.idx2sample_df)

    def set_feat_paths_in_df(self):
        self.feats_df = pd.concat([df_sdir(feats_dir, cols=['fpath', 'fname', self.sample_col]) 
                                   for feats_dir in self.data_source]).drop(['fname'], axis=1).reset_index(drop=True)

        self.data_df = self.data_df.merge(self.feats_df, how='left', on=self.sample_col, validate='1:1')

    def validate_survival_dataset(self):
        assert self.data_df.groupby(self.sample_col)[self.survival_time_col].nunique().eq(1).all(), \
            'Each sample_id must have one unique survival time.'
        assert pd.to_numeric(self.data_df[self.survival_time_col], errors='coerce').notna().all(), \
            'Survival times must be numeric.'
        assert (self.data_df[self.survival_time_col] >= 0).all(), \
            'Survival times must be non-negative.'
        assert self.data_df[self.censorship_col].isin([0, 1]).all(), \
            'Censorship must be binary.'

    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx][self.sample_col]

    def get_feat_paths(self, idx):
        paths = self.data_df.loc[self.get_sample_id(idx), 'fpath']
        return [paths] if isinstance(paths, str) else paths

    def get_labels(self, idx):
        labels = self.data_df.loc[self.get_sample_id(idx),
                                  [self.survival_time_col, self.censorship_col, self.target_col]]
        return list(labels) if isinstance(labels, pd.Series) else list(labels.iloc[0])

    def __getitem__(self, idx):
        sample_id = self.get_sample_id(idx)
        survival_time, censorship, label = self.get_labels(idx)
        features_list = []

        for path in self.get_feat_paths(idx):
            features = torch.load(path, weights_only=True).numpy()
            if len(features.shape) > 2:
                features = np.squeeze(features, axis=0)
            features_list.append(features)

        all_features = torch.from_numpy(np.concatenate(features_list, axis=0))
        all_features, _, attn_mask = apply_sampling(self.bag_size, all_features, None)

        all_tabular = self.data_df.loc[sample_id]
        exclude_cols = ['sample_id', 'progression', 'Time_to_prog_or_FUend', 'fpath', 'disc_label']
        feature_cols = [col for col in self.data_df.columns if col not in exclude_cols]

        rna_cols = feature_cols[:self.rna_dim]
        clinical_cols = feature_cols[self.rna_dim:]

        rna_tensor = torch.tensor(all_tabular[rna_cols].values.astype(np.float32))
        clinical_tensor = torch.tensor(all_tabular[clinical_cols].values.astype(np.float32))

        return {
            'img': all_features,
            'survival_time': torch.tensor([survival_time], dtype=torch.float32),
            'censorship': torch.tensor([censorship], dtype=torch.float32),
            'label': torch.tensor([label], dtype=torch.float32),
            'rna': rna_tensor,
            'clinical': clinical_tensor,
            'time': torch.tensor([survival_time], dtype=torch.float32),
            'event': torch.tensor([censorship], dtype=torch.float32),
            'attn_mask': attn_mask
        }

def compute_discretization(df,
                           survival_time_col='Time_to_prog_or_FUend',
                           censorship_col='progression',
                           n_label_bins=4,
                           label_bins=None):
    df = df[~df['sample_id'].duplicated()]
    if label_bins is not None:
        q_bins = label_bins
    else:
        uncensored_df = df[df[censorship_col] == 0]
        _, q_bins = pd.qcut(uncensored_df[survival_time_col], q=n_label_bins, retbins=True, labels=False)
        q_bins[-1] = 1e6
        q_bins[0] = -1e-6

    disc_labels, q_bins = pd.cut(df[survival_time_col], bins=q_bins, retbins=True, labels=False, include_lowest=True)
    disc_labels.name = 'disc_label'
    return disc_labels, q_bins

def get_split_loader(args, train_df, test_df):
    train_dataset = WSISurvivalDataset(
        df=train_df,
        data_source=args.data_source,
        sample_col=args.sample_col,
        survival_time_col=args.target_col,
        censorship_col=args.event_col,
        bag_size=args.bag_size,
        rna_dim=args.rna_dim
    )

    test_dataset = WSISurvivalDataset(
        df=test_df,
        data_source=args.data_source,
        sample_col=args.sample_col,
        survival_time_col=args.target_col,
        censorship_col=args.event_col,
        bag_size=args.bag_size,
        rna_dim=args.rna_dim
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return {'train': train_loader, 'val': test_loader}
