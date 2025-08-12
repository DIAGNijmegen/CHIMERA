import argparse
import sys, os
import torch
import numpy as np
import pandas as pd
import random

from wsi_datasets.wsi_survival import get_split_loader
from .model_utils import create_downstream_model
from .trainer import train
from utils.my_utils import j_, save_pkl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)
    print(f"\n=== RUNNING SURVIVAL EXPERIMENT | CONFIG {args.exp_code} ===")
    os.makedirs(args.results_dir, exist_ok=True)

    # === Load train/test splits
    train_csv = os.path.join(args.split_path, "train.csv")
    test_csv = os.path.join(args.split_path, "test.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # === Drop survival labels from train/test if they exist (to avoid _x/_y merge issue)
    for col in [args.target_col, args.event_col]:
        if col in train_df.columns:
            train_df.drop(columns=col, inplace=True)
        if col in test_df.columns:
            test_df.drop(columns=col, inplace=True)

    # === Load clinical data (which contains labels + clinical features)
    clinical_df = pd.read_csv(args.clinical_path)
    clinical_df = clinical_df.rename(columns={'sample_id': args.sample_col})

    # === Merge clinical data into train/test
    train_df = train_df.merge(clinical_df, on=args.sample_col, how='left')
    test_df = test_df.merge(clinical_df, on=args.sample_col, how='left')

    print("? Merged train_df columns:", train_df.columns.tolist())
    print("? Merged test_df columns:", test_df.columns.tolist())

    # === Load and merge RNA data (if provided)
    if args.rna_path:
        rna_df = pd.read_csv(args.rna_path)
        rna_df = rna_df.rename(columns={'sample_id': args.sample_col})
        train_df = train_df.merge(rna_df, on=args.sample_col, how='left')
        test_df = test_df.merge(rna_df, on=args.sample_col, how='left')
        print(f"[INFO] Using clinical-based split. RNA dim = {rna_df.shape[1] - 1}")

    # === Check required survival columns exist
    for col in [args.target_col, args.event_col]:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"? Missing required column '{col}' in merged DataFrames.")

    # === Save merged files (optional but helpful for debugging)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # === Get dataloaders
    loaders = get_split_loader(args, train_df, test_df)

    # === Build model
    model = create_downstream_model(args, mode='survival')

    # === Train
    results, dumps = train(loaders, args, mode='survival')

    # === Save results
    save_pkl(results, j_(args.results_dir, "summary_results.pkl"))
    save_pkl(dumps, j_(args.results_dir, "raw_dumps.pkl"))
    print("? Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--clinical_path', type=str, required=True)
    parser.add_argument('--rna_path', type=str, default=None)

    # Column names
    parser.add_argument('--sample_col', type=str, default='sample_id')
    parser.add_argument('--target_col', type=str, default='Time_to_prog_or_FUend')
    parser.add_argument('--event_col', type=str, default='progression')

    # Model & optimization
    parser.add_argument('--model_type', type=str, default='ABMIL_Fusion_RNA_Clinical')
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--rna_dim', type=int, default=19359)
    parser.add_argument('--clinical_dim', type=int, default=13)
    parser.add_argument('--bag_size', type=int, default=2000)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--print_every', type=int, default=1, help='Print status every N epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--l1_alpha', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', default='none', type=str, help='learning rate scheduler: none | cosine | step')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs for cosine scheduler')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping if set')
  






    # Metadata
    parser.add_argument('--task', type=str, default='task3')
    parser.add_argument('--exp_code', type=str, required=True)

    args = parser.parse_args()
    main(args)
