# main_survival_sweep.py
import argparse
from argparse import Namespace
import yaml
import wandb
import os
from os.path import join as j_
from pathlib import Path
import sys
import torch
import pandas as pd
import json

# --- Add Project Root to Python Path ---
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Internal Imports ---
from trainer import train 
from task1_baseline.prediction_model.Aggregators.wsi_datasets import WSISurvivalDataset
from task1_baseline.prediction_model.Aggregators.wsi_datasets.clinical_processor import ClinicalDataProcessor
from task1_baseline.prediction_model.Aggregators.utils.utils import (seed_torch, read_splits, extract_patching_info, parse_model_name, merge_dict, array2list)
from task1_baseline.prediction_model.Aggregators.utils.file_utils import save_pkl
from torch.utils.data import DataLoader

def build_datasets(csv_splits, args, model_type, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    dataset_splits = {}
    label_bins = None
    clinical_processor = None
    if 'clinical_data_path' in train_kwargs and train_kwargs['clinical_data_path'] is not None:
        print(f"Initializing clinical data processor with path: {train_kwargs['clinical_data_path']}")
        clinical_processor = ClinicalDataProcessor(clinical_data_path=train_kwargs['clinical_data_path'])
        if 'train' in csv_splits:
            print("Fitting clinical data processor on training set...")
            train_case_ids = csv_splits['train']['case_id'].unique().tolist()
            clinical_processor.fit(train_case_ids)
            train_kwargs['clinical_processor'] = clinical_processor
            val_kwargs['clinical_processor'] = clinical_processor
    if clinical_processor is not None:
        args.clinical_processor = clinical_processor
        args.clinical_dim = clinical_processor.output_dim
        print(f"Clinical processor initialized. Model clinical_dim set to: {args.clinical_dim}")
    for k in csv_splits.keys():
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if k == 'train' else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSISurvivalDataset(df, **dataset_kwargs)
        current_batch_size = batch_size if dataset_kwargs.get('bag_size', -1) > 0 else 1
        dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=dataset_kwargs['shuffle'], num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()
    return dataset_splits

# === THIS FUNCTION HAS BEEN MODIFIED ===
def run_training_trial(fixed_args: Namespace):
    with wandb.init() as run:
        # wandb.config is populated with the hyperparameters for this specific run
        config = wandb.config

        # 1. Create a descriptive name from the hyperparameters
        run_name = (
            f"lr-{config.lr:.2e}_wd-{config.wd:.2e}_"
            f"opt-{config.opt}_drop-{config.in_dropout:.2f}"
        )
        
        # 2. Update the wandb run with the new, descriptive name
        run.name = run_name

        # Create the unified args object for this run
        args = Namespace(**vars(fixed_args))
        for key, value in config.items():
            if isinstance(value, str) and ('/' in value or '\\' in value) and os.path.exists(value):
                 setattr(args, key, Path(value))
            else:
                 setattr(args, key, value)
        
        # 3. Use the new descriptive name for the results folder
        args.results_dir = Path(j_(args.results_dir, run_name))
        args.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config.json for this run
        args_dict = vars(args)
        for key, value in args_dict.items():
            if isinstance(value, Path):
                args_dict[key] = str(value)
        with open(j_(args.results_dir, 'config.json'), 'w') as f:
            json.dump(args_dict, f, sort_keys=True, indent=4)

        print("\n" + "="*40 + f"\n🚀 Starting W&B Run: {run.name}\n" + "="*40)
        print("Applied Configuration for this run:")
        for key, val in vars(args).items(): print(f"  {key}: {val}")
        print("-"*40)

        seed_torch(args.seed)

        if args.train_bag_size == -1: args.train_bag_size = args.bag_size
        if args.val_bag_size == -1: args.val_bag_size = args.bag_size
        if args.loss_fn != 'nll': args.n_label_bins = 0

        censorship_col = args.target_col.split('_')[0] + '_censorship'
        
        common_kwargs = {'data_source': args.data_source, 'survival_time_col': args.target_col, 'censorship_col': censorship_col, 'n_label_bins': args.n_label_bins, 'mri_feature_path': args.mri_feature_path, 'clinical_data_path': args.clinical_data_path}
        train_kwargs = {**common_kwargs, 'shuffle': True, 'bag_size': args.train_bag_size}
        val_kwargs = {**common_kwargs, 'shuffle': False, 'bag_size': args.val_bag_size}

        csv_splits = read_splits(args)
        dataset_splits = build_datasets(csv_splits, args, model_type=args.model_type, batch_size=args.batch_size, num_workers=args.num_workers, train_kwargs=train_kwargs, val_kwargs=val_kwargs)
        
        fold_results, fold_dumps = train(dataset_splits, args, mode='survival')

        # Save all results files
        all_results = {}
        for split, split_results in fold_results.items():
            all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
            save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split])
        
        final_dict = {}
        for split, split_results in all_results.items():
            final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
        
        final_df = pd.DataFrame(final_dict)
        final_df.to_csv(j_(args.results_dir, 'summary.csv'), index=False)
        with open(j_(args.results_dir, 'summary.csv.json'), 'w') as f:
            f.write(json.dumps(final_dict, sort_keys=True, indent=4))
        
        save_pkl(j_(args.results_dir, 'all_dumps.h5'), fold_dumps)
        try:
            csv_dump_data = []
            for split, dumps in fold_dumps.items():
                for key, values in dumps.items():
                    if isinstance(values, (list, tuple, pd.Series, pd.DataFrame)):
                        for i, val in enumerate(values): csv_dump_data.append({'split': split, 'metric': key, 'index': i, 'value': val})
                    else:
                        csv_dump_data.append({'split': split, 'metric': key, 'index': 0, 'value': values})
            if csv_dump_data:
                pd.DataFrame(csv_dump_data).to_csv(j_(args.results_dir, 'all_dumps.csv'), index=False)
                print(f"Saved detailed results to: {j_(args.results_dir, 'all_dumps.csv')}")
        except Exception as e:
            print(f"Warning: Could not save CSV dump: {e}")

        print(f"✅ Finished W&B Run: {run.name}")

# The main() function remains unchanged
def main():
    parser = argparse.ArgumentParser(description="Run a Weights & Biases sweep for a survival model.")
    # Required Paths
    parser.add_argument('--split_dir', type=Path, required=True)
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--mri_feature_path', type=Path, required=True)
    parser.add_argument('--clinical_data_path', type=Path, required=True)
    # Logging & Sweep Controls
    parser.add_argument('--results_dir', type=Path, default=Path('./results'))
    parser.add_argument('--sweep_yaml', type=Path, default=Path("sweep.yaml"))
    parser.add_argument('--project_name', type=str, default="survival-model-sweep")
    parser.add_argument('--run_count', type=int, default=10)
    # Optimizer Settings
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--opt', type=str, default='adamW', choices=['adamW', 'sgd', 'RAdam'])
    parser.add_argument('--lr_scheduler', type=str, choices=['cosine', 'linear', 'constant'], default='constant')
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--warmup_epochs', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=1)
    # Misc
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    # Early Stopper
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--es_min_epochs', type=int, default=3)
    parser.add_argument('--es_patience', type=int, default=5)
    parser.add_argument('--es_metric', type=str, default='loss')
    # Model Args
    parser.add_argument('--model_type', type=str, choices=['ABMIL'], default='ABMIL')
    parser.add_argument('--emb_model_type', type=str, default='LinEmb_LR')
    parser.add_argument('--ot_eps', default=0.1, type=float)
    parser.add_argument('--model_config', type=str, default='ABMIL_default')
    parser.add_argument('--n_fc_layers', type=int)
    parser.add_argument('--em_iter', type=int)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--out_type', type=str, default='param_cat')
    # Feature/Data Args
    parser.add_argument('--in_dim', default=1024, type=int)
    parser.add_argument('--in_dropout', default=0.1, type=float)
    parser.add_argument('--bag_size', type=int, default=-1)
    parser.add_argument('--train_bag_size', type=int, default=-1)
    parser.add_argument('--val_bag_size', type=int, default=-1)
    # Loss/Label Args
    parser.add_argument('--loss_fn', type=str, default='cox', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'])
    parser.add_argument('--nll_alpha', type=float, default=0)
    # Experiment Args
    parser.add_argument('--exp_code', type=str, default=None)
    parser.add_argument('--task', type=str, default='MM_bcr_survival_task2')
    parser.add_argument('--target_col', type=str, default='bcr_survival_months')
    parser.add_argument('--n_label_bins', type=int, default=4)
    # Dataset/Split Args
    parser.add_argument('--split_names', type=str, default='train,val,test')
    parser.add_argument('--overwrite', action='store_true', default=False)
    # Logging Args
    parser.add_argument('--tags', nargs='+', type=str, default=None)
    
    args = parser.parse_args()
    
    args.split_dir = j_('splits', args.split_dir) if 'splits' not in str(args.split_dir) else args.split_dir
    split_num = os.path.basename(str(args.split_dir)).split('_'); args.split_name_clean = split_num[0]
    args.split_k = int(split_num[1]) if len(split_num) > 1 else 0
    args.data_source = [src.strip() for src in args.data_source.split(',')]
    for src in args.data_source:
        feat_name = os.path.basename(src)
        try:
            mag, patch_size = extract_patching_info(os.path.dirname(src))
        except:
            print(f"Warning: Could not parse patching_info from {src}, using defaults"); mag, patch_size = 20, 256
        parsed = parse_model_name(feat_name); parsed.update({'patch_mag': mag, 'patch_size': patch_size})
        for key, val in parsed.items(): setattr(args, key, val)

    with open(args.sweep_yaml, 'r') as f:
        sweep_config = yaml.safe_load(f)

    if 'parameters' not in sweep_config: sweep_config['parameters'] = {}
    for key, value in vars(args).items():
        if key not in sweep_config['parameters']:
            sweep_config['parameters'][key] = {'value': str(value) if isinstance(value, Path) else value}
    
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)
    print(f"Starting W&B agent to run {args.run_count} trials... (Sweep ID: {sweep_id})")
    wandb.agent(sweep_id, function=lambda: run_training_trial(args), count=args.run_count)
    print("Sweep finished.")

if __name__ == '__main__':
    main()