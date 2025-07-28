"""
Main entry point for survival downstream tasks
"""

from __future__ import print_function

import argparse
import pdb
import os
from os.path import join as j_
import sys
from pathlib import Path

# --- Add Project Root to Python Path ---
# (Assumes this script is in a subdirectory of the project)
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

# internal imports
from task1_baseline.prediction_model.Aggregators.utils.file_utils import save_pkl
from task1_baseline.prediction_model.Aggregators.utils.utils import (seed_torch, array2list, merge_dict, read_splits, 
                         parse_model_name, get_current_time, extract_patching_info)

from task1_baseline.prediction_model.Aggregators.training.trainer import train
from task1_baseline.prediction_model.Aggregators.wsi_datasets import WSISurvivalDataset
# pytorch imports
import torch
from torch.utils.data import DataLoader

import pandas as pd
import json


def build_datasets(csv_splits, model_type, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    
    # Initialize clinical data processor if path is provided
    clinical_processor = None
    if 'clinical_data_path' in train_kwargs and train_kwargs['clinical_data_path'] is not None:
        from wsi_datasets.clinical_processor import ClinicalDataProcessor
        print(f"Initializing clinical data processor with path: {train_kwargs['clinical_data_path']}")
        clinical_processor = ClinicalDataProcessor(clinical_data_path=train_kwargs['clinical_data_path'])
        
        # Fit the processor on the training set case IDs
        if 'train' in csv_splits:
            print("Fitting clinical data processor on training set...")
            train_case_ids = csv_splits['train']['case_id'].unique().tolist()
            clinical_processor.fit(train_case_ids)
            
            # Update clinical dimension in all dataset kwargs
            train_kwargs['clinical_processor'] = clinical_processor
            val_kwargs['clinical_processor'] = clinical_processor
    
    if clinical_processor is not None:
        # Add clinical processor to args for model initialization
        args.clinical_processor = clinical_processor
        
        # Set the clinical dimension in args based on processor output dimension
        args.clinical_dim = clinical_processor.output_dim
        print(f"Clinical processor initialized with output dimension: {args.clinical_dim}")
    
    for k in csv_splits.keys(): # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSISurvivalDataset(df, **dataset_kwargs)

        # If prototype methods, each WSI will have same feature bag dimension and is batchable
        # Otherwise, we need to use batch size of 1 to accommodate to different bag size for each WSI.
        # Alternatively, we can sample same number of patch features per WSI to have larger batch.
  
        batch_size = batch_size if dataset_kwargs.get('bag_size', -1) > 0 else 1

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=dataset_kwargs['shuffle'], num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()
    return dataset_splits


def main(args):
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    censorship_col = args.target_col.split('_')[0] + '_censorship'

    train_kwargs = dict(data_source=args.data_source,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=args.train_bag_size,
                        shuffle=True,
                        mri_feature_path=args.mri_feature_path,
                        clinical_data_path=args.clinical_data_path,
                        )

    # use the whole bag at test time
    val_kwargs = dict(data_source=args.data_source,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=args.val_bag_size,
                      shuffle=False,
                      mri_feature_path=args.mri_feature_path,
                      clinical_data_path=args.clinical_data_path,
                      )

    all_results, all_dumps = {}, {}

    # Cross-validation
    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('successfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits, 
                                    model_type=args.model_type,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    fold_results, fold_dumps = train(dataset_splits, args, mode='survival')

    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)
    
    # Also save a CSV version for easy viewing
    try:
        csv_dump_data = []
        for split, dumps in fold_dumps.items():
            for key, values in dumps.items():
                if isinstance(values, (list, tuple)):
                    for i, val in enumerate(values):
                        csv_dump_data.append({
                            'split': split,
                            'metric': key,
                            'index': i,
                            'value': val
                        })
                else:
                    csv_dump_data.append({
                        'split': split,
                        'metric': key,
                        'index': 0,
                        'value': values
                    })
        
        if csv_dump_data:
            csv_dump_df = pd.DataFrame(csv_dump_data)
            csv_dump_df.to_csv(j_(args.results_dir, 'all_dumps.csv'), index=False)
            print(f"Saved detailed results to: {j_(args.results_dir, 'all_dumps.csv')}")
    except Exception as e:
        print(f"Warning: Could not save CSV dump: {e}")

    return final_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=10,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=4,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=1)

### misc ###
parser.add_argument('--print_every', default=1,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=2)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=0, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=3,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=5,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

### model args ###
parser.add_argument('--model_type', type=str, choices=['ABMIL'], default='ABMIL',
                    help='type of model')
parser.add_argument('--emb_model_type', type=str, default='LinEmb_LR')
parser.add_argument('--ot_eps', default=0.1, type=float,
                    help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_config', type=str,
                    default='ABMIL_default', help="name of model config file")
parser.add_argument('--n_fc_layers', type=int)
parser.add_argument('--em_iter', type=int)
parser.add_argument('--tau', type=float)
parser.add_argument('--out_type', type=str, default='param_cat')


parser.add_argument('--in_dim', default=1024, type=int,
                    help='dim of input features')
parser.add_argument('--in_dropout', default=0.1, type=float,
                    help='Probability of dropping out input features.')
parser.add_argument('--mri_feature_path', type=str, default=None,
                    help='Path to directory containing pre-created MRI features (.pt files)')
parser.add_argument('--clinical_data_path', type=str, default=None,
                    help='Path to directory containing clinical data files (.json files)')
parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)
parser.add_argument('--loss_fn', type=str, default='cox', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
parser.add_argument('--nll_alpha', type=float, default=0,
                    help='Balance between censored / uncensored loss')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default='MM_bcr_survival_task2')
parser.add_argument('--target_col', type=str, default='bcr_survival_months')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print('task: ', args.task)
    args.split_dir = j_('splits', args.split_dir)
    print('split_dir: ', args.split_dir)
    split_num = args.split_dir.split('/')[-1].split('_')
    args.split_name_clean = split_num[0]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0


    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        print('data source: ', src)
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ###
        feat_name = os.path.basename(src)
        try:
            patching_info = extract_patching_info(os.path.dirname(src))
            if patching_info is None or len(patching_info) != 2:
                raise ValueError("Invalid patching info")
            mag, patch_size = patching_info
            if (mag < 0 or patch_size < 0):
                raise ValueError("Negative patching values")
        except:
            # Set default values if parsing fails
            print(f"Warning: Could not parse patching info from {src}, using defaults")
            mag, patch_size = 20, 256  # Default magnification and patch size
        check_params_same.append([feat_name, mag, patch_size])

        #### parse model name ####
        parsed = parse_model_name(feat_name) 
        parsed.update({'patch_mag': mag, 'patch_size': patch_size})
    
    try:
        check_params_same = pd.DataFrame(check_params_same, columns=['feats_name', 'mag', 'patch_size'])
        assert check_params_same.drop(['feats_name'],axis=1).drop_duplicates().shape[0] == 1
        print("All data sources have the same feature extraction parameters.")
    except:
        print("Data sources do not share the same feature extraction parameters. Exiting...")
        sys.exit()
        
    ### Updated parsed mdoel parameters in args.Namespace ###
    for key, val in parsed.items():
        setattr(args, key, val)
    
    ### setup results dir ###
    if args.exp_code is None:
        if args.model_config == 'PANTHER_default':
            exp_code = f"{args.split_name_clean}::{args.model_config}+{args.emb_model_type}::{args.loss_fn}::{feat_name}"
        else:
            exp_code = f"{args.split_name_clean}::{args.model_config}::{feat_name}"
    else:
        pass
    
    args.results_dir = j_(args.results_dir, 
                          args.task, 
                          f'k={args.split_k}', 
                          str(exp_code), 
                          str(exp_code)+f"::{get_current_time()}")

    os.makedirs(args.results_dir, exist_ok=True)

    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results = main(args)

    print("FINISHED!\n\n\n")