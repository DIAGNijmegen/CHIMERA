import re
import math
import os
from os.path import join as j_
import pickle
import pandas as pd
import datetime
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import sampler
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from transformers import (get_constant_schedule_with_warmup, 
                         get_linear_schedule_with_warmup, 
                         get_cosine_schedule_with_warmup)






def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"

def extract_patching_info(s):
    match = re.search(r"extracted_mag(\d+)x_patch(\d+)_fp", s)
    mag, patch_size = -1, -1
    if match:
        mag = int(match.group(1))
        patch_size = int(match.group(2))
        return mag, patch_size


def parse_model_name(model_name, ckpt=None, inference_prec=None):
    """
    Parses a model name string like 'encoder.algo.exp[_fp16]' into its parts.
    If the format is invalid, returns defaults instead of raising an error.
    """
    # Set inference precision
    if inference_prec is None:
        inference_prec = 'fp32'
        if model_name.endswith('_fp16'):
            inference_prec = 'fp16'
            model_name = model_name[:-len('_fp16')]

    # Clean prefix
    model_name = model_name.replace('extracted-', '')

    # If invalid format, return default values
    if model_name.count('.') < 2:
        print(f"[WARNING] Model name '{model_name}' does not follow 'encoder.algo.exp' format. Using defaults.")
        return dict(
            pretrain_enc=model_name,
            pretrain_algo='default_algo',
            pretrain_exp='default_exp',
            pretrain_ckpt=-1,
            inference_prec=inference_prec
        )

    # Parse parts
    parsed = model_name.split('.', maxsplit=2)
    enc = parsed[0]
    algo = parsed[1]
    exp = parsed[2]

    # Parse checkpoint
    if ckpt is None:
        exp_parsed = exp.split('.')
        ckpt_str = exp_parsed[-1].split('_')[-1]
        if ckpt_str.isnumeric():
            ckpt = int(ckpt_str)
            # Remove numeric checkpoint suffix from exp string
            last_exp_part = exp_parsed[-1].split('_')[:-1]
            exp = '.'.join(exp_parsed[:-1] + ['_'.join(last_exp_part)])
        else:
            ckpt = -1
    else:
        ckpt = int(ckpt) if str(ckpt).isnumeric() else -1

    return dict(
        pretrain_enc=enc,
        pretrain_algo=algo,
        pretrain_exp=exp,
        pretrain_ckpt=ckpt,
        inference_prec=inference_prec
    )

def merge_dict(main_dict, new_dict):
    for k, v in new_dict.items():
        if k not in main_dict:
            main_dict[k] = []
        main_dict[k].append(v)
    return main_dict


def array2list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def summarize_reulsts(results_dict, ignore_keys = ['folds']):
    summary = {}
    for k, v in results_dict.items():
        if k in ignore_keys: continue
        summary[f"{k}_avg"] = np.mean(v)
        # summary[f"{k}_std"] = np.std(v)
    return summary


def seed_torch(seed=7):
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_splits(args, fold_idx=None):
    splits_csvs = {}
    split_names = args.split_names.split(',')
    logger.info(f"Looking for splits: {split_names}")

    for split in split_names:
        if fold_idx is not None:
            split_path = j_(args.split_dir, f'{split}_{fold_idx}.csv')
        else:
            split_path = j_(args.split_dir, f'{split}.csv')

        print(f"?? Checking file: {split_path}")

        if not os.path.isfile(split_path):
            print(f" Skipping missing split: {split} (not found at {split_path})")
            continue

        df = pd.read_csv(split_path)

        if df.empty:
            raise ValueError(f"[read_splits] Required column '{args.target_col}' not found in {split_path}. Found columns: {df.columns.tolist()}")


        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

      
        if args.target_col not in df.columns:
            raise ValueError(f"? Required column '{args.target_col}' not found in {split_path}. Columns: {df.columns.tolist()}")

        splits_csvs[split] = df
        print(f"? Loaded {split} with {len(df)} rows.")

    return splits_csvs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='unk', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_steps = args.warmup_steps
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs
    assert not (warmup_steps > 0 and warmup_epochs > 0), "Cannot have both warmup steps and epochs"
    accum_steps = args.accum_steps
    if warmup_steps > 0:
        warmup_steps = warmup_steps
    elif warmup_epochs > 0:
        warmup_steps = warmup_epochs * (len(dataloader) // accum_steps)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps)
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(dataloader) // accum_steps * epochs),
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(dataloader) // accum_steps) * epochs,
        )
    return lr_scheduler


def get_optim(args, model=None, parameters=None):
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    if parameters is None:
        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(
            n, p) and p.requires_grad]
        parameters = [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ]

    if args.opt == "adamW":
        optimizer = optim.AdamW(parameters, lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
    elif args.opt == 'RAdam':
        optimizer = optim.RAdam(parameters, lr=args.lr)
    else:
        raise NotImplementedError
    return optimizer
 

def print_network(net):
    num_params = 0
    num_params_train = 0

    logging.info(str(net))
    # print(str(net))
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    logging.info(f'Total number of parameters: {num_params}')
    logging.info(f'Total number of trainable parameters: {num_params_train}')

    # print('Total number of parameters: %d' % num_params)
    # print('Total number of trainable parameters: %d' % num_params_train)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_dir, patience=20, min_stop_epoch=50, verbose=False, better='min'):
        """
        train_args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            min_stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.patience_counter = 0
        self.min_stop_epoch = min_stop_epoch
        self.better = better
        self.verbose = verbose
        self.best_score = None
        self.save_dir = save_dir

        if better == 'min':
            self.best_score = np.Inf
        else:
            self.best_score = -np.Inf
        self.early_stop = False
        self.counter = 0

    def is_new_score_better(self, score):
        if self.better == 'min':
            return score < self.best_score
        else:
            return score > self.best_score

    def __call__(self, epoch, score, save_ckpt_fn, save_ckpt_kwargs):
        is_better = self.is_new_score_better(score)
        if is_better:
            print(
                f'score improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
            self.save_checkpoint(save_ckpt_fn, save_ckpt_kwargs)
            self.counter = 0
            self.best_score = score
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch >= (self.min_stop_epoch - 1):
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, save_ckpt_fn, save_ckpt_kwargs):
        '''Saves model when score improves.'''
        if 'save_dir' in save_ckpt_kwargs:
            save_ckpt_fn(**save_ckpt_kwargs)
        else:
            save_ckpt_fn(save_dir=self.save_dir, **save_ckpt_kwargs)


def save_checkpoint(config, epoch, model, score, save_dir, fname=None):
    save_state = {'model': model.state_dict(),
                  'score': score,
                  'epoch': epoch,
                  'config': config}

    if fname is None:
        save_path = j_(save_dir, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = j_(save_dir, fname)

    torch.save(save_state, save_path)
