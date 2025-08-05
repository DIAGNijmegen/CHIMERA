import os
from os.path import join as j_
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    cohen_kappa_score, classification_report, accuracy_score
)
from mil_models import create_downstream_model
from utils.utils import (
    EarlyStopping, save_checkpoint, AverageMeter,
    print_network, get_lr_scheduler
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_dict_tensorboard(writer, results, str_prefix, step=0, verbose=False):
    for k, v in results.items():
        if verbose:
            print(f'{k}: {v:.4f}')
        writer.add_scalar(f'{str_prefix}{k}', v, step)
    return writer


def train(datasets, args, mode='classification'):
    assert mode == 'classification', "This trainer is now classification-only."

    writer_dir = args.results_dir
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir, flush_secs=15)

    # === Calculate class weights from training CSV ===
    train_df = pd.read_csv(args.train_csv)
    train_df["label_bin"] = train_df["label"].apply(lambda x: 1 if x == "BRS3" else 0)
    class_counts = train_df["label_bin"].value_counts().sort_index()
    weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32)
    weights = weights / weights.sum()
    print(f"[INFO] Training class counts: {class_counts.to_dict()}")
    print(f"[INFO] Using class weights: {weights.tolist()}")

    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    # === Model ===
    model = create_downstream_model(args, mode=mode).to(device)
    is_fusion_model = getattr(args, "model_type", None) == 'ABMIL_Fusion'
    print_network(model)

    # === Optimizer ===
    if not hasattr(args, 'wd') or args.wd == 0.0:
        args.wd = 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    # === Early stopping ===
    early_stopper = EarlyStopping(
        save_dir=args.results_dir,
        patience=args.es_patience,
        min_stop_epoch=args.es_min_epochs,
        better='max',
        verbose=True
    ) if args.early_stopping else None

    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")

        # Train loop
        train_results = train_loop_classification(
            model, datasets['train'], optimizer, lr_scheduler, loss_fn,
            is_fusion_model=is_fusion_model,
            in_dropout=args.in_dropout, print_every=args.print_every,
            accum_steps=args.accum_steps
        )
        writer = log_dict_tensorboard(writer, train_results, 'train/', epoch)

        # Validation
        if 'val' in datasets:
            val_results, _ = validate_classification(
                model, datasets['val'], loss_fn, is_fusion_model=is_fusion_model
            )
            writer = log_dict_tensorboard(writer, val_results, 'val/', epoch)

            if early_stopper:
                score = val_results['acc']
                stop = early_stopper(epoch, score, save_checkpoint, {
                    'config': vars(args),
                    'epoch': epoch,
                    'model': model,
                    'score': score,
                    'fname': f's_checkpoint.pth'
                })
                if stop:
                    print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                    break

    # Save final model
    final_model_path = j_(args.results_dir, f"s_checkpoint.pth")
    if args.early_stopping:
        if os.path.exists(final_model_path):
            model.load_state_dict(torch.load(final_model_path)['model'])
        else:
            torch.save({'model': model.state_dict()}, final_model_path)
    else:
        torch.save({'model': model.state_dict()}, final_model_path)

    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"[ERROR] Final model was not saved at {final_model_path}")

    # Evaluate
    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f"Evaluating on {k} set...")
        results[k], dumps[k] = validate_classification(
            model, loader, loss_fn, dump_results=True, is_fusion_model=is_fusion_model
        )
        if k != 'train':
            log_dict_tensorboard(writer, results[k], f'final/{k}_', 0, verbose=True)

    # Save metrics
    metrics_path_json = j_(args.results_dir, "metrics.json")
    metrics_path_csv = j_(args.results_dir, "metrics.csv")
    with open(metrics_path_json, "w") as f:
        json.dump(results, f, indent=4)
    flat_metrics = {f"{split}_{metric}": value for split, metrics in results.items() for metric, value in metrics.items()}
    pd.DataFrame([flat_metrics]).to_csv(metrics_path_csv, index=False)

    print(f"[INFO] Metrics saved to {metrics_path_json} and {metrics_path_csv}")
    writer.close()
    return results, dumps


def train_loop_classification(model, loader, optimizer, lr_scheduler, loss_fn,
                              is_fusion_model=False, in_dropout=0.0, print_every=50, accum_steps=1):
    model.train()
    meters = {'bag_size': AverageMeter(), 'cls_acc': AverageMeter()}

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(torch.long).to(device)
        if len(label.shape) == 2 and label.shape[1] == 1:
            label = label.squeeze(dim=-1)

        if in_dropout:
            data = F.dropout(data, p=in_dropout)

        if is_fusion_model:
            clinical = batch['clinical'].to(torch.float32).to(device)
            out = model(data, clinical)
            logits = out['logits']
            loss = loss_fn(logits, label)
        else:
            attn_mask = batch.get('attn_mask', None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            out, _ = model(data, {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn})
            logits = out['logits']
            loss = out['loss']

        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        acc = (label == logits.argmax(dim=-1)).float().mean()
        meters['cls_acc'].update(acc.item(), n=len(data))
        meters['bag_size'].update(data.size(1), n=len(data))

        if (batch_idx + 1) % print_every == 0:
            print(f"Batch [{batch_idx+1}/{len(loader)}] - acc: {acc.item():.4f}")

    results = {k: meter.avg for k, meter in meters.items()}
    results['lr'] = optimizer.param_groups[0]['lr']
    results['loss'] = loss.item()
    return results


@torch.no_grad()
def validate_classification(model, loader, loss_fn=None, print_every=50, dump_results=False, verbose=1, is_fusion_model=False):
    model.eval()
    all_probs, all_labels = [], []
    meters = {'bag_size': AverageMeter(), 'cls_acc': AverageMeter()}

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(torch.long).to(device)
        if len(label.shape) == 2 and label.shape[1] == 1:
            label = label.squeeze(dim=-1)

        if is_fusion_model:
            clinical = batch['clinical'].to(torch.float32).to(device)
            out = model(data, clinical)
            logits = out['logits']
            loss = loss_fn(logits, label) if loss_fn else None
        else:
            attn_mask = batch.get('attn_mask', None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            out, _ = model(data, {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn})
            logits = out['logits']
            loss = out['loss'] if 'loss' in out else None

        acc = (label == logits.argmax(dim=-1)).float().mean()
        meters['cls_acc'].update(acc.item(), n=len(data))
        meters['bag_size'].update(data.size(1), n=len(data))

        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = all_probs.argmax(axis=1)

    results = sweep_classification_metrics(all_probs, all_labels, all_preds, all_probs.shape[1])
    results.update({k: meter.avg for k, meter in meters.items()})

    dumps = {}
    if dump_results:
        dumps['labels'] = all_labels
        dumps['probs'] = all_probs

    return results, dumps


@torch.no_grad()
def sweep_classification_metrics(all_probs, all_labels, all_preds=None, n_classes=None):
    if all_preds is None:
        all_preds = all_probs.argmax(axis=1)

    if n_classes == 2:
        all_probs = all_probs[:, 1]
        roc_kwargs = {}
    else:
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

    return {
        'acc': accuracy_score(all_labels, all_preds),
        'bacc': balanced_accuracy_score(all_labels, all_preds),
        'kappa': cohen_kappa_score(all_labels, all_preds, weights='quadratic'),
        'roc_auc': roc_auc_score(all_labels, all_probs, **roc_kwargs),
        'weighted_f1': classification_report(all_labels, all_preds, output_dict=True)['weighted avg']['f1-score']
    }
