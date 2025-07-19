import os
from os.path import join as j_
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report, accuracy_score)

from mil_models import create_downstream_model
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter,
                         get_optim, print_network, get_lr_scheduler)


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

    loss_fn = nn.CrossEntropyLoss()
    model = create_downstream_model(args, mode=mode).to(device)
    is_fusion_model = args.model_type == 'ABMIL_Fusion'

    print_network(model)

    optimizer = get_optim(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    early_stopper = EarlyStopping(save_dir=args.results_dir,
                                  patience=args.es_patience,
                                  min_stop_epoch=args.es_min_epochs,
                                  better='min',
                                  verbose=True) if args.early_stopping else None

    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        train_results = train_loop_classification(
           model, datasets['train'], optimizer, lr_scheduler, loss_fn,
           is_fusion_model=is_fusion_model,
           in_dropout=args.in_dropout, print_every=args.print_every,
           accum_steps=args.accum_steps
        )
        writer = log_dict_tensorboard(writer, train_results, 'train/', epoch)

        if 'val' in datasets:
            val_results, _ = validate_classification(model, datasets['val'], loss_fn,is_fusion_model=is_fusion_model)
            writer = log_dict_tensorboard(writer, val_results, 'val/', epoch)

            if early_stopper:
                score = val_results['loss']
                stop = early_stopper(epoch, score, save_checkpoint, {
                    'config': vars(args),
                    'epoch': epoch,
                    'model': model,
                    'score': score,
                    'fname': f's_checkpoint.pth'
                })
                if stop:
                    break

    # Save final model
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))

    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f"Evaluating on {k} set...")
        results[k], dumps[k] = validate_classification(
            model, loader, loss_fn, dump_results=True, is_fusion_model=(args.model_type == 'ABMIL_Fusion')
        )

        if k != 'train':
            log_dict_tensorboard(writer, results[k], f'final/{k}_', 0, verbose=True)

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
            loss = loss_fn(logits, label) / accum_steps
        else:
            attn_mask = batch.get('attn_mask', None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            out, log_dict = model(data, {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn})
            loss = out['loss'] / accum_steps
            logits = out['logits']

        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        acc = (label == logits.argmax(dim=-1)).float().mean()
        meters['cls_acc'].update(acc.item(), n=len(data))
        meters['bag_size'].update(data.size(1), n=len(data))

        if not is_fusion_model:
            for k, v in log_dict.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v, n=len(data))
        else:
            if 'loss' not in meters:
                meters['loss'] = AverageMeter()
            meters['loss'].update(loss.item(), n=len(data))

        if (batch_idx + 1) % print_every == 0:
            print(f"Epoch [{batch_idx+1}/{len(loader)}] - acc: {acc.item():.4f}")

    results = {k: meter.avg for k, meter in meters.items()}
    results['lr'] = optimizer.param_groups[0]['lr']
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

        attn_mask = batch.get('attn_mask', None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        if is_fusion_model:
                clinical = batch['clinical'].to(torch.float32).to(device)
                out = model(data, clinical)
                logits = out['logits']
                loss = loss_fn(logits, label)
                out['loss'] = loss
                log_dict = {'loss': loss.item()}
        else:
                out, log_dict = model(data, {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn})


        logits = out['logits']
        acc = (label == logits.argmax(dim=-1)).float().mean()
        meters['cls_acc'].update(acc.item(), n=len(data))
        meters['bag_size'].update(data.size(1), n=len(data))

        for k, v in log_dict.items():
            if k not in meters:
                meters[k] = AverageMeter()
            meters[k].update(v, n=len(data))

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
