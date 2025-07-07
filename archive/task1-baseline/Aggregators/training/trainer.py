import os
from os.path import join as j_
import pdb
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report, accuracy_score)

from mil_models import create_downstream_model
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss, CoxLoss_Time
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter,
                         get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## GENERIC
def log_dict_tensorboard(writer, results, str_prefix, step=0, verbose=False):
    for k, v in results.items():
        if verbose: print(f'{k}: {v:.4f}')
        writer.add_scalar(f'{str_prefix}{k}', v, step)
    return writer


def train(datasets, args, mode='classification'):
    """
    Train for a single fold for classification or suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    writer = SummaryWriter(writer_dir, flush_secs=15)

    assert args.es_metric == 'loss'
    if mode == 'classification':
        loss_fn = nn.CrossEntropyLoss()



    elif mode == 'survival':
        if args.loss_fn == 'nll':
            loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
        elif args.loss_fn == 'cox':
            loss_fn = CoxLoss()
        elif args.loss_fn == 'cox_time':
            loss_fn = CoxLoss_Time()
        elif args.loss_fn == 'rank':
            loss_fn = SurvRankingLoss()
        elif args.loss_fn == 'sumo':
            # Add placeholder for SUMO loss if implemented
            raise NotImplementedError("SUMO loss not implemented yet")
        elif args.loss_fn == 'ipcwls':
            # Add placeholder for IPCWLS loss if implemented  
            raise NotImplementedError("IPCWLS loss not implemented yet")
        else:
            raise ValueError(f"Unknown loss function: {args.loss_fn}")

    print('\nInit Model...', end=' ')


    model = create_downstream_model(args, mode=mode)
    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    print(datasets)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        print('\nNo EarlyStopping...', end=' ')
        early_stopper = None
    
    # Initialize epoch-by-epoch metrics tracking
    epoch_metrics = []
    epoch_log_path = j_(args.results_dir, 'epoch_metrics.csv')
    
    #####################
    # The training loop #
    #####################
    for epoch in range(args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        if mode == 'classification':
            train_results = train_loop_classification(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                                      in_dropout=args.in_dropout, print_every=args.print_every,
                                                      accum_steps=args.accum_steps)
        elif mode == 'survival':
            train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                                in_dropout=args.in_dropout, print_every=args.print_every,
                                                accum_steps=args.accum_steps)

        writer = log_dict_tensorboard(writer, train_results, 'train/', epoch)
        
        # Print training results summary after each epoch
        if mode == 'survival':
            print("\n=== TRAIN EPOCH RESULTS ===")
            msg = [f"{k}: {v:.4f}" for k, v in train_results.items()]
            print("\t".join(msg))
            print("===========================\n")
        elif mode == 'classification':
            print("\n=== TRAIN EPOCH RESULTS ===")
            msg = [f"{k}: {v:.4f}" for k, v in train_results.items()]
            print("\t".join(msg))
            print("===========================\n")

        # Save per-epoch training metrics to CSV
        if mode == 'survival':
            epoch_row = {
                'epoch': epoch,
                'split': 'train',
                'bag_size': train_results.get('bag_size', 0.0),
                'c_index': train_results.get('c_index', 0.0),
                'loss': train_results.get('loss', 0.0),
                'surv_loss': train_results.get('surv_loss', train_results.get('loss', 0.0))
            }
            epoch_metrics.append(epoch_row)
        elif mode == 'classification':
            epoch_row = {
                'epoch': epoch,
                'split': 'train',
                'bag_size': train_results.get('bag_size', 0.0),
                'cls_acc': train_results.get('cls_acc', 0.0),
                'loss': train_results.get('loss', 0.0)
            }
            epoch_metrics.append(epoch_row)

        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            if mode == 'classification':
                val_results, _ = validate_classification(model, datasets['val'], loss_fn,
                                                         print_every=args.print_every, verbose=True)
            elif mode == 'survival':
                val_results, _ = validate_survival(model, datasets['val'], loss_fn,
                                                   print_every=args.print_every, verbose=True)

            writer = log_dict_tensorboard(writer, val_results, 'val/', epoch)

            ### Check Early Stopping (Optional)
            if early_stopper is not None:
                if args.es_metric == 'loss':
                    score = val_results['loss']

                else:
                    raise NotImplementedError
                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        print('#' * (22 + len(f'TRAIN Epoch: {epoch}')), '\n')

    ### End of training: Use best model if early stopping was used, otherwise current model
    checkpoint_path = j_(args.results_dir, "s_checkpoint.pth")
    if args.early_stopping and os.path.exists(checkpoint_path):
        # Load the best model found during early stopping
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("Loaded best model from early stopping for evaluation")
    else:
        # Save current model state (either no early stopping or no checkpoint saved)
        torch.save({'model': model.state_dict()}, checkpoint_path)
        print("Using final trained model for evaluation")

    ### Evaluate on all splits
    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f'End of training. Evaluating on Split {k.upper()}...:')
        if mode == 'classification':
            results[k], dumps[k] = validate_classification(model, loader, loss_fn, print_every=args.print_every,
                                                           dump_results=True, verbose=False)
        elif mode == 'survival':
            results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                     dump_results=True, verbose=False)

        if k == 'train':
            # Keep train results in summary for monitoring training progress
            log_dict_tensorboard(writer, results[k], f'final/{k}_', 0, verbose=True)
        else:
            log_dict_tensorboard(writer, results[k], f'final/{k}_', 0, verbose=True)

    # Add final evaluation metrics to epoch_metrics for comprehensive tracking
    for split_name, split_results in results.items():
        if mode == 'survival':
            final_row = {
                'epoch': 'final',
                'split': split_name,
                'bag_size': split_results.get('bag_size', 0.0),
                'c_index': split_results.get('c_index', 0.0),
                'loss': split_results.get('loss', 0.0),
                'surv_loss': split_results.get('surv_loss', split_results.get('loss', 0.0))
            }
            epoch_metrics.append(final_row)
        elif mode == 'classification':
            final_row = {
                'epoch': 'final',
                'split': split_name,
                'bag_size': split_results.get('bag_size', 0.0),
                'cls_acc': split_results.get('cls_acc', 0.0),
                'loss': split_results.get('loss', 0.0)
            }
            epoch_metrics.append(final_row)

    # Save epoch-by-epoch metrics to CSV file
    if epoch_metrics:
        epoch_df = pd.DataFrame(epoch_metrics)
        epoch_df.to_csv(epoch_log_path, index=False)
        print(f"\nSaved epoch-by-epoch metrics to: {epoch_log_path}")
        print(f"File contains {len(epoch_metrics)} rows of metrics data")

    writer.close()
    return results, dumps


## CLASSIFICATION
def train_loop_classification(model, loader, optimizer, lr_scheduler, loss_fn=None,
                              in_dropout=0.0, print_every=50,
                              accum_steps=1):
    model.train()
    meters = {'bag_size': AverageMeter(), 'cls_acc': AverageMeter()}
    bag_size_meter = meters['bag_size']
    acc_meter = meters['cls_acc']

    #import pdb; pdb.set_trace()
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(torch.long).to(device)
        if len(label.shape) == 2 and label.shape[1] == 1:
            label = label.squeeze(dim=-1)

        if in_dropout:
            data = F.dropout(data, p=in_dropout)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        model_kwargs = {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn}
        out, log_dict = model(data, model_kwargs)

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration classification-specific metrics to calculate / log
        logits = out['logits']
        acc = (label == logits.argmax(dim=-1)).float().mean()

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        acc_meter.update(acc.item(), n=len(data))
        bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch classification-specific metrics to calculate / log
    results = {k: meter.avg for k, meter in meters.items()}
    results['lr'] = optimizer.param_groups[0]['lr']
    return results

@torch.no_grad()
def validate_classification(model, loader,
                            loss_fn=None,
                            print_every=50,
                            dump_results=False,
                            verbose=1):
    model.eval()
    meters = {'bag_size': AverageMeter(), 'cls_acc': AverageMeter()}
    acc_meter = meters['cls_acc']
    bag_size_meter = meters['bag_size']
    all_probs = []
    all_labels = []
        
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(torch.long).to(device)
        if len(label.shape) == 2 and label.shape[1] == 1:
            label = label.squeeze(dim=-1)

        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        model_kwargs = {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn}
        out, log_dict = model(data, model_kwargs)
        

        # End of iteration classification-specific metrics to calculate / log
        logits = out['logits']
        acc = (label == logits.argmax(dim=-1)).float().mean()
        acc_meter.update(acc.item(), n=len(data))
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch classification-specific metrics to calculate / log
    n_classes = logits.size(1)
      
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = all_probs.argmax(axis=1)

    results = sweep_classification_metrics(all_probs, all_labels, all_preds=all_preds, n_classes=n_classes)
    results.update({k: meter.avg for k, meter in meters.items()})

    if 'report' in results:
        del results['report']

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['labels'] = all_labels
        dumps['probs'] = all_probs
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
    return results, dumps


## SURVIVAL
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, in_dropout=0.0, print_every=50,
                        accum_steps=1):
    model.train()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    # For Cox loss, we need to accumulate multiple samples before computing loss
    is_cox_loss = isinstance(loss_fn, (CoxLoss, CoxLoss_Time))
    
    if is_cox_loss:
        # Accumulate samples for Cox loss computation
        accumulated_outputs = []
        accumulated_labels = []
        accumulated_times = []
        accumulated_censorships = []
    
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)

        if in_dropout:
            data = F.dropout(data, p=in_dropout)
        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        
        if is_cox_loss:
            # For Cox loss, get model output without computing loss yet
            model_kwargs = {'attn_mask': attn_mask, 'label': label, 'censorship': censorship, 'loss_fn': None}
            out, log_dict = model(data, model_kwargs)
            
            # Accumulate outputs for batch processing
            # Handle different model output keys
            if 'logits' in out:
                accumulated_outputs.append(out['logits'])
            elif 'predicted_time' in out:
                accumulated_outputs.append(out['predicted_time'])
            else:
                raise ValueError(f"Model output must contain either 'logits' or 'predicted_time', got keys: {list(out.keys())}")
                
            accumulated_labels.append(label)
            accumulated_times.append(event_time)
            accumulated_censorships.append(censorship)
            
            # Initialize loss tracking - set to None initially for Cox losses during accumulation
            out['loss'] = None
            if 'loss' not in log_dict:
                log_dict['loss'] = 0.0
            if 'loss' not in out:
                out['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                
        else:
            # For non-Cox losses, compute normally
            model_kwargs = {'attn_mask': attn_mask, 'label': label, 'censorship': censorship, 'loss_fn': loss_fn}
            out, log_dict = model(data, model_kwargs)

        # Process accumulated Cox loss every accum_steps or at end
        if is_cox_loss and ((batch_idx + 1) % accum_steps == 0 or batch_idx == len(loader) - 1):
            if accumulated_outputs:
                # Combine accumulated samples
                combined_outputs = torch.cat(accumulated_outputs, dim=0)
                combined_times = torch.cat(accumulated_times, dim=0)
                combined_censorships = torch.cat(accumulated_censorships, dim=0)
                
                # Debug: print accumulation info
                if batch_idx % (print_every * accum_steps) == 0:
                    print(f"Cox loss computation: {len(accumulated_outputs)} samples accumulated")
                
                # Compute Cox loss on accumulated batch
                # Note: Cox loss functions expect parameter named 'logits' regardless of whether 
                # it's actual logits (risks) or predicted times
                cox_loss_dict = loss_fn(logits=combined_outputs, 
                                      times=combined_times, 
                                      censorships=combined_censorships)
                loss = cox_loss_dict['loss']
                
                # Update log_dict with actual loss
                log_dict.update(cox_loss_dict)
                out['loss'] = loss
                
                # Clear accumulation
                accumulated_outputs = []
                accumulated_labels = []
                accumulated_times = []
                accumulated_censorships = []
            else:
                # No accumulated outputs - this shouldn't happen but handle gracefully
                if 'loss' not in out:
                    out['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Handle loss computation and backprop
        current_loss = out.get('loss', None)
        if current_loss is not None and current_loss != 0:
            loss = current_loss
            if not is_cox_loss:  # For non-Cox losses, apply accumulation division
                loss = loss / accum_steps
            loss.backward()
            
        # Optimizer step
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        # Only collect metrics when loss is computed (not during Cox accumulation)
        if out.get('loss') is not None:
            # Note: For Cox losses, 'out' might not have the expected keys during accumulation
            if 'risk' in out:
                all_risk_scores.append(out['risk'].detach().cpu().numpy())
            elif 'predicted_time' in out:
                # For time predictions, use negative time as risk for c-index
                all_risk_scores.append((-out['predicted_time']).detach().cpu().numpy())
            elif 'logits' in out:
                # Fallback: use logits as risk
                all_risk_scores.append(out['logits'].detach().cpu().numpy())
            else:
                # This shouldn't happen for computed losses, but provide a fallback
                print(f"Warning: No suitable output key found in {list(out.keys())}")
                all_risk_scores.append(torch.zeros(1).numpy())
                
            all_censorships.append(censorship.cpu().numpy())
            all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            if isinstance(val, torch.Tensor):
                val = val.item()
            meters[key].update(val, n=len(data))

        bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      verbose=1):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []


    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        model_kwargs = {'attn_mask': attn_mask, 'label': label, 'censorship': censorship, 'loss_fn': loss_fn}
        out, log_dict = model(data, model_kwargs)

        # End of iteration survival-specific metrics to calculate / log
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        
        # Handle both risk and time predictions
        if 'risk' in out:
            all_risk_scores.append(out['risk'].cpu().numpy())
        elif 'predicted_time' in out:
            # For time predictions, use negative time as risk for c-index
            all_risk_scores.append((-out['predicted_time']).cpu().numpy())
        else:
            raise ValueError("Model output must contain either 'risk' or 'predicted_time'")
            
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})
    elif recompute_loss_at_end and isinstance(loss_fn, CoxLoss_Time):
        # For CoxLoss_Time, we need to pass the predicted times (negative of risk scores)
        surv_loss_dict = loss_fn(logits=-torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        
        # For time prediction models, also store the predicted times
        # We need to check the last batch output to see if it contains predicted_time
        if 'predicted_time' in out:
            dumps['all_predicted_times'] = (-all_risk_scores)  # Convert risk back to time
    return results, dumps


@torch.no_grad()
def sweep_classification_metrics(all_probs, all_labels, all_preds=None, n_classes=None):
    if n_classes is None:
        n_classes = all_probs.shape[1]

    if all_preds is None:
        all_preds = all_probs.argmax(axis=1)

    if n_classes == 2:
        all_probs = all_probs[:, 1]
        roc_kwargs = {}
    else:
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

    bacc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    cls_rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs, **roc_kwargs)

    results = {'acc': acc,
               'bacc': bacc,
               'report': cls_rep,
               'kappa': kappa,
               'roc_auc': roc_auc,
               'weighted_f1': cls_rep['weighted avg']['f1-score']}
    return results