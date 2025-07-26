import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from lifelines.utils import concordance_index

from .model_utils import create_downstream_model
from utils.my_utils import (
    get_optim,
    get_lr_scheduler,
    AverageMeter,
    EarlyStopping,
    log_dict_tensorboard,
    j_,
    save_checkpoint,
    print_network
)

def train(datasets, args, mode='survival'):
    assert mode == 'survival', "Only survival mode is supported."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer_dir = args.results_dir
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir, flush_secs=15)

    # === Create model
    model = create_downstream_model(args, mode=mode).to(device)
    print_network(model)

    # === Optimizer and scheduler
    optimizer = get_optim(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    # === Early stopping
    early_stopper = EarlyStopping(save_dir=args.results_dir,
                                  patience=args.es_patience,
                                  min_stop_epoch=args.es_min_epochs,
                                  better='min',
                                  verbose=True) if args.early_stopping else None

    # === Training epochs
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        train_results = train_loop_survival(
            model, datasets['train'], optimizer, lr_scheduler,
            print_every=args.print_every, device=device, l1_alpha=args.l1_alpha
        )
        writer = log_dict_tensorboard(writer, train_results, 'train/', epoch)

        if 'val' in datasets:
            val_results, _ = validate_survival(model, datasets['val'], device=device)
            writer = log_dict_tensorboard(writer, val_results, 'val/', epoch)

            if early_stopper and early_stopper(epoch, -val_results['c_index'], save_checkpoint, {
                'config': vars(args), 'epoch': epoch, 'model': model, 'score': -val_results['c_index'], 'fname': f's_checkpoint.pth'
            }):
                break

    # === Save checkpoint
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))

    # === Final evaluation
    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f"Evaluating on {k} set...")
        results[k], dumps[k] = validate_survival(model, loader, device=device)
        if k != 'train':
            log_dict_tensorboard(writer, results[k], f'final/{k}_', 0, verbose=True)

    writer.close()
    return results, dumps


def train_loop_survival(model, loader, optimizer, lr_scheduler, print_every=50, device='cuda', l1_alpha=1e-4):
    model.train()
    meters = {'bag_size': AverageMeter(), 'loss': AverageMeter()}

    for batch_idx, batch in enumerate(loader):
        img = batch['img'].to(device)
        rna = batch['rna'].to(torch.float32).to(device)
        clinical = batch['clinical'].to(torch.float32).to(device)
        time = batch['time'].to(torch.float32).to(device)
        event = batch['event'].to(torch.float32).to(device)

        # Forward + Cox loss
        out, cox_loss, l1_penalty = model(img, rna, clinical,
                                          survival_time=time,
                                          censorship=event,
                                          return_l1=True)
        loss = cox_loss + l1_alpha * l1_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        meters['loss'].update(loss.item(), n=img.size(0))
        meters['bag_size'].update(img.size(1), n=img.size(0))

        if (batch_idx + 1) % print_every == 0:
            print(f"Batch [{batch_idx+1}/{len(loader)}] - loss: {loss.item():.4f}")

    results = {k: meter.avg for k, meter in meters.items()}
    results['lr'] = optimizer.param_groups[0]['lr']
    return results


@torch.no_grad()
def validate_survival(model, loader, device='cuda'):
    model.eval()
    all_risks, all_times, all_events = [], [], []

    for batch in loader:
        img = batch['img'].to(device)
        rna = batch['rna'].to(torch.float32).to(device)
        clinical = batch['clinical'].to(torch.float32).to(device)
        time = batch['time'].cpu().numpy()
        event = batch['event'].cpu().numpy()

        out = model(img, rna, clinical)
        risk = out['risk'].detach().cpu().numpy()

        all_risks.append(risk)
        all_times.append(time)
        all_events.append(event)

    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)

    c_index = concordance_index(
        event_times=all_times,
        predicted_scores=-all_risks,
        event_observed=all_events
    )

    return {'c_index': c_index}, {
        'risk': all_risks,
        'time': all_times,
        'event': all_events
    }
