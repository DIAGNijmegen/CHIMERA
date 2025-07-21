import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from wsi_datasets.wsi_classification import WSIClassificationDataset
from utils.utils import read_splits, build_model_from_args
from utils.file_utils import load_pkl
from training.trainer import run_inference

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading split: {args.split_dir}")
    csv_splits = read_splits(args)
    test_df = csv_splits['test']

    dataset_kwargs = dict(
        data_source=[args.data_source],
        label_map=args.label_map,
        target_col=args.target_col,
        bag_size=args.val_bag_size,
    )

    test_dataset = WSIClassificationDataset(test_df, **dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print("[INFO] Building model...")
    model = build_model_from_args(args).to(device)

    checkpoint_path = os.path.join(args.model_ckpt, "best_model.pth")
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("[INFO] Running inference...")
    preds, probs, gts = run_inference(model, test_loader, device)

    results_df = pd.DataFrame({
        "slide_id": test_df['case_id'].values,
        "true_label": gts,
        "pred_label": preds,
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1]
    })

    os.makedirs(args.out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(args.out_dir, "inference_results.csv"), index=False)
    print(f"[INFO] Results saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--split_dir', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--in_dim', type=int, required=True)
    parser.add_argument('--clinical_in_dim', type=int, required=True)
    parser.add_argument('--val_bag_size', type=int, default=-1)
    parser.add_argument('--target_col', type=str, default="label")
    parser.add_argument('--task', type=str, default="BRS")
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    # label mapping
    label_dicts = {
        "BRS": {"BRS1": 0, "BRS2": 0, "BRS3": 1}
    }
    args.label_map = label_dicts[args.task]
    args.n_classes = len(set(args.label_map.values()))

    main(args)

