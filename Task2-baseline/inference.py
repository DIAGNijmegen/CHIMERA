import os
import subprocess
import sys

model_configs = "ABMIL_default"
fold = 10
model_type = "ABMIL_Fusion"
task = "BRS"
bag_size = 2000
batch_size = 1
in_dim = 1024
clinical_in_dim = 12 
epoch = 20

split_base = '/data/temporary/maryammohamm/FusionModelTask2/clinical_splits_miceforest_encoded'
data_source = "/data/temporary/maryammohamm/FusionModelTask2/features"
results_dir = "/data/temporary/maryammohamm/FusionModelTask2/results_PathologyPlusClinical_miceforest/"
repo_root = "/home/maryammohamm/task2_CHIMERA/ABMIL_CHIMERA"
os.environ['PYTHONPATH'] = repo_root

import pandas as pd
df_check = pd.read_csv(os.path.join(split_base, "fold_0", "train.csv"))
print(f"[INFO] Clinical input dim: {clinical_in_dim}")  

for model_config in model_configs:
    for k in range(fold):
        split_dir = os.path.join(split_base, f"fold_{k}")

        cmd = [
            sys.executable,
            '-m', 'training.main_classification',
            '--data_source', data_source,
            '--split_dir', split_dir,
            '--split_names', 'train,test',
            '--task', task,
            '--batch_size', str(batch_size),
            '--results_dir', results_dir,
            '--in_dim', str(in_dim),
            '--bag_size', str(bag_size),
            '--clinical_in_dim', str(clinical_in_dim),
            '--model_config', model_config,
            '--model_type', model_type,
            '--max_epochs', str(epoch),
        ]

        print(f"\n=== RUNNING FOLD {k} | CONFIG {model_config} ===\n")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            print(line, end='')

        process.wait()
        if process.returncode != 0:
            print(f" Error in fold {k}: Return code {process.returncode}")
        else:
            print(f" Completed fold {k}")
