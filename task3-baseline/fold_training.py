import os
import subprocess
import sys

model_configs = ["ABMIL_default"]
num_folds = 5
model_type = "ABMIL_Fusion_RNA_Clinical"
task = "task3"
bag_size = 2000
batch_size = 4
split_base = '/data/temporary/maryammohamm/FusionModelTask3/split_clinical'  
in_dim = 1024
rna_dim = 19359
epoch = 20
l1_alpha = 0.01
seed = 42
clinical_dim= 13
# Paths to data
pathology_features = "/data/temporary/maryammohamm/FusionModelTask3/feats_pt"
rna_path = "/data/temporary/maryammohamm/FusionModelTask3/rna_data_row.csv"
clinical_path = "/data/temporary/maryammohamm/FusionModelTask3/task3_clinical_features_encoded.csv"


results_dir = "/data/temporary/maryammohamm/FusionModelTask3/results_PathologyPlusRNA/"


print(f"[INFO] Using clinical-based split. RNA dim = {rna_dim}")

for model_config in model_configs:
    for k in range(num_folds):
        split_dir = os.path.join(split_base, f"fold_{k}")

        cmd = [
            sys.executable,
            '-m', 'training.main_survival',
            '--data_source', pathology_features,
            '--split_path', split_dir,
            '--task', task,
            '--clinical_dim',str(clinical_dim),
            '--batch_size', str(batch_size),
            '--results_dir', results_dir,
            '--accum_steps', '1',
            '--warmup_steps', '100',
            '--in_dim', str(in_dim),
            '--bag_size', str(bag_size),
            '--rna_dim', str(rna_dim),
            '--model_type', model_type,
            '--max_epochs', str(epoch),
            '--exp_code', model_config,
            '--sample_col', 'sample_id',
            '--target_col', 'Time_to_prog_or_FUend',
            '--event_col', 'progression',
            '--l1_alpha', str(l1_alpha),
            '--lr', '1e-4',
            '--lr_scheduler', 'cosine',
            '--seed', str(seed),
            '--wd', '1e-5',
            '--opt', 'adam',
            '--clinical_path', clinical_path,
            '--rna_path', rna_path
        ]

        print(f"\n=== RUNNING FOLD {k} | CONFIG {model_config} ===\n")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        print(" ".join(cmd))  # 👈 Add this line to print full shell command

        for line in process.stdout:
            print(line, end='')

        process.wait()
        if process.returncode != 0:
            print(f"❌ Error in fold {k}: Return code {process.returncode}")
        else:
            print(f"✅ Completed fold {k}")

