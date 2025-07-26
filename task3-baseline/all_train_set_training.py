import os
import subprocess
import sys

model_config = "ABMIL_default"
model_type = "ABMIL_Fusion_RNA_Clinical"
task = "task3"
bag_size = 2000
batch_size = 4
in_dim = 1024
rna_dim = 19359
epoch = 20
l1_alpha = 0.01
seed = 42
clinical_dim = 13

# Paths to data
pathology_features = "/data/temporary/maryammohamm/FusionModelTask3/feats_pt"
rna_path = "/data/temporary/maryammohamm/FusionModelTask3/rna_data_row.csv"
clinical_path = "/data/temporary/maryammohamm/FusionModelTask3/task3_clinical_features_encoded.csv"
results_dir = "/data/temporary/maryammohamm/FusionModelTask3/results_PathologyPlusRNA/"
split_path = clinical_path

print(f"[INFO] Running full-data training (no folds). RNA dim = {rna_dim}")

cmd = [
    sys.executable,
    '-m', 'training.main_survival',
    '--data_source', pathology_features,
    '--split_path', split_path,
    '--task', task,
    '--batch_size', str(batch_size),
    '--results_dir', results_dir,
    '--in_dim', str(in_dim),
    '--bag_size', str(bag_size),
    '--clinical_dim', str(clinical_dim),
    '--rna_dim', str(rna_dim),
    '--model_type', model_type,
    '--max_epochs', str(epoch),
    '--exp_code', model_config,
    '--sample_col', 'sample_id',
    '--target_col', 'Time_to_prog_or_FUend',
    '--event_col', 'progression',
    '--l1_alpha', str(l1_alpha),
    '--lr', '1e-4',
    '--accum_steps', '1',
    '--seed', str(seed),
    '--lr_scheduler', 'cosine',
    '--warmup_steps', '100',
    '--wd', '1e-5',
    '--opt', 'adam',
    '--clinical_path', clinical_path,
    '--rna_path', rna_path
]

print(f"\n=== RUNNING FULL DATA | CONFIG {model_config} ===\n")

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end='')

process.wait()
if process.returncode != 0:
    print(f"❌ Error: Return code {process.returncode}")
else:
    print(f"✅ Completed full-data training")
