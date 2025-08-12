import os
import subprocess
import sys

# === CONFIGURATION ===
model_config = "ABMIL_default"
model_type = "ABMIL_Fusion"
task = "BRS"
bag_size = 500
batch_size = 1
epoch = 10

# === PATHS ===
clinical_train = '/data/temporary/maryammohamm/FusionModelTask2/processed_clinical_train.csv'
feature_train = '/data/temporary/maryammohamm/FusionModelTask2/features/'
clinical_val = '/data/temporary/maryammohamm/FusionModelTask2/processed_clinical_val.csv'
feature_val = '/data/temporary/chimera/bladder/features_validation_task2_uni_maryam/features_validation_task2_uni_maryam/features'

results_dir = "/data/temporary/maryammohamm/FusionModelTask2/result_TrainValidation/"
repo_root = "/home/maryammohamm/task2_CHIMERA2/ABMIL_CHIMERA"

os.makedirs(results_dir, exist_ok=True)
os.environ['PYTHONPATH'] = repo_root

# === Run training ===
fold_results_dir = os.path.join(results_dir, "train_val_run")
os.makedirs(fold_results_dir, exist_ok=True)

cmd = [
    sys.executable,
    '-m', 'training.main_classification',
    '--train_csv', clinical_train,
    '--train_feats_dir', feature_train,
    '--val_csv', clinical_val,
    '--val_feats_dir', feature_val,
    '--model_config', os.path.join(repo_root, 'configs', model_config, 'config.json'),
    '--results_dir', fold_results_dir,
    '--bag_size', str(bag_size),
    '--batch_size', str(batch_size),
    '--max_epochs', str(epoch),
    '--model_type', model_type,
    '--early_stopping'
]

print(f"\n=== RUNNING TRAIN + VALIDATION | CONFIG {model_config} ===\n")
subprocess.run(cmd)
