import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# === Settings
clinical_csv = "/data/temporary/maryammohamm/FusionModelTask3/task3_clinical_features_encoded.csv"
split_base = "/data/temporary/maryammohamm/FusionModelTask3/split_clinical"
num_folds = 5
sample_col = "sample_id"
event_col = "progression"
time_col = "Time_to_prog_or_FUend"

# === Load clinical data
df = pd.read_csv(clinical_csv)
print(f"✅ Loaded {df.shape[0]} clinical samples")

# === Setup KFold stratified by Progression (event)
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df[event_col])):
    fold_dir = os.path.join(split_base, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df.iloc[train_idx][[sample_col, time_col, event_col]]
    test_df = df.iloc[test_idx][[sample_col, time_col, event_col]]

    train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

    print(f"✅ Fold {fold}: {len(train_df)} train / {len(test_df)} test")

print("✅ All folds saved to:", split_base)
