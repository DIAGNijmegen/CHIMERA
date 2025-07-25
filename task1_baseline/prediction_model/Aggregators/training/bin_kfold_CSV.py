# ==============================================================================
# CORRECTED K-FOLD SPLITTING SCRIPT FOR SURVIVAL ANALYSIS
# ==============================================================================
# This script creates balanced 5-fold cross-validation splits.
#
# STRATEGY:
# It uses a simplified and targeted stratification approach to solve the
# issue of covariate shift that previously caused poor performance on fold 3.
#
# 1. It stratifies on the binary survival outcome (event/censored).
# 2. It ALSO stratifies on binned pre-operative PSA, the key feature that
#    was previously imbalanced.
#
# This ensures that each fold's training and test split has a similar
# distribution of both event rates AND key prognostic features.
# ==============================================================================

import os
import json
import pandas as pd
import glob
import argparse
from sklearn.model_selection import StratifiedKFold

def load_clinical_data(clin_dat_path: str) -> pd.DataFrame:
    """
    Load clinical data from JSON files, including key prognostic variables.
    """
    clinical_data = []
    json_files = glob.glob(os.path.join(clin_dat_path, "*.json"))
    print(f"Found {len(json_files)} JSON files in {clin_dat_path}")

    for json_file in json_files:
        patient_id = os.path.basename(json_file).replace('.json', '')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            bcr_months = data.get("time_to_follow-up/BCR")
            bcr_status = data.get("BCR")

            if bcr_status == "1.0":
                censorship = 0  # Event (BCR) occurred
            elif bcr_status == "0.0":
                censorship = 1  # Censored (no BCR)
            else:
                continue

            if bcr_months is not None and bcr_months > 0:
                clinical_record = {
                    'patient_id': patient_id,
                    'bcr_months': bcr_months,
                    'censorship': censorship,
                    'psa': data.get("pre_operative_PSA"),
                    'isup_grade': data.get("ISUP")
                }
                clinical_data.append(clinical_record)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return pd.DataFrame(clinical_data)

def find_feature_files(feat_path: str, patient_ids: list) -> pd.DataFrame:
    """
    Find feature files for each patient.
    """
    feature_data = []
    for patient_id in patient_ids:
        pattern = os.path.join(feat_path, f"{patient_id}_*.pt")
        pt_files = glob.glob(pattern)
        if pt_files:
            feature_data.append({
                'patient_id': patient_id,
                'slide_id': os.path.basename(pt_files[0]).replace('.pt', '')
            })
    return pd.DataFrame(feature_data)

def create_merged_dataset(clinical_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge clinical and feature data, creating one row per patient.
    """
    merged_df = feature_df.merge(clinical_df, on='patient_id', how='inner')
    merged_df = merged_df.rename(columns={
        'bcr_months': 'bcr_survival_months',
        'censorship': 'bcr_censorship',
        'patient_id': 'case_id'
    })
    return merged_df

def create_stratification_variable(df: pd.DataFrame) -> pd.Series:
    """
    Create a SIMPLIFIED composite stratification variable for balanced folds.
    This version stratifies ONLY on binned PSA and BCR status.
    """
    print("\nCreating simplified stratification variable...")
    df_copy = df.copy()
    df_copy['psa'] = pd.to_numeric(df_copy['psa'], errors='coerce')

    if df_copy['psa'].isna().any():
        psa_median = df_copy['psa'].median()
        print(f"Warning: Missing PSA values found. Filling with median ({psa_median:.2f}).")
        df_copy['psa'].fillna(psa_median, inplace=True)

    # Bin PSA into 4 quantile-based groups (quartiles)
    try:
        # `duplicates='drop'` handles cases where bin edges are not unique
        psa_binned = pd.qcut(df_copy['psa'], q=4, labels=[f'PSA_Q{i+1}' for i in range(4)], duplicates='drop')
    except ValueError:
        print("Warning: Could not create 4 PSA bins. Falling back to 3 bins.")
        psa_binned = pd.qcut(df_copy['psa'], q=3, labels=[f'PSA_Q{i+1}' for i in range(3)], duplicates='drop')

    # Get BCR status as a string
    bcr_str = df_copy['bcr_censorship'].apply(lambda x: 'event' if x == 0 else 'censored')

    # Combine into a simple, robust stratum
    strata = psa_binned.astype(str) + '_' + bcr_str
    print("Final stratum distribution:")
    print(strata.value_counts())
    return strata

def perform_cross_validation(merged_df: pd.DataFrame, output_dir: str, n_splits: int = 5) -> None:
    """
    Perform stratified k-fold cross-validation and save the splits.
    Includes validation statistics to confirm balance.
    """
    stratification_var = create_stratification_variable(merged_df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)

    fold_stats = []
    print("\n--- Performing Cross-Validation ---")
    for fold, (train_idx, test_idx) in enumerate(skf.split(merged_df, stratification_var)):
        train_df = merged_df.iloc[train_idx]
        test_df = merged_df.iloc[test_idx]

        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        columns_to_save = ['slide_id', 'bcr_survival_months', 'bcr_censorship', 'case_id']
        train_df[columns_to_save].to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        test_df[columns_to_save].to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

        # Collect statistics for validation
        stats = {
            'Fold': fold,
            'Train Size': len(train_df),
            'Test Size': len(test_df),
            'Test Event Rate': (test_df['bcr_censorship'] == 0).mean(),
            'Test Median PSA': test_df['psa'].median(),
            'Test Median ISUP': test_df['isup_grade'].median()
        }
        fold_stats.append(stats)
        print(f"Fold {fold}: Train={len(train_df)}, Test={len(test_df)}")

    # --- Print Validation Summary Table ---
    stats_df = pd.DataFrame(fold_stats)
    print("\n--- FOLD BALANCE VALIDATION SUMMARY ---")
    print(stats_df.to_string(index=False))

    # Check the variance of key stats across test folds (lower is better)
    event_rate_var = stats_df['Test Event Rate'].var()
    psa_var = stats_df['Test Median PSA'].var()
    print("\nVariance across test folds (lower is better):")
    print(f"  - Event Rate Variance: {event_rate_var:.6f}")
    print(f"  - Median PSA Variance: {psa_var:.4f}")
    print("\n✅ Cross-validation splits created successfully.")
    print(f"Review the table above. The 'Test Median PSA' values should be very similar across all folds.")


def main():
    parser = argparse.ArgumentParser(description='Create CORRECTED k-fold splits for survival analysis.')
    parser.add_argument('--clin_dat_path', type=str, required=True, help='Path to directory with JSON clinical files')
    parser.add_argument('--feat_path', type=str, required=True, help='Path to directory with feature (.pt) files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for fold splits')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds')
    args = parser.parse_args()

    print("Loading clinical data...")
    clinical_df = load_clinical_data(args.clin_dat_path)
    if clinical_df.empty:
        print("Error: No valid clinical data found.")
        return

    print("Finding feature files...")
    feature_df = find_feature_files(args.feat_path, clinical_df['patient_id'].tolist())
    if feature_df.empty:
        print("Error: No feature files found for the given patients.")
        return

    print("Creating merged dataset...")
    merged_df = create_merged_dataset(clinical_df, feature_df)
    if merged_df.empty:
        print("Error: Merged dataset is empty.")
        return

    perform_cross_validation(merged_df, args.output_dir, args.n_splits)

if __name__ == "__main__":
    main()