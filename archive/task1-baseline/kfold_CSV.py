import os
import json
import pandas as pd
import glob
import argparse
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def load_clinical_data(clin_dat_path):
    """
    Load clinical data from JSON files for Task 1 (BCR prediction)
    """
    clinical_data = []
    json_files = glob.glob(os.path.join(clin_dat_path, "*.json"))
    
    print(f"Found {len(json_files)} JSON files in {clin_dat_path}")
    
    for json_file in json_files:
        patient_id = os.path.basename(json_file).replace('.json', '')
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract survival information
            bcr_months = data.get("time_to_follow-up/BCR")
            bcr_status = data.get("BCR")
            
            # Convert BCR status to censorship (0 = event occurred, 1 = censored)
            if bcr_status == "1.0":
                censorship = 0  # Event occurred (BCR happened)
            elif bcr_status == "0.0":
                censorship = 1  # Censored (no BCR)
            else:
                print(f"Warning: Unexpected BCR value '{bcr_status}' for patient {patient_id}")
                continue
            
            # Keep survival time in months
            if bcr_months is not None and bcr_months > 0:
                clinical_data.append({
                    'patient_id': patient_id,
                    'bcr_months': bcr_months,
                    'censorship': censorship
                })
            else:
                print(f"Warning: Invalid survival time for patient {patient_id}: {bcr_months} months")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return pd.DataFrame(clinical_data)

def find_feature_files(feat_path, patient_ids):
    """
    Find feature files for each patient
    Note: Multiple .pt files per patient will be automatically loaded as a bag during training
    """
    feature_data = []
    
    for patient_id in patient_ids:
        # Find all feature files for this patient
        pattern = os.path.join(feat_path, f"{patient_id}_*.pt")
        pt_files = glob.glob(pattern)
        
        if not pt_files:
            print(f"Warning: No feature files found for patient {patient_id}")
            continue
        
        # Use the first slide as representative - training will load all slides for this patient
        first_slide = pt_files[0]
        slide_id = os.path.basename(first_slide).replace('.pt', '')
        
        feature_data.append({
            'patient_id': patient_id,
            'slide_id': slide_id,
            'num_slides': len(pt_files)  # Track how many slides this patient has
        })
        
        if len(pt_files) > 1:
            print(f"Patient {patient_id} has {len(pt_files)} slides - will be loaded as bag during training")
    
    return pd.DataFrame(feature_data)

def create_merged_dataset(clinical_df, feature_df):
    """
    Merge clinical and feature data - create one row per patient
    """
    # Group feature files by patient (each patient may have multiple slides)
    patient_slide_counts = feature_df.groupby('patient_id').size()
    print(f"Slide counts per patient: min={patient_slide_counts.min()}, max={patient_slide_counts.max()}, mean={patient_slide_counts.mean():.1f}")
    
    # Create one row per patient with their first slide_id as representative
    # The actual training will load all slides for each patient automatically
    feature_df_unique = feature_df.groupby('patient_id').first().reset_index()
    
    # Merge clinical and feature data
    merged_df = feature_df_unique.merge(clinical_df, on='patient_id', how='inner')
    
    print(f"Total patients after merging: {len(merged_df)}")
    print(f"Unique patients: {merged_df['patient_id'].nunique()}")
    
    # Rename columns for consistency with survival training expectations
    merged_df = merged_df.rename(columns={
        'bcr_months': 'bcr_survival_months', 
        'censorship': 'bcr_censorship',
        'patient_id': 'case_id'  # Expected by WSI survival dataset
    })
    
    return merged_df

def perform_cross_validation_by_patient(merged_df: pd.DataFrame, output_dir: str, n_splits: int = 5) -> None:
    """
    Perform stratified k-fold cross-validation at patient level for survival data
    """
    # Since we now have one row per patient, we can work directly with merged_df
    patient_data = merged_df.copy()
    
    print(f"Total unique patients: {len(patient_data)}")
    print(f"Events: {(patient_data['bcr_censorship'] == 0).sum()}")
    print(f"Censored: {(patient_data['bcr_censorship'] == 1).sum()}")
    
    # Use BCR status for stratification to maintain event balance across folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_data, patient_data['bcr_censorship'])):
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Get train and test data
        train_df = patient_data.iloc[train_idx]
        test_df = patient_data.iloc[test_idx]
        
        # Prepare output columns - use case_id instead of patient_id for compatibility
        columns = ['slide_id', 'bcr_survival_months', 'bcr_censorship', 'case_id']
        
        # Save splits
        train_df[columns].to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        test_df[columns].to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
        
        print(f"Fold {fold}:")
        print(f"  Train: {len(train_df)} patients")
        print(f"  Test: {len(test_df)} patients")
        
        # Print event distribution for this fold
        train_events = (train_df['bcr_censorship'] == 0).sum()
        train_censored = (train_df['bcr_censorship'] == 1).sum()
        test_events = (test_df['bcr_censorship'] == 0).sum()
        test_censored = (test_df['bcr_censorship'] == 1).sum()
        
        print(f"  Train events: {train_events}, censored: {train_censored}")
        print(f"  Test events: {test_events}, censored: {test_censored}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Create k-fold splits for Task 1')
    parser.add_argument('--clin_dat_path', type=str, required=True,
                       help='Path to directory containing JSON clinical data files')
    parser.add_argument('--feat_path', type=str, required=True,
                       help='Path to directory containing feature (.pt) files')
    parser.add_argument('--output_dir', type=str, 
                       default='/Volumes/temporary/chimera/Baseline_models/Task1_ABMIL/splits_chimera',
                       help='Output directory for fold splits')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds (default: 5)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.clin_dat_path):
        print(f"Error: Clinical data path does not exist: {args.clin_dat_path}")
        return
    
    if not os.path.exists(args.feat_path):
        print(f"Error: Feature path does not exist: {args.feat_path}")
        return
    
    print("Loading clinical data...")
    clinical_df = load_clinical_data(args.clin_dat_path)
    
    if clinical_df.empty:
        print("Error: No valid clinical data found")
        return
    
    print(f"Loaded clinical data for {len(clinical_df)} patients")
    
    print("Finding feature files...")
    feature_df = find_feature_files(args.feat_path, clinical_df['patient_id'].tolist())
    
    if feature_df.empty:
        print("Error: No feature files found")
        return
    
    print(f"Found feature files for {feature_df['patient_id'].nunique()} patients")
    
    print("Creating merged dataset...")
    merged_df = create_merged_dataset(clinical_df, feature_df)
    
    print("Performing cross-validation...")
    perform_cross_validation_by_patient(merged_df, args.output_dir, args.n_splits)
    
    print("Done!")

if __name__ == "__main__":
    main()
