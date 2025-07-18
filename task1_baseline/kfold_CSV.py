import os
import json
import pandas as pd
import glob
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from pathlib import Path

def load_clinical_data(clin_dat_path):
    """
    Load clinical data from JSON files for Task 1 (BCR prediction)
    Enhanced to include prognostic variables for stratified splitting
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
            
            # Extract key prognostic variables for stratification
            # These were identified as the main drivers of the fold 3 distribution issue
            psa = data.get("pre_operative_PSA")
            clinical_t = data.get("clinical_T_stage") 
            isup_grade = data.get("ISUP")
            primary_gleason = data.get("primary_gleason")
            secondary_gleason = data.get("secondary_gleason")
            age = data.get("age_at_prostatectomy")
            
            # Keep survival time in months
            if bcr_months is not None and bcr_months > 0:
                clinical_record = {
                    'patient_id': patient_id,
                    'bcr_months': bcr_months,
                    'censorship': censorship,
                    # Prognostic variables for stratification
                    'psa': psa,
                    'clinical_t': clinical_t,
                    'isup_grade': isup_grade,
                    'primary_gleason': primary_gleason,
                    'secondary_gleason': secondary_gleason,
                    'age': age
                }
                clinical_data.append(clinical_record)
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
    Perform enhanced stratified k-fold cross-validation at patient level for survival data
    Uses composite stratification on PSA, ISUP grade, clinical T-stage, and BCR status
    to prevent the covariate shift issue observed in fold 3
    """
    # Since we now have one row per patient, we can work directly with merged_df
    patient_data = merged_df.copy()
    
    print(f"Total unique patients: {len(patient_data)}")
    print(f"Events: {(patient_data['bcr_censorship'] == 0).sum()}")
    print(f"Censored: {(patient_data['bcr_censorship'] == 1).sum()}")
    
    # Print clinical variable distributions
    print("\nClinical variable distributions:")
    for var in ['psa', 'clinical_t', 'isup_grade', 'age']:
        if var in patient_data.columns:
            values = pd.to_numeric(patient_data[var], errors='coerce').dropna()
            if len(values) > 0:
                print(f"{var}: median={values.median():.1f}, range=[{values.min():.1f}, {values.max():.1f}], missing={patient_data[var].isna().sum()}")
    
    # Create composite stratification variable
    print("\n" + "="*60)
    stratification_var = create_stratification_variable(patient_data)
    
    # Use the composite stratification variable for balanced splits
    print("\n" + "="*60)
    print("Performing stratified k-fold cross-validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)

    # Store fold statistics for validation
    fold_stats = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_data, stratification_var)):
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
        
        # Calculate detailed statistics for this fold
        train_events = (train_df['bcr_censorship'] == 0).sum()
        train_censored = (train_df['bcr_censorship'] == 1).sum()
        test_events = (test_df['bcr_censorship'] == 0).sum()
        test_censored = (test_df['bcr_censorship'] == 1).sum()
        
        # Calculate clinical statistics for validation
        fold_stat = {
            'fold': fold,
            'train_n': len(train_df),
            'test_n': len(test_df),
            'train_events': train_events,
            'train_censored': train_censored,
            'test_events': test_events,
            'test_censored': test_censored,
            'train_event_rate': train_events / len(train_df),
            'test_event_rate': test_events / len(test_df)
        }
        
        # Add clinical variable medians
        for var in ['psa', 'clinical_t', 'isup_grade', 'age']:
            if var in test_df.columns:
                train_values = pd.to_numeric(train_df[var], errors='coerce').dropna()
                test_values = pd.to_numeric(test_df[var], errors='coerce').dropna()
                
                if len(train_values) > 0:
                    fold_stat[f'train_{var}_median'] = train_values.median()
                if len(test_values) > 0:
                    fold_stat[f'test_{var}_median'] = test_values.median()
        
        fold_stats.append(fold_stat)
        
        print(f"Fold {fold}:")
        print(f"  Train: {len(train_df)} patients (events: {train_events}, censored: {train_censored}, event_rate: {train_events/len(train_df):.3f})")
        print(f"  Test:  {len(test_df)} patients (events: {test_events}, censored: {test_censored}, event_rate: {test_events/len(test_df):.3f})")
        
        # Print key clinical variables for this fold's test set
        for var in ['psa', 'clinical_t', 'isup_grade']:
            if var in test_df.columns:
                test_values = pd.to_numeric(test_df[var], errors='coerce').dropna()
                if len(test_values) > 0:
                    print(f"  Test {var} median: {test_values.median():.1f}")
        print()
    
    # Save fold statistics for analysis
    fold_stats_df = pd.DataFrame(fold_stats)
    fold_stats_df.to_csv(os.path.join(output_dir, 'fold_statistics.csv'), index=False)
    
    # Print summary comparison to validate balanced distributions
    print("="*60)
    print("FOLD BALANCE VALIDATION")
    print("="*60)
    
    print("Event rates across folds:")
    for _, row in fold_stats_df.iterrows():
        print(f"  Fold {row['fold']}: Train {row['train_event_rate']:.3f}, Test {row['test_event_rate']:.3f}")
    
    print(f"\nTest event rate variance: {fold_stats_df['test_event_rate'].var():.6f} (lower is better)")
    
    for var in ['psa', 'clinical_t', 'isup_grade']:
        test_col = f'test_{var}_median'
        if test_col in fold_stats_df.columns:
            values = fold_stats_df[test_col].dropna()
            if len(values) > 0:
                print(f"Test {var} median variance: {values.var():.3f} (lower is better)")
    
    print(f"\nFold statistics saved to: {os.path.join(output_dir, 'fold_statistics.csv')}")
    print("Review these statistics to ensure balanced distributions across folds.")

def create_stratification_variable(df):
    """
    Create a composite stratification variable to ensure balanced folds
    Based on the analysis, we need to stratify on PSA, clinical T-stage, and ISUP grade
    """
    print("Creating stratification variable for balanced fold distribution...")
    
    # Create a copy to work with
    df = df.copy()
    
    # Handle missing values and convert to numeric
    numeric_cols = ['psa', 'clinical_t', 'isup_grade', 'age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create risk stratification based on key prognostic factors
    strata = []
    
    for idx, row in df.iterrows():
        stratum_components = []
        
        # 1. PSA stratification (critical factor from analysis)
        psa = row.get('psa')
        if pd.notna(psa):
            if psa <= 6:
                stratum_components.append('PSA_low')
            elif psa <= 10:
                stratum_components.append('PSA_med')
            else:
                stratum_components.append('PSA_high')
        else:
            stratum_components.append('PSA_unk')
        
        # 2. ISUP grade stratification (critical factor from analysis)
        isup = row.get('isup_grade')
        if pd.notna(isup):
            if isup <= 2:
                stratum_components.append('ISUP_low')
            elif isup == 3:
                stratum_components.append('ISUP_med')
            else:
                stratum_components.append('ISUP_high')
        else:
            stratum_components.append('ISUP_unk')
        
        # 3. Clinical T-stage stratification (critical factor from analysis)
        clinical_t = row.get('clinical_t')
        if pd.notna(clinical_t):
            if clinical_t <= 2:
                stratum_components.append('T_early')
            else:
                stratum_components.append('T_advanced')
        else:
            stratum_components.append('T_unk')
        
        # 4. BCR status (for event balance)
        bcr_status = 'event' if row['bcr_censorship'] == 0 else 'censored'
        stratum_components.append(bcr_status)
        
        # Combine into composite stratum
        stratum = '_'.join(stratum_components)
        strata.append(stratum)
    
    df['stratum'] = strata
    
    # Check stratum distribution
    stratum_counts = df['stratum'].value_counts()
    print(f"Created {len(stratum_counts)} unique strata")
    print("Stratum distribution:")
    print(stratum_counts.head(10))
    
    # If we have too many small strata, simplify
    min_stratum_size = max(2, len(df) // 50)  # At least 2% of data per stratum
    small_strata = stratum_counts[stratum_counts < min_stratum_size].index
    
    if len(small_strata) > 0:
        print(f"Simplifying {len(small_strata)} small strata (< {min_stratum_size} patients)")
        
        # Simplify by removing less critical factors for small groups
        simplified_strata = []
        for idx, row in df.iterrows():
            if row['stratum'] in small_strata:
                # Use only the most critical factors for small groups
                components = []
                
                # Keep PSA and ISUP as most critical
                psa = row.get('psa')
                if pd.notna(psa):
                    components.append('PSA_high' if psa > 10 else 'PSA_low_med')
                else:
                    components.append('PSA_unk')
                
                isup = row.get('isup_grade')
                if pd.notna(isup):
                    components.append('ISUP_high' if isup >= 3 else 'ISUP_low_med')
                else:
                    components.append('ISUP_unk')
                
                # BCR status
                bcr_status = 'event' if row['bcr_censorship'] == 0 else 'censored'
                components.append(bcr_status)
                
                simplified_strata.append('_'.join(components))
            else:
                simplified_strata.append(row['stratum'])
        
        df['stratum'] = simplified_strata
        
        # Check final distribution
        final_counts = df['stratum'].value_counts()
        print(f"Final stratification: {len(final_counts)} strata")
        print("Final stratum distribution:")
        print(final_counts)
    
    return df['stratum']

def main():
    parser = argparse.ArgumentParser(description='Create k-fold splits for Task 1 with enhanced stratification')
    parser.add_argument('--clin_dat_path', type=str, required=True,
                       help='Path to directory containing JSON clinical data files')
    parser.add_argument('--feat_path', type=str, required=True,
                       help='Path to directory containing feature (.pt) files')
    parser.add_argument('--output_dir', type=str, 
                       default='/data/temporary/chimera/Baseline_models/Task1_ABMIL/splits_chimera2/',
                       help='Output directory for fold splits')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds (default: 5)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED K-FOLD CROSS-VALIDATION FOR SURVIVAL ANALYSIS")
    print("="*80)
    print("This version addresses the covariate shift issue in fold 3 by:")
    print("• Stratifying on PSA levels (key prognostic factor)")
    print("• Stratifying on ISUP grade (tumor aggressiveness)")
    print("• Stratifying on clinical T-stage (disease extent)")
    print("• Maintaining balanced event rates across folds")
    print("• Validating distributions across all folds")
    print("="*80)
    
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
    
    if merged_df.empty:
        print("Error: Merged dataset is empty")
        return
    
    print(f"Merged dataset columns: {list(merged_df.columns)}")
    print(f"Sample of merged data:")
    print(merged_df.head())
    
    print("Performing enhanced stratified cross-validation...")
    try:
        perform_cross_validation_by_patient(merged_df, args.output_dir, args.n_splits)
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("="*80)
    print("✅ ENHANCED CROSS-VALIDATION COMPLETE!")
    print("="*80)
    print("Next steps:")
    print("1. Review the fold_statistics.csv file to verify balanced distributions")
    print("2. Re-run your survival analysis with the new folds")
    print("3. Compare model performance across folds - they should be more consistent")
    print("4. The fold 3 covariate shift issue should be resolved!")
    print("="*80)

if __name__ == "__main__":
    main()
