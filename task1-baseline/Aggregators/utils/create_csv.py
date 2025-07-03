#!/usr/bin/env python3
"""
Script to create dataset CSV for Task 1 (Prostate Cancer BCR Prediction)
Usage: python create_splits.py --clin_dat_path /path/to/json/files --feat_path /path/to/features --output_path /path/to/output
"""

import os
import json
import pandas as pd
import glob
import argparse
from pathlib import Path

def load_clinical_data(clin_dat_path):
    """
    Load clinical data from JSON files
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
            
            # Convert months to days (approximately)
            bcr_days = bcr_months * 30.44 if bcr_months is not None else None
            
            if bcr_days is not None and bcr_days > 0:
                clinical_data.append({
                    'patient_id': patient_id,
                    'bcr_months': bcr_months,
                    'bcr_days': bcr_days,
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
    """
    feature_data = []
    
    for patient_id in patient_ids:
        # Find all feature files for this patient
        pattern = os.path.join(feat_path, f"{patient_id}_*.pt")
        pt_files = glob.glob(pattern)
        
        if not pt_files:
            print(f"Warning: No feature files found for patient {patient_id}")
            continue
        
        # For now, we'll create one entry per slide
        for pt_file in pt_files:
            slide_id = os.path.basename(pt_file).replace('.pt', '')
            feature_data.append({
                'patient_id': patient_id,
                'slide_id': slide_id
            })
    
    return pd.DataFrame(feature_data)

def create_dataset(clinical_df, feature_df, output_path):
    """
    Create single dataset CSV file
    """
    # Merge clinical and feature data
    merged_df = feature_df.merge(clinical_df, on='patient_id', how='inner')
    
    print(f"Total samples after merging: {len(merged_df)}")
    print(f"Unique patients: {merged_df['patient_id'].nunique()}")
    
    # Prepare output columns
    columns = ['slide_id', 'bcr_survival_days', 'bcr_censorship', 'patient_id']
    
    # Rename columns for consistency
    merged_df = merged_df.rename(columns={'bcr_days': 'bcr_survival_days', 'censorship': 'bcr_censorship'})
    
    # Save dataset
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'task1_train_data.csv')
    merged_df[columns].to_csv(output_file, index=False)
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"Total: {len(merged_df)} slides from {merged_df['patient_id'].nunique()} patients")
    
    print(f"\nEvent Distribution:")
    events = (merged_df['bcr_censorship'] == 0).sum()
    censored = (merged_df['bcr_censorship'] == 1).sum()
    print(f"Events (BCR occurred): {events}")
    print(f"Censored (no BCR): {censored}")
    
    print(f"\nDataset saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create dataset CSV for Task 1')
    parser.add_argument('--clin_dat_path', type=str, required=True, 
                       help='Path to directory containing JSON clinical data files')
    parser.add_argument('--feat_path', type=str, required=True,
                       help='Path to directory containing feature (.pt) files')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for CSV dataset file')
    
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
    
    print("Creating dataset...")
    create_dataset(clinical_df, feature_df, args.output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
