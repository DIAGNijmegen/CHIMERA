import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

# Add the current directory and Aggregators to the path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "task1_baseline" / "Aggregators"))

try:
    from wsi_datasets.clinical_processor import ClinicalDataProcessor
    CLINICAL_PROCESSOR_AVAILABLE = True
except ImportError:
    print("Warning: ClinicalDataProcessor not found. Using fallback method.")
    CLINICAL_PROCESSOR_AVAILABLE = False
from sklearn.preprocessing import StandardScaler
import torch
import warnings
warnings.filterwarnings('ignore')

# Paths (edit if needed)
FOLDS_DIR = '/data/temporary/chimera/Baseline_models/Task1_ABMIL/folds2'
WSI_FEATURES_DIR = '/data/pa_cpgarchive/projects/chimera/_aws/task1/pathology/features/features'
MRI_FEATURES_DIR = '/data/pa_cpgarchive/projects/chimera/prostate/radiology_data/train_features_picai'
CLINICAL_DATA_DIR = '/data/pa_cpgarchive/projects/chimera/_aws/task1/clinical_data'

# Utility functions
def load_csv(path):
    return pd.read_csv(path, sep=None, engine='python')

def get_case_ids(df):
    return df['case_id'].astype(str).unique().tolist()

def get_slide_ids(df):
    return df['slide_id'].astype(str).unique().tolist()

def load_wsi_features_summary(slide_ids):
    """Load WSI features and compute summary statistics per slide"""
    features_summary = []
    missing_files = []
    
    for sid in slide_ids:
        feat_path = os.path.join(WSI_FEATURES_DIR, f'{sid}.pt')
        if os.path.exists(feat_path):
            try:
                features = torch.load(feat_path)
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                
                # Handle different shapes
                if len(features.shape) > 2:
                    features = features.squeeze()
                
                # Compute summary statistics
                summary = {
                    'slide_id': sid,
                    'n_patches': features.shape[0] if len(features.shape) > 1 else 1,
                    'feature_dim': features.shape[1] if len(features.shape) > 1 else features.shape[0],
                    'mean_feature': np.mean(features, axis=0 if len(features.shape) > 1 else None),
                    'std_feature': np.std(features, axis=0 if len(features.shape) > 1 else None),
                    'min_feature': np.min(features, axis=0 if len(features.shape) > 1 else None),
                    'max_feature': np.max(features, axis=0 if len(features.shape) > 1 else None)
                }
                features_summary.append(summary)
            except Exception as e:
                print(f"Error loading WSI features for {sid}: {e}")
                missing_files.append(sid)
        else:
            missing_files.append(sid)
    
    return features_summary, missing_files

def load_mri_features_summary(case_ids):
    """Load MRI features and compute summary statistics per case"""
    features_summary = []
    missing_files = []
    
    for cid in case_ids:
        feat_path = os.path.join(MRI_FEATURES_DIR, f'{cid}_0001_features.pt')
        if os.path.exists(feat_path):
            try:
                features = torch.load(feat_path)
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                
                # Ensure features are flattened for MRI
                if len(features.shape) > 1:
                    features = features.flatten()
                
                summary = {
                    'case_id': cid,
                    'feature_dim': len(features),
                    'mean_feature': np.mean(features),
                    'std_feature': np.std(features),
                    'min_feature': np.min(features),
                    'max_feature': np.max(features)
                }
                features_summary.append(summary)
            except Exception as e:
                print(f"Error loading MRI features for {cid}: {e}")
                missing_files.append(cid)
        else:
            missing_files.append(cid)
    
    return features_summary, missing_files

def load_clinical_features_processed(case_ids):
    """Load and process clinical data using the actual ClinicalDataProcessor"""
    try:
        if CLINICAL_PROCESSOR_AVAILABLE:
            clinical_data_dir = "/data/pa_cpgarchive/projects/chimera/_aws/task1/clinical_data"
            processor = ClinicalDataProcessor(clinical_data_dir)
            
            # First fit the processor on all case IDs
            print(f"    🔧 Fitting clinical processor on {len(case_ids)} cases...", flush=True)
            processor.fit(case_ids)
            
            # Then transform each case and collect the data
            clinical_data_list = []
            for case_id in case_ids:
                try:
                    # Get raw clinical data for analysis
                    raw_data = processor._load_clinical_data(case_id)
                    raw_data['case_id'] = case_id
                    clinical_data_list.append(raw_data)
                except Exception as e:
                    print(f"    ⚠️ Error loading clinical data for case {case_id}: {e}", flush=True)
                    continue
            
            if clinical_data_list:
                clinical_df = pd.DataFrame(clinical_data_list)
                print(f"    ✅ Loaded clinical data for {len(clinical_df)} cases using ClinicalDataProcessor", flush=True)
                return clinical_df
            else:
                print(f"    ⚠️ No clinical data could be loaded", flush=True)
                return None
        else:
            # Fallback method
            return load_clinical_features_fallback(case_ids)
            
    except Exception as e:
        print(f"    ❌ Error loading clinical data with processor: {e}", flush=True)
        return load_clinical_features_fallback(case_ids)


def load_clinical_features_fallback(case_ids):
    """Fallback method to load clinical data directly from files"""
    try:
        clinical_data_dir = "/data/pa_cpgarchive/projects/chimera/_aws/task1/clinical_data"
        
        # Try to load from JSON files first (individual case files)
        clinical_data_list = []
        json_files_found = 0
        
        print(f"    🔍 Checking for individual JSON files in {clinical_data_dir}...", flush=True)
        for case_id in case_ids:
            json_path = os.path.join(clinical_data_dir, f"{case_id}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    data['case_id'] = case_id
                    clinical_data_list.append(data)
                    json_files_found += 1
                except Exception as e:
                    print(f"    ⚠️ Error reading {json_path}: {e}", flush=True)
        
        if json_files_found > 0:
            clinical_df = pd.DataFrame(clinical_data_list)
            print(f"    ✅ Loaded clinical data from {json_files_found} JSON files", flush=True)
            return clinical_df
        
        # If no JSON files, try CSV files
        clinical_files = [
            os.path.join(clinical_data_dir, "clinical_data.csv"),
            os.path.join(clinical_data_dir, "processed_clinical_data.csv"),
            "/data/pa_cpgarchive/projects/chimera/clinical_data.csv"
        ]
        
        print(f"    🔍 Checking for CSV files...", flush=True)
        for clinical_file in clinical_files:
            if os.path.exists(clinical_file):
                print(f"    📄 Found CSV file: {clinical_file}", flush=True)
                try:
                    clinical_df = pd.read_csv(clinical_file)
                    
                    # Filter to requested case IDs
                    if 'case_id' in clinical_df.columns:
                        clinical_df = clinical_df[clinical_df['case_id'].astype(str).isin([str(cid) for cid in case_ids])]
                    elif 'PatientID' in clinical_df.columns:
                        clinical_df = clinical_df[clinical_df['PatientID'].astype(str).isin([str(cid) for cid in case_ids])]
                        clinical_df = clinical_df.rename(columns={'PatientID': 'case_id'})
                    else:
                        print(f"    ⚠️ No case_id or PatientID column found in {clinical_file}", flush=True)
                        continue
                    
                    print(f"    ✅ Loaded clinical data for {len(clinical_df)} cases from CSV", flush=True)
                    return clinical_df
                    
                except Exception as e:
                    print(f"    ❌ Error reading CSV file {clinical_file}: {e}", flush=True)
                    continue
        
        print(f"    ❌ No clinical data files found in expected locations", flush=True)
        return None
        
    except Exception as e:
        print(f"    ❌ Error in fallback clinical data loading: {e}", flush=True)
        return None

def compare_test_sets_survival(all_test_data):
    """Compare survival distributions across all test sets, highlighting fold 3"""
    plt.figure(figsize=(15, 10))
    
    # Colors for each fold, highlighting fold 3
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    fold_3_color = 'red'
    
    # Plot 1: Survival time distributions
    plt.subplot(2, 3, 1)
    for fold_idx, test_df in all_test_data.items():
        color = fold_3_color if fold_idx == 3 else colors[fold_idx % len(colors)]
        linewidth = 3 if fold_idx == 3 else 1.5
        alpha = 1.0 if fold_idx == 3 else 0.7
        sns.histplot(test_df['bcr_survival_months'], label=f'Fold {fold_idx}', 
                    alpha=alpha, bins=20, color=color, element="step", linewidth=linewidth)
    plt.title('Test Sets: Survival Time Distributions')
    plt.xlabel('BCR Survival (months)')
    plt.legend()
    
    # Plot 2: Censorship distributions
    plt.subplot(2, 3, 2)
    fold_indices = list(all_test_data.keys())
    event_rates = []
    censored_rates = []
    
    for fold_idx in fold_indices:
        test_df = all_test_data[fold_idx]
        event_rate = (test_df['bcr_censorship'] == 0).mean()
        censored_rate = (test_df['bcr_censorship'] == 1).mean()
        event_rates.append(event_rate)
        censored_rates.append(censored_rate)
    
    x = np.arange(len(fold_indices))
    width = 0.35
    colors_bar = [fold_3_color if i == 3 else 'lightblue' for i in fold_indices]
    
    plt.bar(x - width/2, event_rates, width, label='Events', color=colors_bar, alpha=0.8)
    plt.bar(x + width/2, censored_rates, width, label='Censored', color='lightgray', alpha=0.8)
    
    plt.title('Test Sets: Event vs Censored Rates')
    plt.xlabel('Fold')
    plt.ylabel('Rate')
    plt.xticks(x, [f'Fold {i}' for i in fold_indices])
    plt.legend()
    
    # Plot 3: Survival time boxplots
    plt.subplot(2, 3, 3)
    survival_data = []
    fold_labels = []
    fold_colors = []
    
    for fold_idx in fold_indices:
        test_df = all_test_data[fold_idx]
        survival_data.extend(test_df['bcr_survival_months'].tolist())
        fold_labels.extend([f'Fold {fold_idx}'] * len(test_df))
        fold_colors.extend([fold_3_color if fold_idx == 3 else 'lightblue'] * len(test_df))
    
    survival_df = pd.DataFrame({'survival': survival_data, 'fold': fold_labels})
    box_plot = sns.boxplot(data=survival_df, x='fold', y='survival')
    
    # Highlight fold 3 box - find the position of 'Fold 3' in the x-axis categories
    fold_categories = [f'Fold {fold_idx}' for fold_idx in fold_indices]
    if 'Fold 3' in fold_categories:
        fold_3_box_idx = fold_categories.index('Fold 3')
        if fold_3_box_idx < len(box_plot.artists):
            box_plot.artists[fold_3_box_idx].set_facecolor(fold_3_color)
            box_plot.artists[fold_3_box_idx].set_alpha(0.8)
    
    plt.title('Test Sets: Survival Time Distributions')
    plt.ylabel('BCR Survival (months)')
    plt.xticks(rotation=45)
    
    # Plot 4: Event time distributions (events only)
    plt.subplot(2, 3, 4)
    for fold_idx, test_df in all_test_data.items():
        events_only = test_df[test_df['bcr_censorship'] == 0]
        if len(events_only) > 0:
            color = fold_3_color if fold_idx == 3 else colors[fold_idx % len(colors)]
            linewidth = 3 if fold_idx == 3 else 1.5
            alpha = 1.0 if fold_idx == 3 else 0.7
            sns.histplot(events_only['bcr_survival_months'], label=f'Fold {fold_idx}', 
                        alpha=alpha, bins=15, color=color, element="step", linewidth=linewidth)
    plt.title('Test Sets: Event Time Distributions (Events Only)')
    plt.xlabel('BCR Survival (months)')
    plt.legend()
    
    # Plot 5: Summary statistics comparison
    plt.subplot(2, 3, 5)
    stats_data = []
    for fold_idx in fold_indices:
        test_df = all_test_data[fold_idx]
        stats = {
            'fold': fold_idx,
            'median_survival': test_df['bcr_survival_months'].median(),
            'mean_survival': test_df['bcr_survival_months'].mean(),
            'event_rate': (test_df['bcr_censorship'] == 0).mean(),
            'n_samples': len(test_df)
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Plot median survival
    colors_bar = [fold_3_color if i == 3 else 'lightblue' for i in fold_indices]
    plt.bar(range(len(fold_indices)), stats_df['median_survival'], color=colors_bar, alpha=0.8)
    plt.title('Test Sets: Median Survival Time')
    plt.xlabel('Fold')
    plt.ylabel('Median Survival (months)')
    plt.xticks(range(len(fold_indices)), [f'Fold {i}' for i in fold_indices])
    
    # Plot 6: Sample sizes
    plt.subplot(2, 3, 6)
    plt.bar(range(len(fold_indices)), stats_df['n_samples'], color=colors_bar, alpha=0.8)
    plt.title('Test Sets: Sample Sizes')
    plt.xlabel('Fold')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(fold_indices)), [f'Fold {i}' for i in fold_indices])
    
    plt.tight_layout()
    plt.savefig('test_sets_survival_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return stats_df

def analyze_clinical_data(train_clinical, test_clinical, fold_idx):
    """Analyze clinical data distributions"""
    if not train_clinical or not test_clinical:
        print(f"Fold {fold_idx}: No clinical data to analyze")
        return
    
    # Convert to DataFrame for easier analysis
    train_df = pd.DataFrame(train_clinical)
    test_df = pd.DataFrame(test_clinical)
    
    # Identify numeric columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print(f"Fold {fold_idx}: No numeric clinical features found")
        return
    
    # Plot distributions for each numeric feature
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i+1)
        
        try:
            if train_df[col].nunique() > 1 or test_df[col].nunique() > 1:
                sns.histplot(train_df[col], label='Train', alpha=0.6, stat='density')
                sns.histplot(test_df[col], label='Test', alpha=0.6, stat='density')
            else:
                # For constant values, just show bar plot
                train_val = train_df[col].iloc[0] if len(train_df) > 0 else 0
                test_val = test_df[col].iloc[0] if len(test_df) > 0 else 0
                plt.bar(['Train', 'Test'], [train_val, test_val])
                
            plt.title(f'{col}')
            plt.legend()
        except Exception as e:
            print(f"Could not plot {col}: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig(f'fold_{fold_idx}_clinical_distributions.png', dpi=150)
    plt.close()

def analyze_feature_summaries(train_summaries, test_summaries, feature_type, fold_idx):
    """Analyze summary statistics of WSI or MRI features"""
    if not train_summaries or not test_summaries:
        print(f"Fold {fold_idx}: No {feature_type} features to analyze")
        return
    
    # Convert summaries to DataFrames
    train_df = pd.DataFrame(train_summaries)
    test_df = pd.DataFrame(test_summaries)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Number of patches (for WSI) or feature dimension
    if feature_type == 'WSI':
        plt.subplot(2, 3, 1)
        sns.histplot(train_df['n_patches'], label='Train', alpha=0.6, bins=20)
        sns.histplot(test_df['n_patches'], label='Test', alpha=0.6, bins=20)
        plt.title(f'Fold {fold_idx}: {feature_type} - Number of Patches')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        sns.histplot(train_df['feature_dim'], label='Train', alpha=0.6, bins=20)
        sns.histplot(test_df['feature_dim'], label='Test', alpha=0.6, bins=20)
        plt.title(f'Fold {fold_idx}: {feature_type} - Feature Dimension')
        plt.legend()
    else:  # MRI
        plt.subplot(2, 3, 1)
        sns.histplot(train_df['feature_dim'], label='Train', alpha=0.6, bins=20)
        sns.histplot(test_df['feature_dim'], label='Test', alpha=0.6, bins=20)
        plt.title(f'Fold {fold_idx}: {feature_type} - Feature Dimension')
        plt.legend()
    
    # Plot mean feature statistics
    plt.subplot(2, 3, 3)
    train_means = np.array([np.mean(s) for s in train_df['mean_feature']])
    test_means = np.array([np.mean(s) for s in test_df['mean_feature']])
    sns.histplot(train_means, label='Train', alpha=0.6, bins=20)
    sns.histplot(test_means, label='Test', alpha=0.6, bins=20)
    plt.title(f'Fold {fold_idx}: {feature_type} - Mean Feature Values')
    plt.legend()
    
    # Plot std feature statistics
    plt.subplot(2, 3, 4)
    train_stds = np.array([np.mean(s) for s in train_df['std_feature']])
    test_stds = np.array([np.mean(s) for s in test_df['std_feature']])
    sns.histplot(train_stds, label='Train', alpha=0.6, bins=20)
    sns.histplot(test_stds, label='Test', alpha=0.6, bins=20)
    plt.title(f'Fold {fold_idx}: {feature_type} - Std Feature Values')
    plt.legend()
    
    # Plot min/max ranges
    plt.subplot(2, 3, 5)
    train_ranges = np.array([np.mean(mx) - np.mean(mn) for mx, mn in zip(train_df['max_feature'], train_df['min_feature'])])
    test_ranges = np.array([np.mean(mx) - np.mean(mn) for mx, mn in zip(test_df['max_feature'], test_df['min_feature'])])
    sns.histplot(train_ranges, label='Train', alpha=0.6, bins=20)
    sns.histplot(test_ranges, label='Test', alpha=0.6, bins=20)
    plt.title(f'Fold {fold_idx}: {feature_type} - Feature Ranges')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'fold_{fold_idx}_{feature_type.lower()}_feature_analysis.png', dpi=150)
    plt.close()

def analyze_fold(fold_dir, fold_idx):
    """Main function to analyze a single fold"""
    print(f"\n{'='*50}")
    print(f"ANALYZING FOLD {fold_idx}")
    print(f"{'='*50}")
    
    train_csv = os.path.join(fold_dir, 'train.csv')
    test_csv = os.path.join(fold_dir, 'test.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"Missing CSV files for fold {fold_idx}")
        return
    
    train_df = load_csv(train_csv)
    test_df = load_csv(test_csv)
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # 1. Basic analysis (survival comparison is now done at cross-fold level)
    print("Skipping individual fold analysis - focusing on cross-fold comparison...")
    
    # 2. Analyze clinical data
    print("Loading and analyzing clinical data...")
    train_clinical, train_clinical_processed, train_clinical_missing = load_clinical_features_processed(get_case_ids(train_df))
    test_clinical, test_clinical_processed, test_clinical_missing = load_clinical_features_processed(get_case_ids(test_df))
    
    print(f"Train clinical data: {len(train_clinical)} loaded, {len(train_clinical_missing)} missing")
    print(f"Test clinical data: {len(test_clinical)} loaded, {len(test_clinical_missing)} missing")
    
    analyze_clinical_data(train_clinical, test_clinical, fold_idx)
    
    # 3. Analyze WSI features
    print("Loading and analyzing WSI features...")
    train_wsi_summary, train_wsi_missing = load_wsi_features_summary(get_slide_ids(train_df))
    test_wsi_summary, test_wsi_missing = load_wsi_features_summary(get_slide_ids(test_df))
    
    print(f"Train WSI features: {len(train_wsi_summary)} loaded, {len(train_wsi_missing)} missing")
    print(f"Test WSI features: {len(test_wsi_summary)} loaded, {len(test_wsi_missing)} missing")
    
    analyze_feature_summaries(train_wsi_summary, test_wsi_summary, 'WSI', fold_idx)
    
    # 4. Analyze MRI features
    print("Loading and analyzing MRI features...")
    train_mri_summary, train_mri_missing = load_mri_features_summary(get_case_ids(train_df))
    test_mri_summary, test_mri_missing = load_mri_features_summary(get_case_ids(test_df))
    
    print(f"Train MRI features: {len(train_mri_summary)} loaded, {len(train_mri_missing)} missing")
    print(f"Test MRI features: {len(test_mri_summary)} loaded, {len(test_mri_missing)} missing")
    
    analyze_feature_summaries(train_mri_summary, test_mri_summary, 'MRI', fold_idx)
    
    # 5. Summary statistics
    print(f"\nFOLD {fold_idx} SUMMARY:")
    print(f"{'='*30}")
    print(f"Dataset sizes - Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Clinical data availability - Train: {len(train_clinical)}/{len(get_case_ids(train_df))}, Test: {len(test_clinical)}/{len(get_case_ids(test_df))}")
    print(f"WSI data availability - Train: {len(train_wsi_summary)}/{len(get_slide_ids(train_df))}, Test: {len(test_wsi_summary)}/{len(get_slide_ids(test_df))}")
    print(f"MRI data availability - Train: {len(train_mri_summary)}/{len(get_case_ids(train_df))}, Test: {len(test_mri_summary)}/{len(get_case_ids(test_df))}")
    
    # Survival statistics
    train_events = (train_df['bcr_censorship'] == 0).sum()
    test_events = (test_df['bcr_censorship'] == 0).sum()
    print(f"Event rates - Train: {train_events}/{len(train_df)} ({100*train_events/len(train_df):.1f}%), Test: {test_events}/{len(test_df)} ({100*test_events/len(test_df):.1f}%)")
    print(f"Median survival - Train: {train_df['bcr_survival_months'].median():.1f}, Test: {test_df['bcr_survival_months'].median():.1f}")
    
    return {
        'fold': fold_idx,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_event_rate': train_events/len(train_df),
        'test_event_rate': test_events/len(test_df),
        'train_median_survival': train_df['bcr_survival_months'].median(),
        'test_median_survival': test_df['bcr_survival_months'].median(),
        'clinical_availability_train': len(train_clinical)/len(get_case_ids(train_df)),
        'clinical_availability_test': len(test_clinical)/len(get_case_ids(test_df)),
        'wsi_availability_train': len(train_wsi_summary)/len(get_slide_ids(train_df)),
        'wsi_availability_test': len(test_wsi_summary)/len(get_slide_ids(test_df)),
        'mri_availability_train': len(train_mri_summary)/len(get_case_ids(train_df)),
        'mri_availability_test': len(test_mri_summary)/len(get_case_ids(test_df))
    }

def compare_test_sets_across_folds(all_test_data):
    """Compare test sets across all folds to identify anomalies"""
    print(f"\n{'='*60}")
    print("CROSS-FOLD TEST SET COMPARISON")
    print(f"{'='*60}")
    
    # Extract test set data for each fold
    test_survival_data = {}
    test_clinical_data = {}
    test_wsi_data = {}
    test_mri_data = {}
    
    for fold_idx, data in all_test_data.items():
        test_survival_data[fold_idx] = data['survival_df']
        test_clinical_data[fold_idx] = data['clinical_raw']
        test_wsi_data[fold_idx] = data['wsi_summary']
        test_mri_data[fold_idx] = data['mri_summary']
    
    # 1. Compare survival distributions across test sets
    compare_test_survival_across_folds(test_survival_data)
    
    # 2. Compare clinical feature distributions across test sets
    compare_test_clinical_across_folds(test_clinical_data)
    
    # 3. Compare WSI feature distributions across test sets
    compare_test_features_across_folds(test_wsi_data, 'WSI')
    
    # 4. Compare MRI feature distributions across test sets
    compare_test_features_across_folds(test_mri_data, 'MRI')
    
    # Note: Additional statistical analysis is performed in the main function

def compare_test_survival_across_folds(test_survival_data):
    """Compare survival characteristics across test sets"""
    plt.figure(figsize=(20, 12))
    
    # Collect data for all folds
    all_fold_data = []
    for fold_idx, df in test_survival_data.items():
        all_fold_data.append({
            'fold': fold_idx,
            'event_rate': (df['bcr_censorship'] == 0).mean(),
            'median_survival': df['bcr_survival_months'].median(),
            'mean_survival': df['bcr_survival_months'].mean(),
            'survival_std': df['bcr_survival_months'].std(),
            'n_samples': len(df)
        })
    
    summary_df = pd.DataFrame(all_fold_data)
    
    # Plot 1: Survival time distributions for all test sets
    plt.subplot(2, 4, 1)
    for fold_idx, df in test_survival_data.items():
        color = 'red' if fold_idx == 3 else 'blue'
        alpha = 0.8 if fold_idx == 3 else 0.5
        linewidth = 3 if fold_idx == 3 else 1
        plt.hist(df['bcr_survival_months'], bins=15, alpha=alpha, 
                label=f'Fold {fold_idx}', color=color, histtype='step', linewidth=linewidth)
    plt.title('Test Set Survival Time Distributions')
    plt.xlabel('BCR Survival (months)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Event rates comparison
    plt.subplot(2, 4, 2)
    colors = ['red' if i == 3 else 'blue' for i in summary_df['fold']]
    plt.bar(summary_df['fold'], summary_df['event_rate'], color=colors, alpha=0.7)
    plt.title('Test Set Event Rates by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Event Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Median survival comparison
    plt.subplot(2, 4, 3)
    plt.bar(summary_df['fold'], summary_df['median_survival'], color=colors, alpha=0.7)
    plt.title('Test Set Median Survival by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Median Survival (months)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Sample size comparison
    plt.subplot(2, 4, 4)
    plt.bar(summary_df['fold'], summary_df['n_samples'], color=colors, alpha=0.7)
    plt.title('Test Set Sample Sizes')
    plt.xlabel('Fold')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Survival variability
    plt.subplot(2, 4, 5)
    plt.bar(summary_df['fold'], summary_df['survival_std'], color=colors, alpha=0.7)
    plt.title('Test Set Survival Variability (Std)')
    plt.xlabel('Fold')
    plt.ylabel('Survival Std (months)')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Box plots for survival times
    plt.subplot(2, 4, 6)
    survival_data_list = []
    fold_labels = []
    for fold_idx, df in test_survival_data.items():
        survival_data_list.append(df['bcr_survival_months'])
        fold_labels.append(f'F{fold_idx}')
    
    box_colors = ['red' if i == 3 else 'blue' for i in range(len(fold_labels))]
    box_plot = plt.boxplot(survival_data_list, labels=fold_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.title('Test Set Survival Time Distributions')
    plt.ylabel('BCR Survival (months)')
    
    # Plot 7: Censorship patterns
    plt.subplot(2, 4, 7)
    censorship_data = []
    for fold_idx in summary_df['fold']:
        df = test_survival_data[fold_idx]
        censored_rate = (df['bcr_censorship'] == 1).mean()
        censorship_data.append(censored_rate)
    
    plt.bar(summary_df['fold'], censorship_data, color=colors, alpha=0.7)
    plt.title('Test Set Censorship Rates')
    plt.xlabel('Fold')
    plt.ylabel('Censorship Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Scatter plot of event rate vs median survival
    plt.subplot(2, 4, 8)
    plt.scatter(summary_df['event_rate'], summary_df['median_survival'], 
               c=colors, s=150, alpha=0.8, edgecolors='black')
    for i, fold in enumerate(summary_df['fold']):
        plt.annotate(f'F{fold}', 
                    (summary_df['event_rate'].iloc[i], summary_df['median_survival'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    plt.title('Event Rate vs Median Survival (Test Sets)')
    plt.xlabel('Event Rate')
    plt.ylabel('Median Survival (months)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_sets_survival_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return summary_df

def compare_test_clinical_across_folds(test_clinical_data):
    """Compare clinical features across test sets"""
    print("    📊 Analyzing clinical features...", flush=True)
    
    # Combine all clinical data and identify common features
    all_clinical_dfs = []
    for fold_idx, clinical_list in test_clinical_data.items():
        if clinical_list and len(clinical_list) > 0:
            if isinstance(clinical_list, list):
                df = pd.DataFrame(clinical_list)
            else:
                df = clinical_list.copy()
            df['fold'] = fold_idx
            all_clinical_dfs.append(df)
            print(f"    Fold {fold_idx}: {len(df)} clinical records with {len(df.columns)-1} features", flush=True)
    
    if not all_clinical_dfs:
        print("    ❌ No clinical data available for comparison", flush=True)
        return
    
    combined_df = pd.concat(all_clinical_dfs, ignore_index=True)
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'fold']
    
    print(f"    📈 Found {len(numeric_cols)} numeric clinical features for comparison", flush=True)
    
    if len(numeric_cols) == 0:
        print("    ⚠️ No numeric clinical features found for comparison", flush=True)
        # Still create a summary of categorical features
        categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != 'fold']
        
        if len(categorical_cols) > 0:
            print(f"    📊 Found {len(categorical_cols)} categorical features: {list(categorical_cols)}", flush=True)
            
            # Create summary table for categorical features
            cat_summary = []
            for feature in categorical_cols[:5]:  # Limit to first 5
                for fold_idx in sorted(combined_df['fold'].unique()):
                    fold_data = combined_df[combined_df['fold'] == fold_idx][feature].dropna()
                    if len(fold_data) > 0:
                        value_counts = fold_data.value_counts()
                        most_common = value_counts.index[0] if len(value_counts) > 0 else 'N/A'
                        cat_summary.append({
                            'fold': fold_idx,
                            'feature': feature,
                            'n_samples': len(fold_data),
                            'n_unique': fold_data.nunique(),
                            'most_common': most_common,
                            'most_common_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0
                        })
            
            if cat_summary:
                cat_df = pd.DataFrame(cat_summary)
                cat_df.to_csv('test_sets_clinical_categorical_stats.csv', index=False)
                print("    📁 Categorical feature statistics saved to test_sets_clinical_categorical_stats.csv", flush=True)
        
        return
    
    # Create comparison plots for numeric features
    n_features = min(6, len(numeric_cols))  # Limit to 6 features for visualization
    selected_features = numeric_cols[:n_features]
    
    print(f"    🎨 Creating plots for features: {selected_features}", flush=True)
    
    plt.figure(figsize=(20, 12))
    
    for i, feature in enumerate(selected_features):
        plt.subplot(2, 3, i+1)
        
        # Collect data for each fold
        fold_data_dict = {}
        for fold_idx in sorted(combined_df['fold'].unique()):
            fold_data = combined_df[combined_df['fold'] == fold_idx][feature].dropna()
            if len(fold_data) > 0:
                fold_data_dict[fold_idx] = fold_data
        
        # Plot distributions for each fold
        for fold_idx, fold_data in fold_data_dict.items():
            if fold_data.nunique() > 1:  # Only plot if there's variability
                color = 'red' if fold_idx == 3 else 'blue'
                alpha = 0.8 if fold_idx == 3 else 0.5
                linewidth = 3 if fold_idx == 3 else 1
                plt.hist(fold_data, bins=10, alpha=alpha, label=f'Fold {fold_idx}', 
                        color=color, histtype='step', linewidth=linewidth)
            else:
                # For constant values, show as vertical line
                constant_val = fold_data.iloc[0]
                color = 'red' if fold_idx == 3 else 'blue'
                alpha = 0.8 if fold_idx == 3 else 0.5
                plt.axvline(constant_val, color=color, alpha=alpha, linewidth=2, 
                           label=f'Fold {fold_idx} (const={constant_val:.2f})')
        
        plt.title(f'Test Sets: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        if i == 0:  # Only show legend on first plot
            plt.legend()
        
        # Add statistical info
        fold_3_data = fold_data_dict.get(3, pd.Series())
        other_folds_data = pd.concat([data for fold, data in fold_data_dict.items() if fold != 3])
        
        if len(fold_3_data) > 0 and len(other_folds_data) > 0:
            fold_3_mean = fold_3_data.mean()
            others_mean = other_folds_data.mean()
            plt.text(0.02, 0.98, f'F3: {fold_3_mean:.2f}\nOthers: {others_mean:.2f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('test_sets_clinical_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistical summary
    feature_stats = []
    for feature in selected_features:
        for fold_idx in sorted(combined_df['fold'].unique()):
            fold_data = combined_df[combined_df['fold'] == fold_idx][feature].dropna()
            if len(fold_data) > 0:
                feature_stats.append({
                    'fold': fold_idx,
                    'feature': feature,
                    'mean': fold_data.mean(),
                    'std': fold_data.std(),
                    'median': fold_data.median(),
                    'min': fold_data.min(),
                    'max': fold_data.max(),
                    'n_samples': len(fold_data)
                })
    
    stats_df = pd.DataFrame(feature_stats)
    stats_df.to_csv('test_sets_clinical_stats.csv', index=False)
    print("    📁 Clinical feature statistics saved to test_sets_clinical_stats.csv", flush=True)
    
    # Highlight differences for fold 3
    if 3 in combined_df['fold'].unique():
        print(f"\n    🎯 FOLD 3 vs OTHERS CLINICAL COMPARISON:", flush=True)
        for feature in selected_features:
            fold_3_stats = stats_df[(stats_df['fold'] == 3) & (stats_df['feature'] == feature)]
            others_stats = stats_df[(stats_df['fold'] != 3) & (stats_df['feature'] == feature)]
            
            if len(fold_3_stats) > 0 and len(others_stats) > 0:
                f3_mean = fold_3_stats['mean'].iloc[0]
                others_mean = others_stats['mean'].mean()
                diff_pct = ((f3_mean - others_mean) / others_mean * 100) if others_mean != 0 else 0
                print(f"      {feature}: F3={f3_mean:.3f}, Others={others_mean:.3f}, Diff={diff_pct:+.1f}%", flush=True)

def compare_test_features_across_folds(test_feature_data, feature_type):
    """Compare WSI or MRI features across test sets"""
    print(f"    📊 Analyzing {feature_type} features...", flush=True)
    
    if not any(test_feature_data.values()):
        print(f"    ❌ No {feature_type} data available for comparison", flush=True)
        return
    
    # Collect summary statistics for each fold
    fold_stats = []
    for fold_idx, features in test_feature_data.items():
        if features and len(features) > 0:
            df = pd.DataFrame(features)
            
            if feature_type == 'WSI':
                stats = {
                    'fold': fold_idx,
                    'n_slides': len(df),
                    'mean_patches': df['n_patches'].mean(),
                    'std_patches': df['n_patches'].std(),
                    'median_patches': df['n_patches'].median(),
                    'mean_feature_dim': df['feature_dim'].mean(),
                    'mean_of_means': np.mean([np.mean(x) for x in df['mean_feature']]),
                    'std_of_means': np.std([np.mean(x) for x in df['mean_feature']]),
                    'mean_of_stds': np.mean([np.mean(x) for x in df['std_feature']]),
                    'mean_of_ranges': np.mean([np.mean(mx) - np.mean(mn) for mx, mn in zip(df['max_feature'], df['min_feature'])])
                }
            else:  # MRI
                stats = {
                    'fold': fold_idx,
                    'n_cases': len(df),
                    'mean_feature_dim': df['feature_dim'].mean(),
                    'std_feature_dim': df['feature_dim'].std(),
                    'median_feature_dim': df['feature_dim'].median(),
                    'mean_of_means': df['mean_feature'].mean(),
                    'std_of_means': df['mean_feature'].std(),
                    'median_of_means': df['mean_feature'].median(),
                    'mean_of_stds': df['std_feature'].mean(),
                    'mean_of_ranges': np.mean(df['max_feature'] - df['min_feature'])
                }
            
            fold_stats.append(stats)
            print(f"    Fold {fold_idx}: {len(df)} {feature_type} samples processed", flush=True)
    
    if not fold_stats:
        print(f"    ❌ No valid {feature_type} statistics computed", flush=True)
        return
    
    stats_df = pd.DataFrame(fold_stats)
    
    # Create comparison plots
    plt.figure(figsize=(20, 15))
    
    colors = ['red' if i == 3 else 'blue' for i in stats_df['fold']]
    
    if feature_type == 'WSI':
        # WSI-specific plots
        plt.subplot(3, 4, 1)
        plt.bar(stats_df['fold'], stats_df['n_slides'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Number of {feature_type} Slides')
        plt.xlabel('Fold')
        plt.ylabel('Number of Slides')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 2)
        plt.bar(stats_df['fold'], stats_df['mean_patches'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Mean Patches per Slide')
        plt.xlabel('Fold')
        plt.ylabel('Mean Patches')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 3)
        plt.bar(stats_df['fold'], stats_df['median_patches'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Median Patches per Slide')
        plt.xlabel('Fold')
        plt.ylabel('Median Patches')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 4)
        plt.bar(stats_df['fold'], stats_df['std_patches'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Patches Variability (Std)')
        plt.xlabel('Fold')
        plt.ylabel('Std Patches')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 5)
        plt.bar(stats_df['fold'], stats_df['mean_feature_dim'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Mean Feature Dimension')
        plt.xlabel('Fold')
        plt.ylabel('Feature Dimension')
        plt.grid(True, alpha=0.3)
        
    else:  # MRI
        plt.subplot(3, 4, 1)
        plt.bar(stats_df['fold'], stats_df['n_cases'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Number of {feature_type} Cases')
        plt.xlabel('Fold')
        plt.ylabel('Number of Cases')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 2)
        plt.bar(stats_df['fold'], stats_df['mean_feature_dim'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Mean Feature Dimension')
        plt.xlabel('Fold')
        plt.ylabel('Feature Dimension')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 3)
        plt.bar(stats_df['fold'], stats_df['median_feature_dim'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Median Feature Dimension')
        plt.xlabel('Fold')
        plt.ylabel('Feature Dimension')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 4)
        plt.bar(stats_df['fold'], stats_df['std_feature_dim'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Feature Dimension Variability')
        plt.xlabel('Fold')
        plt.ylabel('Std of Feature Dimensions')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 5)
        plt.bar(stats_df['fold'], stats_df['median_of_means'], color=colors, alpha=0.7)
        plt.title(f'Test Set: Median of Feature Means')
        plt.xlabel('Fold')
        plt.ylabel('Median of Means')
        plt.grid(True, alpha=0.3)
    
    # Common plots for both WSI and MRI
    plt.subplot(3, 4, 6)
    plt.bar(stats_df['fold'], stats_df['mean_of_means'], color=colors, alpha=0.7)
    plt.title(f'Test Set: Mean of Feature Means')
    plt.xlabel('Fold')
    plt.ylabel('Mean of Means')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 7)
    plt.bar(stats_df['fold'], stats_df['std_of_means'], color=colors, alpha=0.7)
    plt.title(f'Test Set: Std of Feature Means')
    plt.xlabel('Fold')
    plt.ylabel('Std of Means')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 8)
    plt.bar(stats_df['fold'], stats_df['mean_of_stds'], color=colors, alpha=0.7)
    plt.title(f'Test Set: Mean of Feature Stds')
    plt.xlabel('Fold')
    plt.ylabel('Mean of Stds')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 9)
    plt.bar(stats_df['fold'], stats_df['mean_of_ranges'], color=colors, alpha=0.7)
    plt.title(f'Test Set: Mean of Feature Ranges')
    plt.xlabel('Fold')
    plt.ylabel('Mean of Ranges')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot comparing key metrics
    plt.subplot(3, 4, 10)
    x_metric = 'mean_of_means'
    y_metric = 'mean_of_stds'
    plt.scatter(stats_df[x_metric], stats_df[y_metric], c=colors, s=150, alpha=0.8, edgecolors='black')
    for i, fold in enumerate(stats_df['fold']):
        plt.annotate(f'F{fold}', 
                    (stats_df[x_metric].iloc[i], stats_df[y_metric].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    plt.title(f'{feature_type}: Mean vs Std of Features')
    plt.xlabel('Mean of Feature Means')
    plt.ylabel('Mean of Feature Stds')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'test_sets_{feature_type.lower()}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats_df.to_csv(f'test_sets_{feature_type.lower()}_stats.csv', index=False)
    print(f"    📁 {feature_type} feature statistics saved to test_sets_{feature_type.lower()}_stats.csv", flush=True)
    
    # Highlight differences for fold 3
    if 3 in stats_df['fold'].values:
        print(f"\n    🎯 FOLD 3 vs OTHERS {feature_type} COMPARISON:", flush=True)
        fold_3_stats = stats_df[stats_df['fold'] == 3].iloc[0]
        others_stats = stats_df[stats_df['fold'] != 3].mean()
        
        key_metrics = ['mean_of_means', 'std_of_means', 'mean_of_stds', 'mean_of_ranges']
        if feature_type == 'WSI':
            key_metrics.extend(['mean_patches', 'std_patches'])
        else:
            key_metrics.extend(['mean_feature_dim', 'std_feature_dim'])
        
        for metric in key_metrics:
            if metric in fold_3_stats.index and metric in others_stats.index:
                f3_val = fold_3_stats[metric]
                others_val = others_stats[metric]
                diff_pct = ((f3_val - others_val) / others_val * 100) if others_val != 0 else 0
                print(f"      {metric}: F3={f3_val:.6f}, Others={others_val:.6f}, Diff={diff_pct:+.1f}%", flush=True)

def perform_statistical_tests(all_test_data):
    """Perform statistical tests comparing fold 3 vs others"""
    fold_indices = list(all_test_data.keys())
    fold_3_data = all_test_data[3]
    other_folds_data = pd.concat([all_test_data[i] for i in fold_indices if i != 3])
    
    print("\n" + "="*50)
    print("STATISTICAL TESTS: FOLD 3 vs OTHER FOLDS")
    print("="*50)
    
    # Test survival differences
    from scipy import stats
    
    # Survival time comparison
    fold_3_survival = fold_3_data['bcr_survival_months'].dropna()
    others_survival = other_folds_data['bcr_survival_months'].dropna()
    
    t_stat, t_pval = stats.ttest_ind(fold_3_survival, others_survival)
    u_stat, u_pval = stats.mannwhitneyu(fold_3_survival, others_survival, alternative='two-sided')
    
    print(f"\nSURVIVAL TIME COMPARISON:")
    print(f"Fold 3 median: {fold_3_survival.median():.2f} months")
    print(f"Others median: {others_survival.median():.2f} months")
    print(f"T-test p-value: {t_pval:.4f}")
    print(f"Mann-Whitney U p-value: {u_pval:.4f}")
    
    # Event rate comparison
    fold_3_event_rate = (fold_3_data['bcr_censorship'] == 0).mean()
    others_event_rate = (other_folds_data['bcr_censorship'] == 0).mean()
    
    print(f"\nEVENT RATE COMPARISON:")
    print(f"Fold 3 event rate: {fold_3_event_rate:.3f}")
    print(f"Others event rate: {others_event_rate:.3f}")
    
    # Chi-square test for event rates
    fold_3_events = (fold_3_data['bcr_censorship'] == 0).sum()
    fold_3_censored = (fold_3_data['bcr_censorship'] == 1).sum()
    others_events = (other_folds_data['bcr_censorship'] == 0).sum()
    others_censored = (other_folds_data['bcr_censorship'] == 1).sum()
    
    contingency_table = [[fold_3_events, fold_3_censored], 
                        [others_events, others_censored]]
    chi2_stat, chi2_pval = stats.chi2_contingency(contingency_table)[:2]
    print(f"Chi-square test p-value: {chi2_pval:.4f}")
    
    return {
        'survival_t_test_pval': t_pval,
        'survival_mannwhitney_pval': u_pval,
        'event_rate_chi2_pval': chi2_pval,
        'fold_3_median_survival': fold_3_survival.median(),
        'others_median_survival': others_survival.median(),
        'fold_3_event_rate': fold_3_event_rate,
        'others_event_rate': others_event_rate
    }


def collect_all_test_data_simple():
    """Collect basic test data from all folds (survival data only)"""
    print("Loading test data from all folds...")
    print(f"Looking in: {FOLDS_DIR}")
    
    all_test_data = {}
    
    for fold_idx in range(5):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{fold_idx}")
        test_csv_path = os.path.join(fold_dir, "test.csv")
        
        print(f"\nFold {fold_idx}:")
        print(f"  Looking for: {test_csv_path}")
        
        if not os.path.exists(test_csv_path):
            print(f"  ❌ Test CSV not found for fold {fold_idx}")
            continue
            
        try:
            print(f"  ✅ Loading test data...")
            test_df = load_csv(test_csv_path)
            print(f"  📊 Loaded {len(test_df)} samples with columns: {list(test_df.columns)}")
            
            # Check for required columns
            if 'bcr_survival_months' not in test_df.columns:
                print(f"  ⚠️ Warning: 'bcr_survival_months' column not found")
            if 'bcr_censorship' not in test_df.columns:
                print(f"  ⚠️ Warning: 'bcr_censorship' column not found")
                
            all_test_data[fold_idx] = test_df
            
        except Exception as e:
            print(f"  ❌ Error loading test data for fold {fold_idx}: {e}")
            continue
    
    print(f"\n📈 Successfully loaded test data for {len(all_test_data)} folds")
    return all_test_data

def collect_all_test_features(all_test_data):
    """Collect all features (clinical, WSI, MRI) for cross-fold test set comparison"""
    print("🔄 Loading features for all test sets...", flush=True)
    
    all_features = {
        'clinical': {},
        'wsi': {},
        'mri': {}
    }
    
    for fold_idx, test_df in all_test_data.items():
        print(f"\n  Fold {fold_idx}:", flush=True)
        
        # Get case and slide IDs
        case_ids = get_case_ids(test_df)
        slide_ids = get_slide_ids(test_df)
        
        print(f"    📋 {len(case_ids)} cases, {len(slide_ids)} slides", flush=True)
        
        # Load clinical features
        try:
            print(f"    🏥 Loading clinical data...", flush=True)
            clinical_data = load_clinical_features_processed(case_ids)
            if clinical_data is not None and len(clinical_data) > 0:
                # Convert to list of dictionaries for consistency
                all_features['clinical'][fold_idx] = clinical_data.to_dict('records')
                print(f"    ✅ Loaded clinical data for {len(clinical_data)} cases", flush=True)
            else:
                all_features['clinical'][fold_idx] = []
                print(f"    ⚠️ No clinical data loaded", flush=True)
        except Exception as e:
            print(f"    ❌ Error loading clinical data: {e}", flush=True)
            all_features['clinical'][fold_idx] = []
        
        # Load WSI features
        try:
            print(f"    🔬 Loading WSI features...", flush=True)
            wsi_summary, wsi_missing = load_wsi_features_summary(slide_ids)
            all_features['wsi'][fold_idx] = wsi_summary
            print(f"    ✅ Loaded WSI features for {len(wsi_summary)} slides ({len(wsi_missing)} missing)", flush=True)
        except Exception as e:
            print(f"    ❌ Error loading WSI features: {e}", flush=True)
            all_features['wsi'][fold_idx] = []
        
        # Load MRI features
        try:
            print(f"    🧠 Loading MRI features...", flush=True)
            mri_summary, mri_missing = load_mri_features_summary(case_ids)
            all_features['mri'][fold_idx] = mri_summary
            print(f"    ✅ Loaded MRI features for {len(mri_summary)} cases ({len(mri_missing)} missing)", flush=True)
        except Exception as e:
            print(f"    ❌ Error loading MRI features: {e}", flush=True)
            all_features['mri'][fold_idx] = []
    
    # Summary
    print(f"\n📊 Feature loading summary:", flush=True)
    for feature_type, data in all_features.items():
        loaded_folds = [f for f, features in data.items() if len(features) > 0]
        print(f"  {feature_type.upper()}: Available in folds {loaded_folds}", flush=True)
    
    return all_features

def main():
    """Simplified main function with extensive debugging"""
    import sys
    import os
    
    # Force output flushing
    print("🔍 Cross-Fold Test Set Analysis", flush=True)
    print("="*60, flush=True)
    print(f"📍 Script started at: {os.getcwd()}", flush=True)
    print(f"🐍 Python version: {sys.version}", flush=True)
    print(f"📁 FOLDS_DIR: {FOLDS_DIR}", flush=True)
    print(f"📂 FOLDS_DIR exists: {os.path.exists(FOLDS_DIR)}", flush=True)
    
    if os.path.exists(FOLDS_DIR):
        try:
            contents = os.listdir(FOLDS_DIR)
            print(f"📋 FOLDS_DIR contents: {contents}", flush=True)
        except Exception as e:
            print(f"⚠️ Cannot list FOLDS_DIR: {e}", flush=True)
    
    try:
        print("\n🔄 Starting data loading...", flush=True)
        # Use simple data loading
        all_test_data = collect_all_test_data_simple()
        
        if not all_test_data:
            print("❌ No test data found. Check the FOLDS_DIR path.", flush=True)
            return
        
        print(f"\n✅ Successfully loaded {len(all_test_data)} folds", flush=True)
        
        # Basic survival comparison
        print("📊 Creating survival comparison plot...", flush=True)
        survival_stats = compare_test_sets_survival(all_test_data)
        print("📊 Saved: test_sets_survival_comparison.png", flush=True)
        
        # Enhanced analysis: Load all features for cross-fold comparison
        print("\n🔬 Loading features for cross-fold comparison...", flush=True)
        all_test_features = collect_all_test_features(all_test_data)
        
        # Compare clinical features across test sets
        if any(all_test_features['clinical'].values()):
            print("📈 Comparing clinical features across test sets...", flush=True)
            compare_test_clinical_across_folds(all_test_features['clinical'])
            print("📊 Saved: test_sets_clinical_comparison.png", flush=True)
        else:
            print("⚠️ No clinical data available for comparison", flush=True)
        
        # Compare WSI features across test sets
        if any(all_test_features['wsi'].values()):
            print("🔍 Comparing WSI features across test sets...", flush=True)
            compare_test_features_across_folds(all_test_features['wsi'], 'WSI')
            print("📊 Saved: test_sets_wsi_comparison.png", flush=True)
        else:
            print("⚠️ No WSI data available for comparison", flush=True)
        
        # Compare MRI features across test sets
        if any(all_test_features['mri'].values()):
            print("🧠 Comparing MRI features across test sets...", flush=True)
            compare_test_features_across_folds(all_test_features['mri'], 'MRI')
            print("📊 Saved: test_sets_mri_comparison.png", flush=True)
        else:
            print("⚠️ No MRI data available for comparison", flush=True)
        
        # Statistical tests if we have fold 3
        if 3 in all_test_data:
            print("🧪 Running statistical tests...", flush=True)
            test_results = perform_statistical_tests(all_test_data)
            print(f"\n🎯 Key finding: Fold 3 vs Others", flush=True)
            print(f"   Survival: {test_results['fold_3_median_survival']:.1f} vs {test_results['others_median_survival']:.1f} months", flush=True)
            print(f"   Events: {test_results['fold_3_event_rate']:.3f} vs {test_results['others_event_rate']:.3f}", flush=True)
        
        print(f"\n{'='*60}", flush=True)
        print("✅ ANALYSIS COMPLETE", flush=True)
        print(f"{'='*60}", flush=True)
        
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()


if __name__ == "__main__":
    print("🚀 Starting script...")
    main()
    print("🏁 Script finished.")
