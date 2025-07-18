#!/usr/bin/env python3
"""
Simple Cross-Fold Test Set Comparison Script
Focuses on comparing test sets across folds to identify fold 3 anomalies
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Data paths
FOLDS_DIR = '/data/temporary/chimera/Baseline_models/Task1_ABMIL/folds2'

def load_test_data():
    """Load test data from all folds - basic survival data only"""
    print("Loading test data from all folds...")
    print(f"Looking in: {FOLDS_DIR}")
    
    all_test_data = {}
    
    for fold_idx in range(5):
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{fold_idx}")
        test_csv = os.path.join(fold_dir, "test.csv")
        
        print(f"\nFold {fold_idx}:")
        print(f"  Path: {test_csv}")
        
        if not os.path.exists(test_csv):
            print(f"  ❌ File not found")
            continue
        
        try:
            # Load the CSV
            df = pd.read_csv(test_csv)
            print(f"  ✅ Loaded {len(df)} samples")
            print(f"  📋 Columns: {list(df.columns)}")
            
            # Check for survival columns
            required_cols = ['bcr_survival_months', 'bcr_censorship']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ⚠️  Missing columns: {missing_cols}")
                continue
            
            # Basic validation
            survival_data = df['bcr_survival_months'].dropna()
            censorship_data = df['bcr_censorship'].dropna()
            
            print(f"  📊 Survival: mean={survival_data.mean():.1f}, median={survival_data.median():.1f}")
            print(f"  🎯 Events: {(censorship_data == 0).sum()}/{len(censorship_data)} ({(censorship_data == 0).mean()*100:.1f}%)")
            
            all_test_data[fold_idx] = df
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    return all_test_data


def compare_survival_across_folds(all_test_data):
    """Compare survival characteristics across test sets with fold 3 highlighted"""
    if len(all_test_data) < 2:
        print("Need at least 2 folds for comparison")
        return
    
    print(f"\n{'='*60}")
    print("SURVIVAL COMPARISON ACROSS TEST SETS")
    print(f"{'='*60}")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Test Set Survival Comparison Across Folds (Fold 3 in RED)', fontsize=16)
    
    # Collect summary stats
    fold_stats = []
    
    # Plot 1: Survival time distributions
    ax = axes[0, 0]
    for fold_idx, df in all_test_data.items():
        color = 'red' if fold_idx == 3 else 'lightblue'
        linewidth = 3 if fold_idx == 3 else 1.5
        alpha = 1.0 if fold_idx == 3 else 0.7
        
        survival_times = df['bcr_survival_months'].dropna()
        ax.hist(survival_times, bins=20, alpha=alpha, label=f'Fold {fold_idx}', 
                color=color, histtype='step', linewidth=linewidth, density=True)
        
        # Collect stats
        events = (df['bcr_censorship'] == 0).sum()
        fold_stats.append({
            'fold': fold_idx,
            'n_samples': len(df),
            'n_events': events,
            'event_rate': events / len(df),
            'median_survival': survival_times.median(),
            'mean_survival': survival_times.mean(),
            'std_survival': survival_times.std()
        })
    
    ax.set_title('Survival Time Distributions')
    ax.set_xlabel('BCR Survival (months)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert to DataFrame for easier plotting
    stats_df = pd.DataFrame(fold_stats)
    
    # Plot 2: Event rates
    ax = axes[0, 1]
    colors = ['red' if i == 3 else 'lightblue' for i in stats_df['fold']]
    bars = ax.bar(stats_df['fold'], stats_df['event_rate'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Event Rates by Fold')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Event Rate')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, stats_df['event_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Median survival times
    ax = axes[0, 2]
    bars = ax.bar(stats_df['fold'], stats_df['median_survival'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Median Survival Times')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Median Survival (months)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, stats_df['median_survival']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Sample sizes
    ax = axes[1, 0]
    bars = ax.bar(stats_df['fold'], stats_df['n_samples'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Test Set Sample Sizes')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Number of Samples')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Box plots
    ax = axes[1, 1]
    survival_data_list = []
    fold_labels = []
    box_colors = []
    
    for fold_idx in sorted(all_test_data.keys()):
        df = all_test_data[fold_idx]
        survival_data_list.append(df['bcr_survival_months'].dropna())
        fold_labels.append(f'Fold {fold_idx}')
        box_colors.append('red' if fold_idx == 3 else 'lightblue')
    
    box_plot = ax.boxplot(survival_data_list, labels=fold_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax.set_title('Survival Time Box Plots')
    ax.set_ylabel('BCR Survival (months)')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Event rate vs median survival scatter
    ax = axes[1, 2]
    scatter = ax.scatter(stats_df['event_rate'], stats_df['median_survival'], 
                        c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add fold labels
    for i, row in stats_df.iterrows():
        ax.annotate(f'F{row["fold"]}', 
                   (row['event_rate'], row['median_survival']),
                   xytext=(5, 5), textcoords='offset points', 
                   fontweight='bold', fontsize=12)
    
    ax.set_title('Event Rate vs Median Survival')
    ax.set_xlabel('Event Rate')
    ax.set_ylabel('Median Survival (months)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_sets_survival_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: test_sets_survival_comparison.png")
    plt.close()
    
    return stats_df


def perform_statistical_tests(all_test_data):
    """Statistical tests comparing fold 3 vs others"""
    if 3 not in all_test_data:
        print("Fold 3 not found in data")
        return
    
    print(f"\n{'='*60}")
    print("STATISTICAL TESTS: FOLD 3 vs OTHER FOLDS")
    print(f"{'='*60}")
    
    # Get fold 3 data
    fold_3_data = all_test_data[3]
    
    # Combine other folds
    other_folds_list = []
    for fold_idx, df in all_test_data.items():
        if fold_idx != 3:
            df_copy = df.copy()
            df_copy['source_fold'] = fold_idx
            other_folds_list.append(df_copy)
    
    if not other_folds_list:
        print("No other folds found for comparison")
        return
    
    other_folds_data = pd.concat(other_folds_list, ignore_index=True)
    
    # Survival time comparison
    fold_3_survival = fold_3_data['bcr_survival_months'].dropna()
    others_survival = other_folds_data['bcr_survival_months'].dropna()
    
    # Statistical tests
    t_stat, t_pval = stats.ttest_ind(fold_3_survival, others_survival)
    u_stat, u_pval = stats.mannwhitneyu(fold_3_survival, others_survival, alternative='two-sided')
    
    print(f"\n🔍 SURVIVAL TIME COMPARISON:")
    print(f"   Fold 3:     n={len(fold_3_survival)}, median={fold_3_survival.median():.2f}, mean={fold_3_survival.mean():.2f}")
    print(f"   Others:     n={len(others_survival)}, median={others_survival.median():.2f}, mean={others_survival.mean():.2f}")
    print(f"   T-test:     p-value = {t_pval:.4f}")
    print(f"   Mann-Whitney: p-value = {u_pval:.4f}")
    
    # Event rate comparison
    fold_3_events = (fold_3_data['bcr_censorship'] == 0).sum()
    fold_3_total = len(fold_3_data)
    fold_3_event_rate = fold_3_events / fold_3_total
    
    others_events = (other_folds_data['bcr_censorship'] == 0).sum()
    others_total = len(other_folds_data)
    others_event_rate = others_events / others_total
    
    print(f"\n🎯 EVENT RATE COMPARISON:")
    print(f"   Fold 3:     {fold_3_events}/{fold_3_total} = {fold_3_event_rate:.3f}")
    print(f"   Others:     {others_events}/{others_total} = {others_event_rate:.3f}")
    
    # Chi-square test
    contingency_table = [
        [fold_3_events, fold_3_total - fold_3_events],
        [others_events, others_total - others_events]
    ]
    
    chi2_stat, chi2_pval = stats.chi2_contingency(contingency_table)[:2]
    print(f"   Chi-square: p-value = {chi2_pval:.4f}")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    if u_pval < 0.05:
        print(f"   ⚠️  SIGNIFICANT difference in survival times (p={u_pval:.4f})")
    else:
        print(f"   ✅ No significant difference in survival times (p={u_pval:.4f})")
    
    if chi2_pval < 0.05:
        print(f"   ⚠️  SIGNIFICANT difference in event rates (p={chi2_pval:.4f})")
    else:
        print(f"   ✅ No significant difference in event rates (p={chi2_pval:.4f})")
    
    return {
        'fold_3_median_survival': fold_3_survival.median(),
        'others_median_survival': others_survival.median(),
        'fold_3_event_rate': fold_3_event_rate,
        'others_event_rate': others_event_rate,
        'survival_mannwhitney_pval': u_pval,
        'event_rate_chi2_pval': chi2_pval
    }


def main():
    """Main analysis function"""
    print("🔍 Cross-Fold Test Set Analysis")
    print("="*60)
    print("Goal: Identify what makes fold 3's test set different from others")
    print("="*60)
    
    try:
        # Load test data
        all_test_data = load_test_data()
        
        if not all_test_data:
            print("❌ No test data found. Check the FOLDS_DIR path.")
            return
        
        print(f"\n✅ Successfully loaded {len(all_test_data)} folds: {list(all_test_data.keys())}")
        
        # Compare survival across folds
        print(f"\n📊 Comparing survival distributions...")
        stats_df = compare_survival_across_folds(all_test_data)
        
        if stats_df is not None:
            # Save summary statistics
            stats_df.to_csv('fold_comparison_summary.csv', index=False)
            print("📄 Saved: fold_comparison_summary.csv")
            
            # Print summary table
            print(f"\n📋 SUMMARY TABLE:")
            print(stats_df.round(3).to_string(index=False))
        
        # Statistical tests
        test_results = perform_statistical_tests(all_test_data)
        
        print(f"\n{'='*60}")
        print("🎯 ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print("Generated files:")
        print("  - test_sets_survival_comparison.png")
        print("  - fold_comparison_summary.csv")
        print("\nCheck the plots to see how fold 3 (in RED) differs from others!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
