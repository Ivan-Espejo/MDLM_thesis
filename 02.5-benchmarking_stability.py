#!/usr/bin/env python3
"""
=============================================================================
Script: 02.5-benchmarking_stability.py
Description:
    Aggregates master_benchmarking_metrics.csv from every run, calculates 
    cross-cohort stability box plots, and runs statistical significance tests 
    (including MCC and Balanced Accuracy).
=============================================================================
"""

import argparse
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import re
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def perform_statistical_analysis(master_df, global_out_dir, metrics_to_test=['ROC_AUC', 'F1_Score', 'MCC', 'Balanced_Accuracy']):
    print("\n[4/4] Performing Non-Parametric Statistical Analysis...")
    
    stat_report = []
    stat_report.append("============================================================================")
    stat_report.append("                   CROSS-COHORT STATISTICAL ANALYSIS                        ")
    stat_report.append("============================================================================\n")
    
    for metric in metrics_to_test:
        if metric not in master_df.columns:
            continue
            
        stat_report.append(f"--- METRIC: {metric} ---")
        
        try:
            pivot_df = master_df.pivot(index='Run', columns='Model', values=metric)
            pivot_df = pivot_df.fillna(0)
        except Exception as e:
            stat_report.append(f"Could not pivot data for {metric}. Error: {e}")
            continue
            
        pivot_df = pivot_df.dropna(axis=1, how='all')
        models = pivot_df.columns.tolist()
        
        if len(models) < 2:
            stat_report.append("Not enough models to perform statistical testing.\n")
            continue
            
        stat_report.append("\nSummary Statistics (Median [IQR]):")
        summary = []
        for model in models:
            median = pivot_df[model].median()
            iqr = pivot_df[model].quantile(0.75) - pivot_df[model].quantile(0.25)
            summary.append((model, median, iqr))
            
        summary.sort(key=lambda x: x[1], reverse=True)
        for model, median, iqr in summary:
            stat_report.append(f"  {model:25s}: {median:.4f} [{iqr:.4f}]")
            
        stat, p_friedman = friedmanchisquare(*[pivot_df[col].dropna() for col in models])
        stat_report.append(f"\nGlobal Friedman Test p-value: {p_friedman:.4e}")
        
        if p_friedman >= 0.05:
            stat_report.append("  -> Result: No significant difference between models overall.\n")
            continue
        
        stat_report.append("  -> Result: Significant difference detected. Proceeding to pairwise tests.\n")
        
        stat_report.append("Pairwise Wilcoxon Tests (FDR Corrected):")
        p_values = []
        comparisons = []
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]
                
                diff = pivot_df[model_a] - pivot_df[model_b]
                
                if np.all(diff == 0):
                    p_val = 1.0
                else:
                    _, p_val = wilcoxon(pivot_df[model_a], pivot_df[model_b], zero_method='zsplit')
                    
                p_values.append(p_val)
                comparisons.append(f"{model_a} vs {model_b}")
                
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        results = []
        for comp, p_raw, p_adj, rej in zip(comparisons, p_values, pvals_corrected, reject):
            sig_star = "*" if rej else " "
            results.append((comp, p_adj, sig_star))
            
        results.sort(key=lambda x: x[1])
        for comp, p_adj, sig_star in results:
            stat_report.append(f"  [{sig_star}] {comp:45s} | p(adj) = {p_adj:.4f}")
            
        stat_report.append("\n" + "-"*80 + "\n")
        
    report_path = os.path.join(global_out_dir, "statistical_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(stat_report))
        
    print(f"Statistical report saved to: {report_path}")


def generate_global_stability_plots(benchmark_root_dir, global_out_dir):
    print(f"\n[1/4] Scanning for completed runs in: {benchmark_root_dir}")
    os.makedirs(global_out_dir, exist_ok=True)
    
    search_pattern = os.path.join(benchmark_root_dir, "run*", "metrics", "master_benchmarking_metrics.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"ERROR: No master_benchmarking_metrics.csv files found matching {search_pattern}")
        return
        
    print(f"Found {len(csv_files)} completed runs. Aggregating data...")
    
    all_runs_data = []
    for file in csv_files:
        run_name = file.split(os.sep)[-3]
        try:
            df = pd.read_csv(file)
            df['Run'] = run_name  
            if 'F1_Score' not in df.columns and 'F_Max_Theoretical' in df.columns:
                df['F1_Score'] = df['F_Max_Theoretical']
            all_runs_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    master_df = pd.concat(all_runs_data, ignore_index=True)
    
    # ---------------------------------------------------------
    # NOMENCLATURE FIX (Intercepts and corrects raw CSV strings)
    # ---------------------------------------------------------
    
    # 1. Upgrade the Run names from "run1" to "Fold 1", "run2" to "Fold 2"
    master_df['Run'] = master_df['Run'].str.replace('run', 'Fold ', regex=False)

    # 2. Map the models to their highly professional names
    # 2. Map the models to their highly professional names
    rename_map = {
        'Alpha Diversity Baseline (GLMNET)': 'Baseline (Alpha-Div.)', # Catches newer CSVs
        'Alpha (GLMNET)': 'Baseline (Alpha-Div.)',                  # Catches older CSVs
        'Mikropml (GLMNET)': 'mikropml (GLMNET)',
        'Deepmicro (SVM)': 'DeepMicro (SVM)'
    }
    master_df['Model'] = master_df['Model'].replace(rename_map)
    
    print("[2/4] Structuring DataFrame for plotting...")
    
    # Strictly define the 9 models using the NEW corrected stylings
    target_models = [
        'Baseline (Alpha-Div.)',
        'mikropml (GLMNET)', 'mikropml (RF)', 'mikropml (SVM)', 
        'DeepMicro (GLMNET)', 'DeepMicro (RF)', 'DeepMicro (SVM)', 
        'MDLM', 'MGM'
    ]
    master_df = master_df[master_df['Model'].isin(target_models)]

    # Thematic Color Dictionary for the Boxes
    box_palette = {
        'Baseline (Alpha-Div.)': '#969696', 
        'mikropml (GLMNET)': '#9ecae1',  
        'mikropml (RF)': '#3182bd',      
        'mikropml (SVM)': '#6baed6',     
        'DeepMicro (GLMNET)': '#fdae6b', 
        'DeepMicro (RF)': '#e6550d',     
        'DeepMicro (SVM)': '#fd8d3c',    
        'MDLM': '#31a354',               
        'MGM': '#a1d99b'                 
    }

    metrics_to_plot = ['ROC_AUC', 'PR_AUC', 'F1_Score', 'Accuracy', 'Balanced_Accuracy', 'Sensitivity', 'Specificity', 'MCC']
    print(f"[3/4] Generating {len(metrics_to_plot)} global stability box plots...")
    
    # Custom Mathematical Sorter for the Runs (so Fold 2 comes before Fold 10)
    def extract_run_num(run_str):
        match = re.search(r'\d+', run_str)
        return int(match.group()) if match else 0
        
    unique_runs = sorted(master_df['Run'].unique(), key=extract_run_num)
    
    # Palette for the individual run dots
    run_palette = sns.color_palette("tab20", len(unique_runs))
    run_color_dict = dict(zip(unique_runs, run_palette))

    for metric in metrics_to_plot:
        if metric not in master_df.columns:
            continue
            
        # Wider canvas specifically for MCC to accommodate the external left legend
        if metric == 'MCC':
            plt.figure(figsize=(16, 9))
        else:
            plt.figure(figsize=(14, 9))
        
        # Sort models dynamically from Best to Worst median
        median_order = master_df.groupby('Model')[metric].median().sort_values(ascending=False).index
        
        # Draw the thematic Box Plot
        ax = sns.boxplot(
            data=master_df, x=metric, y='Model', 
            order=median_order, palette=box_palette, 
            showfliers=False, width=0.6, boxprops=dict(alpha=0.6, edgecolor='black')
        )
        
        # Overlay the individual Folds as colored dots
        sns.stripplot(
            data=master_df, x=metric, y='Model', hue='Run',
            order=median_order, palette=run_color_dict, hue_order=unique_runs,
            dodge=False, size=7, alpha=0.9, linewidth=0.5, edgecolor='#333333'
        )
        
        # ---------------------------------------------------------
        # CUSTOM VISUAL ADDITIONS
        # ---------------------------------------------------------
        
        if metric == 'ROC_AUC':
            plt.axvline(x=0.5, color='#525252', linestyle='--', linewidth=2.5, alpha=0.7, zorder=0)
            plt.text(0.505, -0.4, 'Random Chance (0.5)', color='#525252', fontsize=11, fontweight='bold', va='center')
            x_label_str = 'Receiver Operating Characteristic (ROC-AUC)'
            plt.xlim(-0.05, 1.05)
            plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12, fontweight='bold')
        elif metric == 'Balanced_Accuracy':
            plt.axvline(x=0.5, color='#525252', linestyle='--', linewidth=2.5, alpha=0.7, zorder=0)
            plt.text(0.505, -0.4, 'Random Chance (0.5)', color='#525252', fontsize=11, fontweight='bold', va='center')
            x_label_str = 'Balanced Accuracy'
            plt.xlim(-0.05, 1.05)
            plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12, fontweight='bold')
        elif metric == 'MCC':
            plt.axvline(x=0.0, color='#525252', linestyle='--', linewidth=2.5, alpha=0.7, zorder=0)
            plt.text(0.02, -0.4, 'Random Chance (0.0)', color='#525252', fontsize=11, fontweight='bold', va='center')
            x_label_str = 'Matthews Correlation Coefficient (MCC)'
            plt.xlim(-0.35, 1.05)
            plt.xticks(np.arange(-0.2, 1.1, 0.2), fontsize=12, fontweight='bold')
        else:
            x_label_str = metric.replace("_", " ")
            plt.xlim(-0.05, 1.05)
            plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12, fontweight='bold')
        
        # Strip the default legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
        # Construct Custom Legend (Model Architecture Categories)
        cat_handles = [
            mpatches.Patch(color='#a1d99b', label='DL Architectures (MDLM, MGM)'),
            mpatches.Patch(color='#fdae6b', label='Latent Embeddings (DeepMicro)'),
            mpatches.Patch(color='#9ecae1', label='Classical Baselines (mikropml)'),
            mpatches.Patch(color='#969696', label='Baseline (Alpha-Div.)')
        ]
        
        # Construct Second Custom Legend (The Sorted Folds)
        run_handles = [
            mlines.Line2D([], [], color='w', marker='o', markerfacecolor=run_color_dict[run],
                          markersize=9, markeredgecolor='#333333', markeredgewidth=0.5, label=run)
            for run in unique_runs
        ]
        
        # Conditional Legend Placement (External Left for MCC, Inside-Left for others)
        if metric == 'MCC':
            cat_handles = [
                mpatches.Patch(color='#a1d99b', label='DL Architectures'),
                mpatches.Patch(color='#fdae6b', label='Latent Embeddings'),
                mpatches.Patch(color='#9ecae1', label='Classical Baselines'),
                mpatches.Patch(color='#969696', label='Baseline (Alpha-Div.)')
            ]

            leg_arch = ax.legend(handles=cat_handles, title='Model Families',
                                 loc='upper left', bbox_to_anchor=(-0.42, 1.0),
                                 ncol=1, frameon=True, fontsize=11, title_fontsize=13, facecolor='white', framealpha=0.9)
            ax.add_artist(leg_arch) 
            
            leg_runs = ax.legend(handles=run_handles, title='Evaluation Folds',
                                 loc='upper left', bbox_to_anchor=(-0.42, 0.65),
                                 ncol=1, frameon=True, fontsize=11, title_fontsize=13, facecolor='white', framealpha=0.9)
            
            # Provide extra padding on the left for the external legends
            plt.subplots_adjust(left=0.32)
        else:
            leg_arch = ax.legend(handles=cat_handles, title='Model Families',
                                 loc='upper left', bbox_to_anchor=(0.02, 0.98),
                                 frameon=True, fontsize=11, title_fontsize=13, facecolor='white', framealpha=0.9)
            ax.add_artist(leg_arch) 
            
            leg_runs = ax.legend(handles=run_handles, title='Evaluation Folds',
                                 loc='upper left', bbox_to_anchor=(0.02, 0.74),
                                 ncol=1, frameon=True, fontsize=11, title_fontsize=13, facecolor='white', framealpha=0.9)
        
        # ---------------------------------------------------------
        
        # Aesthetic formatting
        plt.title(f'Cross-Cohort Stability Distribution: {metric.replace("_", " ")}', fontsize=17, fontweight='bold', pad=20)
        plt.xlabel(x_label_str, fontsize=14, fontweight='bold')
        plt.ylabel('', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Right-Aligned Y-Axis
        ax.yaxis.tick_right()
        plt.yticks(fontsize=12, fontweight='bold')
        
        # Prevent tight_layout from overriding our manual margin adjustment on MCC
        if metric != 'MCC':
            plt.tight_layout()
        
        out_path = os.path.join(global_out_dir, f'global_stability_boxplot_{metric}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Pass the filtered DataFrame to the statistical analysis
    try:
        perform_statistical_analysis(master_df, global_out_dir)
    except NameError:
        print("\nNote: perform_statistical_analysis function not found in this snippet. Make sure it is defined above.")
        
    print(f"\n=======================================================")
    print(f"[SUCCESS] Global stability benchmarking box plots complete.")
    print(f"=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates individual run metrics into global stability plots and runs statistical tests.")
    parser.add_argument("--benchmark_dir", type=str, default="output/results/benchmark", 
                        help="Root directory containing the run subdirectories")
    parser.add_argument("--outdir", type=str, default="output/results/benchmark/global", 
                        help="Directory to save the global line plots and statistical reports.")
    
    args = parser.parse_args()
    generate_global_stability_plots(args.benchmark_dir, args.outdir)