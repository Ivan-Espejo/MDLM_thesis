#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def generate_learning_curves(benchmark_root_dir, global_out_dir):
    os.makedirs(global_out_dir, exist_ok=True)
    search_pattern = os.path.join(benchmark_root_dir, "**", "metrics", "benchmarking_metrics_*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print("[ERROR] No metrics found.")
        return
        
    all_runs = [pd.read_csv(f).assign(Run=os.path.basename(f).replace('benchmarking_metrics_', '').replace('.csv', '')) for f in csv_files]
    df = pd.concat(all_runs, ignore_index=True)

    rename_map = {
        'Alpha Diversity (GLMNET)': 'Baseline (Alpha-Div.)',
        'Alpha Diversity Baseline (GLMNET)': 'Baseline (Alpha-Div.)',
        'mikropml (GLMNET)': 'mikropml (GLMNET)',
        'DeepMicro (SVM)': 'DeepMicro (SVM)'
    }
    df['Model'] = df['Model'].replace(rename_map)
    
    # Extract the numeric fraction from the run string (e.g., "run10" -> 10)
    df['Training_Fraction'] = df['Run'].apply(lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    df = df.sort_values('Training_Fraction')

    # Enforce strict color consistency with previous boxplots
    box_palette = {
        'Baseline (Alpha-Div.)': '#969696',
        'mikropml (GLMNET)': '#9ecae1', 'mikropml (RF)': '#3182bd', 'mikropml (SVM)': '#6baed6',
        'DeepMicro (GLMNET)': '#fdae6b', 'DeepMicro (RF)': '#e6550d', 'DeepMicro (SVM)': '#fd8d3c',
        'MDLM': '#31a354', 'MGM': '#a1d99b'
    }

    metrics_to_plot = ['ROC_AUC', 'PR_AUC', 'MCC', 'Balanced_Accuracy']
    
    for metric in metrics_to_plot:
        if metric not in df.columns: 
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(data=df, x='Training_Fraction', y=metric, hue='Model', marker='o', lw=2, palette=box_palette, ax=ax)
        
        ax.set_xlim([0, 100])
        ax.set_ylim([0.0, 1.05] if metric != 'MCC' else [-0.3, 1.05])
        ax.set_xlabel('Training Data Fraction (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_title(f'Learning Curve: {metric.replace("_", " ")}', fontsize=14, fontweight='bold')
        
        # Move legend outside the plot to prevent data obstruction
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        
        fig.savefig(os.path.join(global_out_dir, f'learning_curve_{metric.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Generated learning curve: {metric}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root_dir", required=True)
    parser.add_argument("--global_out_dir", required=True)
    args = parser.parse_args()
    generate_learning_curves(args.benchmark_root_dir, args.global_out_dir)