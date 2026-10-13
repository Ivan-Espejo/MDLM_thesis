#!/usr/bin/env python3
"""=============================================================================
Script: 04-fold_comparison.py
Author: Ivan Espejo

Description:
    Tests how the selected cohort's assignment to train or test relates to LOCO
    performance and compares prediction-score distributions for two chosen folds.

Inputs:
    --benchmark_root_dir and --split_data_dir: LOCO metrics and project splits.
    --best_fold_dir, --worst_fold_dir, and --labels_path: fold predictions and
    true labels; --cohort_of_interest and --metric select the comparison.

Outputs:
    Cohort-presence plot, best/worst-fold density plot, and summary table under
    --global_out_dir.
============================================================================="""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def generate_cohort_presence_roc_plot(
    benchmark_root_dir,
    split_data_dir,
    global_out_dir,
    cohort_of_interest="PRJNA1259137",
    metric="ROC_AUC",
    color_in_train="#D62728",    # Red
    color_in_test="#FFFFFF"      # White with black border
):
    os.makedirs(global_out_dir, exist_ok=True)
    print(f"\n[1/5] Aggregating LOCO metric files from: {benchmark_root_dir}")

    # Search for all generated run metric CSVs (handles both naming conventions)
    search_pattern = os.path.join(benchmark_root_dir, "run*", "metrics", "*metrics*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"[ERROR] No metric CSV files found matching: {search_pattern}")
        return

    all_runs_data = []
    for file in csv_files:
        run_name = file.split(os.sep)[-3]  # e.g., "run1"
        try:
            df = pd.read_csv(file)
            df['Run'] = run_name
            if 'F1_Score' not in df.columns and 'F_Max_Theoretical' in df.columns:
                df['F1_Score'] = df['F_Max_Theoretical']
            all_runs_data.append(df)
        except Exception as e:
            print(f"  [WARN] Could not read {file}: {e}")

    if not all_runs_data:
        print("[ERROR] No valid benchmark data could be parsed.")
        return

    master_df = pd.concat(all_runs_data, ignore_index=True)

    # ---------------------------------------------------------
    # 1. Standardize Nomenclature
    # ---------------------------------------------------------
    rename_map = {
        'Alpha Diversity Baseline (GLMNET)': 'Baseline (Alpha-Div.)',
        'Alpha Diversity (GLMNET)': 'Baseline (Alpha-Div.)',
        'Alpha (GLMNET)': 'Baseline (Alpha-Div.)',
        'Mikropml (GLMNET)': 'mikropml (GLMNET)',
        'Deepmicro (SVM)': 'DeepMicro (SVM)'
    }
    master_df['Model'] = master_df['Model'].replace(rename_map)

    target_models = [
        'Baseline (Alpha-Div.)',
        'mikropml (GLMNET)', 'mikropml (RF)', 'mikropml (SVM)',
        'DeepMicro (GLMNET)', 'DeepMicro (RF)', 'DeepMicro (SVM)',
        'MDLM', 'MGM'
    ]
    master_df = master_df[master_df['Model'].isin(target_models)]

    if metric not in master_df.columns:
        print(f"[ERROR] Metric '{metric}' not found in aggregated dataset.")
        return

    # ---------------------------------------------------------
    # 2. Determine Cohort Allocation (Train vs. Test)
    # ---------------------------------------------------------
    print(f"[2/5] Mapping '{cohort_of_interest}' presence across fold splits...")

    def cohort_in_training(run_name):
        train_path = os.path.join(split_data_dir, run_name, "train_projects.txt")
        if not os.path.exists(train_path):
            # Fallback if nested inside a split_data_loco subdirectory
            train_path = os.path.join(split_data_dir, "split_data_loco", run_name, "train_projects.txt")
            if not os.path.exists(train_path):
                print(f"  [WARN] Missing train_projects.txt for {run_name}")
                return None
        with open(train_path, "r") as fh:
            train_projects = fh.read().split()
        return cohort_of_interest in train_projects

    presence_map = {run: cohort_in_training(run) for run in master_df['Run'].unique()}
    master_df['CohortInTrain'] = master_df['Run'].map(presence_map)
    master_df = master_df.dropna(subset=['CohortInTrain'])

    master_df['PresenceLabel'] = master_df['CohortInTrain'].map(
        {True: f'{cohort_of_interest} in Training',
         False: f'{cohort_of_interest} in Testing'}
    )

    # ---------------------------------------------------------
    # 3. Generate Comparative Boxplot
    # ---------------------------------------------------------
    print(f"[3/5] Rendering Cohort-Presence Boxplot for {metric}...")

    box_palette = {
        'Baseline (Alpha-Div.)': '#969696',
        'mikropml (GLMNET)': '#9ecae1', 'mikropml (RF)': '#3182bd', 'mikropml (SVM)': '#6baed6',
        'DeepMicro (GLMNET)': '#fdae6b', 'DeepMicro (RF)': '#e6550d', 'DeepMicro (SVM)': '#fd8d3c',
        'MDLM': '#31a354', 'MGM': '#a1d99b',
    }

    plt.figure(figsize=(14, 9))
    median_order = master_df.groupby('Model')[metric].median().sort_values(ascending=False).index

    ax = sns.boxplot(
        data=master_df, x=metric, y='Model', hue='Model', legend=False,
        order=median_order, palette=box_palette,
        showfliers=False, width=0.6, boxprops=dict(alpha=0.6, edgecolor='black')
    )

    presence_palette = {
        f'{cohort_of_interest} in Training': color_in_train,
        f'{cohort_of_interest} in Testing': color_in_test,
    }

    sns.stripplot(
        data=master_df, x=metric, y='Model', hue='PresenceLabel',
        order=median_order,
        hue_order=[f'{cohort_of_interest} in Training', f'{cohort_of_interest} in Testing'],
        palette=presence_palette,
        dodge=False, size=8, alpha=0.95, linewidth=0.8, edgecolor='black'
    )

    # Formatting axes and threshold reference lines
    if metric == 'ROC_AUC':
        plt.axvline(x=0.5, color='#525252', linestyle='--', linewidth=2.5, alpha=0.7, zorder=0)
        plt.text(0.505, -0.4, 'Random Chance (0.5)', color='#525252', fontsize=11, fontweight='bold', va='center')
        plt.xlim(-0.05, 1.05)
        plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12, fontweight='bold')
        x_label_str = 'Receiver Operating Characteristic (ROC-AUC)'
    else:
        plt.xlim(-0.05, 1.05)
        plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12, fontweight='bold')
        x_label_str = metric.replace("_", " ")

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Legend 1: Model Families
    cat_handles = [
        mpatches.Patch(color='#a1d99b', label='DL Architectures (MDLM, MGM)'),
        mpatches.Patch(color='#fdae6b', label='Latent Embeddings (DeepMicro)'),
        mpatches.Patch(color='#9ecae1', label='Classical Baselines (mikropml)'),
        mpatches.Patch(color='#969696', label='Baseline (Alpha-Div.)'),
    ]
    leg_arch = ax.legend(
        handles=cat_handles, title='Model Families',
        loc='upper left', bbox_to_anchor=(0.02, 0.98),
        frameon=True, fontsize=11, title_fontsize=13,
        facecolor='white', framealpha=0.9
    )
    ax.add_artist(leg_arch)

    # Legend 2: Target Cohort Presence
    presence_handles = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=color_in_train,
                      markersize=11, markeredgecolor='black', markeredgewidth=0.8,
                      label=f'{cohort_of_interest} in training set'),
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=color_in_test,
                      markersize=11, markeredgecolor='black', markeredgewidth=0.8,
                      label=f'{cohort_of_interest} in test set'),
    ]
    leg_presence = ax.legend(
        handles=presence_handles, title=f'{cohort_of_interest} Allocation',
        loc='upper left', bbox_to_anchor=(0.02, 0.74),
        frameon=True, fontsize=11, title_fontsize=13,
        facecolor='white', framealpha=0.95
    )
    ax.add_artist(leg_presence)

    plt.title(f'Cross-Cohort {metric.replace("_", "-")} by {cohort_of_interest} Allocation', fontsize=17, fontweight='bold', pad=20)
    plt.xlabel(x_label_str, fontsize=14, fontweight='bold')
    plt.ylabel('', fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.yaxis.tick_right()
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_file = os.path.join(global_out_dir, f'roc_boxplot_by_{cohort_of_interest}_presence.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {out_file}")

    # ---------------------------------------------------------
    # 4. Summary Table
    # ---------------------------------------------------------
    summary = master_df.groupby(['Model', 'PresenceLabel'])[metric].median().unstack()
    print(f"\n[4/5] Median {metric} by Cohort Allocation:\n")
    print(summary.round(4))
    
    summary_path = os.path.join(global_out_dir, f'median_{metric.lower()}_by_{cohort_of_interest}.csv')
    summary.round(4).to_csv(summary_path)
    print(f"  [OK] Summary table saved to: {summary_path}")


# ---------------------------------------------------------
# Density Plot Extension 
# ---------------------------------------------------------
def empirical_threshold(y_prob, y_pred):
    probs_as_0 = y_prob[y_pred == 0]
    probs_as_1 = y_prob[y_pred == 1]
    if len(probs_as_0) > 0 and len(probs_as_1) > 0:
        return (probs_as_0.max() + probs_as_1.min()) / 2.0
    return 0.5

def load_fold_predictions(input_dir, labels_path):
    data = {}
    search_path = os.path.join(input_dir, '**', '*.csv')
    csv_files = glob.glob(search_path, recursive=True)
    
    for f in csv_files:
        filename = os.path.basename(f)
        if "Predictions_" not in filename and filename != "y_score.csv":
            continue
            
        model_name_raw = filename.replace("Predictions_", "").replace(".csv", "")
        
        rename_map = {'MIKROPML (RF)': 'mikropml (RF)', 'RF': 'mikropml (RF)', 'MDLM': 'MDLM', 'MGM': 'MGM', 'Y_SCORE': 'MGM'}
        std_name = rename_map.get(model_name_raw.upper(), model_name_raw)

        if std_name not in ["mikropml (RF)", "MGM", "MDLM"]:
            continue
            
        try:
            if std_name == "MGM":
                # MGM requires softmax conversion and merging with true labels
                scores_df = pd.read_csv(f, index_col=0)
                scores_df.columns = scores_df.columns.astype(str)
                labels_df = pd.read_csv(labels_path)
                merged_df = labels_df.merge(scores_df, left_on=labels_df.columns[0], right_index=True, how='inner')
                
                logits = merged_df[['0', '1']].values
                logits_stable = logits - np.max(logits, axis=1, keepdims=True)
                probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable), axis=1, keepdims=True)
                
                y_true = merged_df['Label'].values
                y_prob = probs[:, 1]
                y_pred = np.argmax(probs, axis=1)
                
            else:
                # RF and MDLM use standard string mapping
                df = pd.read_csv(f)
                y_true = df['Actual'].map({'Healthy': 0, 'IBD': 1}).values if df['Actual'].dtype == object else df['Actual'].values
                y_prob = df['IBD'].values
                y_pred = df['Predicted'].map({'Healthy': 0, 'IBD': 1}).values if df['Predicted'].dtype == object else df['Predicted'].values

            data[std_name] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
            
        except Exception as e:
            print(f"  [WARN] Could not process {f}: {e}")
            
    return data
    
def plot_best_worst_density(best_dir, worst_dir, labels_path, global_out_dir):
    print(f"\n[5/5] Rendering High/Low ROC Mixed Density Plot...")
    folds = [
        ("Highest ROC Fold", best_dir),
        ("Lowest ROC Fold", worst_dir)
    ]
    target_models = ["mikropml (RF)", "MGM", "MDLM"]
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), sharex=True)

    for r, (row_label, fold_dir) in enumerate(folds):
        data = load_fold_predictions(fold_dir, labels_path)
        for c, model in enumerate(target_models):
            ax = axes[r][c]
            if model not in data:
                ax.set_visible(False)
                continue
                
            d = data[model]
            y_true, y_prob, y_pred = d['y_true'], d['y_prob'], d['y_pred']
            thresh = empirical_threshold(y_prob, y_pred)

            sns.kdeplot(y_prob[y_true == 1], fill=True, color='#D55E00', alpha=0.4,
                        label='True IBD', ax=ax, bw_adjust=0.5, warn_singular=False)
            sns.kdeplot(y_prob[y_true == 0], fill=True, color='#009E73', alpha=0.4,
                        label='True Healthy', ax=ax, bw_adjust=0.5, warn_singular=False)
            
            ax.axvline(x=thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
            ax.text(thresh + 0.02, ax.get_ylim()[1] * 0.75, f"Cutoff: {thresh:.2f}",
                    color='red', fontweight='bold', fontsize=9)

            ax.set_xlim([-0.05, 1.05])
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(True, linestyle=':', alpha=0.6)

            if r == 0:
                ax.set_title(model, fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{row_label}\nDensity" if c == 0 else "", fontsize=10)
            ax.set_xlabel("Predicted Probability of IBD" if r == 1 else "", fontsize=10)
            
            if r == 0 and c == 0:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    os.makedirs(global_out_dir, exist_ok=True)
    out_path = os.path.join(global_out_dir, "highest_lowest_roc_density.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohort Presence Impact Analysis & Fold Density Plot.")
    
    parser.add_argument("--benchmark_root_dir", default="output/loco/benchmark", 
                        help="Directory containing run*/metrics/ CSV files")
    parser.add_argument("--split_data_dir", default="output/loco/split_data_loco", 
                        help="Directory containing run*/train_projects.txt")
    parser.add_argument("--global_out_dir", default="output/loco/benchmark/global_stability", 
                        help="Destination directory for output plots and tables")
    parser.add_argument("--cohort_of_interest", default="PRJNA1259137", 
                        help="Study/Project ID whose allocation determines dot color")
    parser.add_argument("--metric", default="ROC_AUC", 
                        help="Target performance metric (default: ROC_AUC)")
    parser.add_argument("--color_in_train", default="#D62728", 
                        help="Dot color when the cohort is in training")
    parser.add_argument("--color_in_test", default="#FFFFFF", 
                        help="Dot color when the cohort is in testing")
    
    # New Arguments for Density Plot
    parser.add_argument("--best_fold_dir", default="output/loco/results/run4", required=True, 
                        help="Path to the highest ROC fold predictions directory")
    parser.add_argument("--worst_fold_dir", default="output/loco/results/run10", required=True, 
                        help="Path to the lowest ROC fold predictions directory")
    parser.add_argument("--labels_path", default="data/input/labels.csv", 
                        help="Path to global labels.csv")

    args = parser.parse_args()

    generate_cohort_presence_roc_plot(
        benchmark_root_dir=args.benchmark_root_dir,
        split_data_dir=args.split_data_dir,
        global_out_dir=args.global_out_dir,
        cohort_of_interest=args.cohort_of_interest,
        metric=args.metric,
        color_in_train=args.color_in_train,
        color_in_test=args.color_in_test
    )

    plot_best_worst_density(
        best_dir=args.best_fold_dir,
        worst_dir=args.worst_fold_dir,
        labels_path=args.labels_path,
        global_out_dir=args.global_out_dir
    )