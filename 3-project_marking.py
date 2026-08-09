#!/usr/bin/env python3
"""
=============================================================================
Script: 03-fold-comparison.py
Description:
    Generates a curated cross-fold density comparison for the Fold-Level Case
    Analysis. Plots a selected subset of models (columns) for the best- and
    worst-performing folds (rows), so each model can be read best-vs-worst
    vertically. Reuses the loading and empirical-threshold logic of
    02-benchmarking.py so panels match the full per-fold grids.
=============================================================================
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# --- Empirical threshold, copied from 02-benchmarking.py for consistency ---
def empirical_threshold(y_prob, y_pred):
    probs_as_0 = y_prob[y_pred == 0]
    probs_as_1 = y_prob[y_pred == 1]
    if len(probs_as_0) > 0 and len(probs_as_1) > 0:
        return (probs_as_0.max() + probs_as_1.min()) / 2.0
    return 0.5


# --- Loader, copied from 02-benchmarking.py (unchanged logic) ---
def load_standardized_data(input_dir, labels_path):
    master_data = {}
    for framework in ['alpha', 'mikropml', 'deepmicro']:
        search_path = os.path.join(input_dir, framework, 'Predictions_*.csv')
        for file_path in glob.glob(search_path):
            model_name = os.path.basename(file_path).replace('Predictions_', '').replace('.csv', '').upper()
            if framework == 'mikropml':
                fw_display = 'mikropml'
            elif framework == 'deepmicro':
                fw_display = 'DeepMicro'
            else:
                fw_display = 'Alpha Diversity Baseline'
            full_name = f"{fw_display} ({model_name})"
            try:
                df = pd.read_csv(file_path)
                if df['Actual'].dtype == object:
                    y_true = df['Actual'].map({'Healthy': 0, 'IBD': 1}).values
                else:
                    y_true = df['Actual'].values
                y_prob = df['IBD'].values
                if df['Predicted'].dtype == object:
                    y_pred = df['Predicted'].map({'Healthy': 0, 'IBD': 1}).values
                else:
                    y_pred = df['Predicted'].values
                master_data[full_name] = {'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred}
            except Exception as e:
                print(f"  [WARN] Failed to load {file_path}: {e}")

    mdlm_path = os.path.join(input_dir, 'mdlm', 'sample_predictions.csv')
    if os.path.exists(mdlm_path):
        try:
            df_mdlm = pd.read_csv(mdlm_path)
            master_data['MDLM'] = {
                'y_true': df_mdlm['disease_tag'].values,
                'y_prob': df_mdlm['raw_probability'].values,
                'y_pred': df_mdlm['prediction_disease'].values,
            }
        except Exception as e:
            print(f"  [WARN] Failed to load MDLM: {e}")

    mgm_path = os.path.join(input_dir, 'mgm', 'y_score.csv')
    if os.path.exists(mgm_path) and os.path.exists(labels_path):
        try:
            scores_df = pd.read_csv(mgm_path, index_col=0)
            scores_df.columns = scores_df.columns.astype(str)
            labels_df = pd.read_csv(labels_path)
            merged_df = labels_df.merge(scores_df, left_on=labels_df.columns[0], right_index=True, how='inner')
            logits = merged_df[['0', '1']].values
            logits_stable = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable), axis=1, keepdims=True)
            master_data['MGM'] = {
                'y_true': merged_df['Label'].values,
                'y_prob': probs[:, 1],
                'y_pred': np.argmax(probs, axis=1),
            }
        except Exception as e:
            print(f"  [WARN] Failed to load MGM: {e}")

    return master_data


def plot_fold_comparison(folds, models, out_path):
    """
    folds : list of (row_label, input_dir, labels_path), top row first.
    models: list of model keys (columns), left to right.
    """
    n_rows, n_cols = len(folds), len(models)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.6 * n_rows),
                             squeeze=False)

    for r, (row_label, input_dir, labels_path) in enumerate(folds):
        data = load_standardized_data(input_dir, labels_path)
        for c, model in enumerate(models):
            ax = axes[r][c]
            if model not in data:
                ax.set_visible(False)
                print(f"  [WARN] {model} missing in {input_dir}")
                continue
            d = data[model]
            y_true, y_prob, y_pred = d['y_true'], d['y_prob'], d['y_pred']
            thresh = empirical_threshold(y_prob, y_pred)

            sns.kdeplot(y_prob[y_true == 1], fill=True, color='#D55E00', alpha=0.4,
                        label='True IBD', ax=ax, bw_adjust=0.5, warn_singular=False)
            sns.kdeplot(y_prob[y_true == 0], fill=True, color='#009E73', alpha=0.4,
                        label='True Healthy', ax=ax, bw_adjust=0.5, warn_singular=False)
            ax.axvline(x=thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
            ax.text(thresh + 0.015, ax.get_ylim()[1] * 0.7, f"Cutoff: {thresh:.2f}",
                    color='red', fontweight='bold', fontsize=9)

            ax.set_xlim([-0.05, 1.05])
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(True, linestyle=':', alpha=0.6)

            # Column titles only on the top row
            if r == 0:
                ax.set_title(model, fontsize=12, fontweight='bold')
            # Row label only on the leftmost column
            ax.set_ylabel(f"{row_label}\nDensity" if c == 0 else "", fontsize=10)
            # X-label only on the bottom row
            ax.set_xlabel("Predicted Probability of IBD" if r == n_rows - 1 else "", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def generate_cohort_presence_roc_plot(
    benchmark_root_dir,
    split_data_dir,
    global_out_dir,
    cohort_of_interest="PRJNA1259137",
    metric="ROC_AUC",
    color_in_train="#D62728",     # red
    color_in_test="#FFFFFF",      # white
):
    """
    Reproduces the cross-cohort ROC-AUC boxplot but recolours each fold's dot
    by whether `cohort_of_interest` was in that fold's TRAINING set (red) or
    its TEST set (white), both with black edges. Fold membership is read from
    split_data_dir/runX/train_projects.txt so the plot regenerates correctly
    if the splits change.
    """
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    os.makedirs(global_out_dir, exist_ok=True)

    # --- 1. Aggregate run metrics (same logic as generate_global_stability_plots) ---
    search_pattern = os.path.join(benchmark_root_dir, "run*", "metrics",
                                  "master_benchmarking_metrics.csv")
    csv_files = glob.glob(search_pattern)
    if not csv_files:
        print(f"ERROR: No metrics found matching {search_pattern}")
        return

    all_runs_data = []
    for file in csv_files:
        run_name = file.split(os.sep)[-3]   # "run1"
        try:
            df = pd.read_csv(file)
            df['Run'] = run_name
            if 'F1_Score' not in df.columns and 'F_Max_Theoretical' in df.columns:
                df['F1_Score'] = df['F_Max_Theoretical']
            all_runs_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    master_df = pd.concat(all_runs_data, ignore_index=True)

    # --- 2. Same nomenclature fixes ---
    rename_map = {
        'Alpha Diversity Baseline (GLMNET)': 'Baseline (Alpha-Div.)',
        'Alpha (GLMNET)': 'Baseline (Alpha-Div.)',
        'Mikropml (GLMNET)': 'mikropml (GLMNET)',
        'Deepmicro (SVM)': 'DeepMicro (SVM)',
    }
    master_df['Model'] = master_df['Model'].replace(rename_map)

    target_models = [
        'Baseline (Alpha-Div.)',
        'mikropml (GLMNET)', 'mikropml (RF)', 'mikropml (SVM)',
        'DeepMicro (GLMNET)', 'DeepMicro (RF)', 'DeepMicro (SVM)',
        'MDLM', 'MGM'
    ]
    master_df = master_df[master_df['Model'].isin(target_models)]

    # --- 3. Determine cohort presence per run from the split files ---
    def cohort_in_training(run_name):
        """run_name like 'run1'. Returns True if cohort is in that fold's train set."""
        train_path = os.path.join(split_data_dir, run_name, "train_projects.txt")
        if not os.path.exists(train_path):
            print(f"  [WARN] Missing split file: {train_path}")
            return None
        with open(train_path) as fh:
            train_projects = fh.read().split()   # whitespace-separated
        return cohort_of_interest in train_projects

    # master_df['Run'] is still "run1"..."run10" at this point
    presence_map = {run: cohort_in_training(run) for run in master_df['Run'].unique()}
    master_df['CohortInTrain'] = master_df['Run'].map(presence_map)

    # Drop any run whose split file was missing (presence == None)
    n_missing = master_df['CohortInTrain'].isna().sum()
    if n_missing:
        print(f"  [WARN] Dropping {n_missing} rows with unresolved cohort presence.")
        master_df = master_df.dropna(subset=['CohortInTrain'])

    master_df['PresenceLabel'] = master_df['CohortInTrain'].map(
        {True: f'{cohort_of_interest} in Training',
         False: f'{cohort_of_interest} in Testing'}
    )

    # --- 4. Plot ---
    box_palette = {
        'Baseline (Alpha-Div.)': '#969696',
        'mikropml (GLMNET)': '#9ecae1', 'mikropml (RF)': '#3182bd', 'mikropml (SVM)': '#6baed6',
        'DeepMicro (GLMNET)': '#fdae6b', 'DeepMicro (RF)': '#e6550d', 'DeepMicro (SVM)': '#fd8d3c',
        'MDLM': '#31a354', 'MGM': '#a1d99b',
    }

    if metric not in master_df.columns:
        print(f"ERROR: metric {metric} not in data.")
        return

    plt.figure(figsize=(14, 9))
    median_order = master_df.groupby('Model')[metric].median().sort_values(ascending=False).index

    ax = sns.boxplot(
        data=master_df, x=metric, y='Model',
        order=median_order, palette=box_palette,
        showfliers=False, width=0.6, boxprops=dict(alpha=0.6, edgecolor='black')
    )

    # Two-state coloured dots: red = in-train, white = in-test, black edge on both
    presence_palette = {
        f'{cohort_of_interest} in Training': color_in_train,
        f'{cohort_of_interest} in Testing':  color_in_test,
    }
    sns.stripplot(
        data=master_df, x=metric, y='Model', hue='PresenceLabel',
        order=median_order,
        hue_order=[f'{cohort_of_interest} in Training',
                   f'{cohort_of_interest} in Testing'],
        palette=presence_palette,
        dodge=False, size=8, alpha=0.95, linewidth=0.8, edgecolor='black'
    )

    if metric == 'ROC_AUC':
        plt.axvline(x=0.5, color='#525252', linestyle='--', linewidth=2.5, alpha=0.7, zorder=0)
        plt.text(0.505, -0.4, 'Random Chance (0.5)', color='#525252',
                 fontsize=11, fontweight='bold', va='center')
        plt.xlim(-0.05, 1.05)
        plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12, fontweight='bold')
        x_label_str = 'Receiver Operating Characteristic (ROC-AUC)'
    else:
        plt.xlim(-0.05, 1.05)
        plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12, fontweight='bold')
        x_label_str = metric.replace("_", " ")

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # --- Legend 1: Model Families (top, matching the other boxplots) ---
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

    # --- Legend 2: cohort allocation (below the families legend) ---
    presence_handles = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=color_in_train,
                      markersize=11, markeredgecolor='black', markeredgewidth=0.8,
                      label=f'{cohort_of_interest} in training set'),
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=color_in_test,
                      markersize=11, markeredgecolor='black', markeredgewidth=0.8,
                      label=f'{cohort_of_interest} in test set'),
    ]
    leg_presence = ax.legend(
        handles=presence_handles, title=f'{cohort_of_interest} allocation',
        loc='upper left', bbox_to_anchor=(0.02, 0.74),
        frameon=True, fontsize=11, title_fontsize=13,
        facecolor='white', framealpha=0.95
    )
    ax.add_artist(leg_presence)

    plt.title(f'Cross-Cohort ROC-AUC by {cohort_of_interest} Allocation',
              fontsize=17, fontweight='bold', pad=20)
    plt.xlabel(x_label_str, fontsize=14, fontweight='bold')
    plt.ylabel('', fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.yaxis.tick_right()
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(global_out_dir,
                            f'roc_boxplot_by_{cohort_of_interest}_presence.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out_path}")

    # --- 5. Optional: quick per-group summary you can quote in the text ---
    summary = (master_df.groupby(['Model', 'PresenceLabel'])[metric]
               .median().unstack())
    print("\nMedian", metric, "by cohort allocation:\n", summary.round(4))
    return master_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curated cross-fold density comparison and cohort-presence ROC plot.")

    # --- Density comparison args ---
    parser.add_argument("--best_dir",     default="output/results/predictions_with_alpha/run3")
    parser.add_argument("--worst_dir",    default="output/results/predictions_with_alpha/run10")
    parser.add_argument("--best_labels",  default="input/labels.csv")
    parser.add_argument("--worst_labels", default="input/labels.csv")
    parser.add_argument("-o", "--out",    default="output/results/benchmark_with_alpha/global/fold_density_comparison.png")

    # --- Cohort-presence ROC plot args ---
    parser.add_argument("--benchmark_root_dir", default="output/results/benchmark_with_alpha",
                        help="Directory containing run*/metrics/master_benchmarking_metrics.csv")
    parser.add_argument("--split_data_dir", default="output/split_data",
                        help="Directory containing run*/train_projects.txt")
    parser.add_argument("--global_out_dir", default="output/results/benchmark_with_alpha/global",
                        help="Output directory for the global cohort-presence plot")
    parser.add_argument("--cohort_of_interest", default="PRJNA1259137",
                        help="Project ID whose train/test allocation colours the fold dots")
    parser.add_argument("--presence_metric", default="ROC_AUC",
                        help="Metric to plot (must be a column in the metrics CSVs)")
    parser.add_argument("--color_in_train", default="#D62728",
                        help="Dot colour when the cohort is in the training set")
    parser.add_argument("--color_in_test", default="#FFFFFF",
                        help="Dot colour when the cohort is in the test set")

    args = parser.parse_args()

    # --- Density comparison ---
    MODELS = ["mikropml (RF)", "MDLM", "MGM"]
    FOLDS = [
        ("Fold 3 (best-case)",   args.best_dir,  args.best_labels),
        ("Fold 10 (worst-case)", args.worst_dir, args.worst_labels),
    ]
    plot_fold_comparison(FOLDS, MODELS, args.out)

    # --- Cohort-presence ROC plot ---
    generate_cohort_presence_roc_plot(
        benchmark_root_dir=args.benchmark_root_dir,
        split_data_dir=args.split_data_dir,
        global_out_dir=args.global_out_dir,
        cohort_of_interest=args.cohort_of_interest,
        metric=args.presence_metric,
        color_in_train=args.color_in_train,
        color_in_test=args.color_in_test,
    )