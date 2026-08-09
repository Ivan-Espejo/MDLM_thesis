#!/usr/bin/env python3
"""
=============================================================================
Script: 02-benchmarking.py
Description:
    A universal benchmarking script designed to evaluate and compare multiple 
    machine learning models (MGM, MDLM, mikropml, DeepMicro, and Alpha) from a single 
    standardized input directory using the true empirical thresholds and binary 
    decisions (y_pred) utilized by the models, including the Matthews Correlation 
    Coefficient (MCC) and Balanced Accuracy.
=============================================================================
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, balanced_accuracy_score
)

def calculate_actual_metrics(y_true, y_prob, y_pred):
    """Calculates operational metrics including MCC and Balanced Accuracy based strictly on the model's actual y_pred decisions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    f_max = np.max(f1_scores)
    
    probs_as_0 = y_prob[y_pred == 0]
    probs_as_1 = y_prob[y_pred == 1]
    if len(probs_as_0) > 0 and len(probs_as_1) > 0:
        actual_threshold = (probs_as_0.max() + probs_as_1.min()) / 2.0
    else:
        actual_threshold = 0.5
        
    return accuracy, balanced_acc, sensitivity, specificity, f_max, actual_threshold, mcc


def load_standardized_data(input_dir, labels_path):
    """Crawls the input directory and standardizes all model predictions including y_pred."""
    master_data = {}
    print(f"Scanning directory: {input_dir}")
    
    # 1. Mikropml, DeepMicro & Alpha (Format: SampleID, Actual, Predicted, Healthy, IBD)
    for framework in ['alpha', 'mikropml', 'deepmicro']:
        search_path = os.path.join(input_dir, framework, 'Predictions_*.csv')
        for file_path in glob.glob(search_path):
            model_name = os.path.basename(file_path).replace('Predictions_', '').replace('.csv', '').upper()
            
            # ---------------------------------------------------------
            # OFFICIAL NOMENCLATURE FORMATTING
            # ---------------------------------------------------------
            if framework == 'mikropml':
                fw_display = 'mikropml'
            elif framework == 'deepmicro':
                fw_display = 'DeepMicro'
            else:
                fw_display = 'Alpha Diversity Baseline'
                
            full_name = f"{fw_display} ({model_name})"
            # ---------------------------------------------------------

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
                print(f"  [OK] Loaded: {full_name}")
            except Exception as e:
                print(f"  [WARN] Failed to load {file_path}: {e}")

    # 2. MDLM
    mdlm_path = os.path.join(input_dir, 'mdlm', 'sample_predictions.csv')
    if os.path.exists(mdlm_path):
        try:
            df_mdlm = pd.read_csv(mdlm_path) 
            y_true = df_mdlm['disease_tag'].values
            y_prob = df_mdlm['raw_probability'].values
            y_pred = df_mdlm['prediction_disease'].values
            
            master_data['MDLM'] = {'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred}
            print("  [OK] Loaded: MDLM")
        except Exception as e:
            print(f"  [WARN] Failed to load MDLM: {e}")

    # 3. MGM
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
            
            y_true = merged_df['Label'].values
            y_prob = probs[:, 1]
            y_pred = np.argmax(probs, axis=1)
            
            master_data['MGM'] = {'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred}
            print("  [OK] Loaded: MGM")
        except Exception as e:
            print(f"  [WARN] Failed to load MGM: {e}")
            
    return master_data


def run_benchmarking(input_dir, labels_path, out_dir):
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'metrics'), exist_ok=True)
    
    # Extract raw folder name (e.g., "run1")
    run_name = os.path.basename(os.path.normpath(input_dir))
    
    # Format for professional plot titles (e.g., "run1" -> "Fold 1")
    display_name = run_name.replace('run', 'Fold ')
    
    master_data = load_standardized_data(input_dir, labels_path)
    if not master_data:
        print("[ERROR] No valid prediction files found. Check your input paths.")
        return

    order_hierarchy = ['alpha', 'mikropml', 'deepmicro', 'mdlm', 'mgm']
    results_list = []

    print("\nCalculating metrics and generating curves...")
    for model_name, data in master_data.items():
        y_true, y_prob, y_pred = data['y_true'], data['y_prob'], data['y_pred']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        acc, b_acc, sn, sp, f_max, actual_thresh, mcc = calculate_actual_metrics(y_true, y_prob, y_pred)
        
        operational_f1 = f1_score(y_true, y_pred, zero_division=0) 
        
        base_family = model_name.split()[0].lower()
        sort_index = order_hierarchy.index(base_family) if base_family in order_hierarchy else 99
        
        results_list.append({
            'Model': model_name,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'F_Max_Theoretical': f_max,
            'F1_Score': operational_f1,
            'MCC': mcc,
            'Accuracy': acc,
            'Balanced_Accuracy': b_acc,
            'Sensitivity': sn,
            'Specificity': sp,
            'Operational_Threshold': actual_thresh,
            '_sort_idx': sort_index,
            '_tpr': tpr,
            '_fpr': fpr,
            '_precision': precision,
            '_recall': recall
        })

    results_df = pd.DataFrame(results_list).sort_values(by=['_sort_idx', 'Model'])
    export_df = results_df.drop(columns=['_sort_idx', '_tpr', '_fpr', '_precision', '_recall'])
    export_df.to_csv(os.path.join(out_dir, 'metrics', 'master_benchmarking_metrics.csv'), index=False)

    colors = sns.color_palette("husl", len(results_df))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for idx, row in results_df.reset_index(drop=True).iterrows():
        ax_roc.plot(row['_fpr'], row['_tpr'], color=colors[idx], lw=2, label=f"{row['Model']} (AUC = {row['ROC_AUC']:.3f})")
        ax_pr.plot(row['_recall'], row['_precision'], color=colors[idx], lw=2, label=f"{row['Model']} (AUC = {row['PR_AUC']:.3f})")

    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax_roc.set_title(f'Comparative ROC Curves - {display_name}', fontsize=13, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(out_dir, 'figures', 'comparative_roc_curves.png'), dpi=300)
    plt.close(fig_roc)

    first_model = list(master_data.keys())[0]
    baseline = sum(master_data[first_model]['y_true']) / len(master_data[first_model]['y_true'])
    ax_pr.plot([0, 1], [baseline, baseline], color='gray', lw=2, linestyle='--', label=f'Baseline ({baseline:.2f})')
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax_pr.set_ylabel('Precision', fontsize=11)
    ax_pr.set_title(f'Comparative Precision-Recall Curves - {display_name}', fontsize=13, fontweight='bold')
    ax_pr.legend(loc="lower left", fontsize=9)
    ax_pr.grid(alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(out_dir, 'figures', 'comparative_pr_curves.png'), dpi=300)
    plt.close(fig_pr)

    metrics_to_plot = ['ROC_AUC', 'Accuracy', 'Balanced_Accuracy', 'Sensitivity', 'Specificity', 'F1_Score', 'MCC']
    melted_df = export_df.melt(id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    plt.gca().set_axisbelow(True)
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    plt.yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sns.barplot(data=melted_df, x='Model', y='Score', hue='Metric', palette='viridis', order=export_df['Model'])
    plt.ylim(-0.25, 1.1)
    plt.title(f'Metrics Across Frameworks - {display_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Score (-0.2 to 1.0)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figures', 'comparative_metrics_barplot.png'), dpi=300)
    plt.close()

    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(7, 6))
    for model_name, data in master_data.items():
        y_true, y_prob = data['y_true'], data['y_prob']
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy='quantile')
        plt.plot(prob_pred, prob_true, marker='s', markersize=6, linewidth=2.0, label=model_name, alpha=0.85)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1.5, label='Perfect Calibration')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Average Predicted Probability of IBD', fontsize=11)
    plt.ylabel('Observed Real-World Fraction of IBD', fontsize=11)
    plt.title(f'Multi-Model Calibration Curves - {display_name}', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figures', 'multi_model_calibration_curve.png'), dpi=300)
    plt.close()
    
    models = export_df['Model'].values
    num_models = len(models)
    
    cols = 3
    rows = (num_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < num_models:
            model_name = models[idx]
            data = master_data[model_name]
            y_true, y_prob = data['y_true'], data['y_prob']
            thresh = export_df.loc[export_df['Model'] == model_name, 'Operational_Threshold'].values[0]
            
            sns.kdeplot(y_prob[y_true == 1], fill=True, color='#D55E00', alpha=0.4, label='True IBD', ax=ax, bw_adjust=0.5, warn_singular=False)
            sns.kdeplot(y_prob[y_true == 0], fill=True, color='#009E73', alpha=0.4, label='True Healthy', ax=ax, bw_adjust=0.5, warn_singular=False)
            
            ax.axvline(x=thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
            ax.text(thresh + 0.015, ax.get_ylim()[1] * 0.7, f"Cutoff: {thresh:.2f}", color='red', fontweight='bold', fontsize=9)
            
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=10)
            ax.set_xlabel('Predicted Probability of IBD', fontsize=10)
            
            ax.set_xlim([-0.05, 1.05])
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if idx == 0:
                ax.legend(loc='upper right')
        else:
            ax.set_visible(False)
            
    fig.suptitle(f'Probability Density Distributions by Model - {display_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figures', 'multi_model_probability_densities.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== BENCHMARKING COMPLETE ===")
    print(f"Master Metrics Table saved to: {out_dir}/metrics/master_benchmarking_metrics.csv")
    print(f"All figures saved to:          {out_dir}/figures/")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal benchmarking suite for microbiome ML models.")
    parser.add_argument("-i", "--input", required=False, default="input", help="Root input directory containing model results.")
    parser.add_argument("-l", "--labels", required=False, default="input/labels.csv", help="Path to external labels CSV.")
    parser.add_argument("-o", "--outdir", required=False, default="benchmarking_results", help="Output destination directory.")
    
    args = parser.parse_args()
    run_benchmarking(args.input, args.labels, args.outdir)