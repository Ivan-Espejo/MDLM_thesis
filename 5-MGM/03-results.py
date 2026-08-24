#!/usr/bin/env python3
"""
=============================================================================
Script: 03-results.py
Author: Ivan Espejo

Description:
    This script processes the raw logit scores outputted by the MGM model's 
    predict function, merges them with the true biological labels, and evaluates 
    the model's performance. 

    It performs three critical tasks:
    1. Softmax Transformation: Converts raw model logits into standardized 
       probabilities (0 to 1) for both the Healthy and IBD classes.
    2. Thresholding: Assigns a final prediction based on the highest probability.
    3. Visualization: Generates publication-ready figures including a Confusion 
       Matrix Heatmap, ROC Curve, and Precision-Recall Curve.

Inputs (Default):
    -s / --scores : output/y_score.csv (MGM raw output)
    -l / --labels : input/labels.csv (True metadata)
    
Outputs (Default):
    -o / --outdir : output/figures/ (Saves .png files)
=============================================================================
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
    accuracy_score
)

def generate_results(scores_path, labels_path, out_dir):
    print("Loading model scores and true labels...")
    
    # 1. Load the raw scores (index is the SampleID)
    try:
        scores_df = pd.read_csv(scores_path, index_col=0)
        # Ensure column names are strings to avoid indexing errors
        scores_df.columns = scores_df.columns.astype(str)
    except Exception as e:
        print(f"[ERROR] Could not load scores file: {e}")
        return

    # 2. Load the true labels
    try:
        labels_df = pd.read_csv(labels_path)
    except Exception as e:
        print(f"[ERROR] Could not load labels file: {e}")
        return

    # 3. Merge the datasets
    # labels_df has 'Sample_Name' and 'Label'. scores_df index is the sample ID.
    merged_df = labels_df.merge(scores_df, left_on='Sample_Name', right_index=True, how='inner')
    
    if merged_df.empty:
        print("[ERROR] Merge failed. Ensure Sample_Name in labels.csv matches y_score.csv index.")
        return
        
    print(f"Successfully matched {merged_df.shape[0]} samples.")

    # 4. Convert raw logits to probabilities (Softmax)
    # This prevents math errors and standardizes the scores
    logits = merged_df[['0', '1']].values
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    
    y_true = merged_df['Label'].values
    y_prob_ibd = probs[:, 1]
    
    # Predict Class 1 (IBD) if its probability is > Class 0 (Healthy)
    y_pred = (y_prob_ibd > 0.5).astype(int)

    # 5. Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # --- FIGURE 1: CONFUSION MATRIX HEATMAP ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={"size": 16, "weight": "bold"},
                xticklabels=['Healthy (0)', 'IBD (1)'], 
                yticklabels=['Healthy (0)', 'IBD (1)'])
    
    plt.title(f'Evaluation Confusion Matrix\nAccuracy: {acc*100:.1f}%', fontsize=14, pad=15)
    plt.ylabel('True Biological State', fontsize=12, fontweight='bold')
    plt.xlabel('MGM Predicted State', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # --- FIGURE 2: ROC & PR CURVES (SIDE-BY-SIDE) ---
    print("Generating ROC and PR Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob_ibd)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic', fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Plot Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob_ibd)
    pr_auc = average_precision_score(y_true, y_prob_ibd)
    baseline_pr = sum(y_true) / len(y_true) 

    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    ax2.plot([0, 1], [baseline_pr, baseline_pr], color='gray', lw=2, linestyle='--', label=f'Baseline ({baseline_pr:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14)
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_pr_curves.png'), dpi=300)
    plt.close()

    print("\n=== POSTPROCESSING COMPLETE ===")
    print(f"Figures saved to: {out_dir}")
    print(f"Final ROC-AUC:    {roc_auc:.3f}")
    print(f"Final PR-AUC:     {pr_auc:.3f}")

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Generates evaluation metrics and plots from MGM model predictions."
    )
    
    # Setting required=False and providing the default pathways
    parser.add_argument(
        "-s", "--scores", 
        required=False, 
        default="output/predictions.csv/y_score.csv",
        help="Path to the raw logits CSV generated by the MGM predict script."
    )
    parser.add_argument(
        "-l", "--labels", 
        required=False, 
        default="../data/input/labels.csv",
        help="Path to the true labels CSV."
    )
    parser.add_argument(
        "-o", "--outdir", 
        required=False, 
        default="output/figures",
        help="Directory where the generated plots will be saved."
    )
    
    args = parser.parse_args()
    
    generate_results(args.scores, args.labels, args.outdir)