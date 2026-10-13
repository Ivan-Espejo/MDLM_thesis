#!/usr/bin/env python3
"""=============================================================================
Script: 01-generate_random_split.py
Author: Microbiome Analytics Pipeline

Description:
     Creates deterministic, stratified train/test sample splits at increasing
     training fractions for random-mode learning-curve analysis.

Inputs:
     --labels: CSV containing Sample_Name and binary Label columns.
     --outdir: directory in which to create run folders.
     The random_state is fixed at 42 in the split function.

Outputs:
     run*/train_samples.txt, test_samples.txt, and run*_info.txt under --outdir.
============================================================================="""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_learning_curve_splits(labels_path, out_dir, seed=42):
    print(f"Generating learning curve splits from {labels_path}...")
    df = pd.read_csv(labels_path)
    
    # Train percentages from 5% to 90%
    train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for size in train_sizes:
        # Folder names like run10, run20, run50, etc.
        folder_name = f"run{int(size * 100)}"
        run_dir = os.path.join(out_dir, folder_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Stratified split to ensure the balance of IBD vs Healthy is maintained
        train_df, test_df = train_test_split(
            df, train_size=size, stratify=df['Label'], random_state=seed
        )
        
        # Save sample lists (just the names, one per line)
        with open(os.path.join(run_dir, "train_samples.txt"), 'w') as f:
            f.write("\n".join(train_df['Sample_Name'].tolist()))
            
        with open(os.path.join(run_dir, "test_samples.txt"), 'w') as f:
            f.write("\n".join(test_df['Sample_Name'].tolist()))
            
        # -------------------------------------------------------------
        # Calculate Diagnostics for runXX_info.txt
        # -------------------------------------------------------------
        ts_ratio = 1.0 - size
        
        tr_total = len(train_df)
        tr_ibd = train_df['Label'].sum()
        tr_healthy = tr_total - tr_ibd
        tr_ibd_ratio = tr_ibd / tr_total if tr_total > 0 else 0
        
        ts_total = len(test_df)
        ts_ibd = test_df['Label'].sum()
        ts_healthy = ts_total - ts_ibd
        ts_ibd_ratio = ts_ibd / ts_total if ts_total > 0 else 0
        
        info_path = os.path.join(run_dir, f"{folder_name}_info.txt")
        with open(info_path, "w") as f:
            f.write(f"======================================\n")
            f.write(f"RANDOM Split {folder_name} Diagnostics\n")
            f.write(f"======================================\n")
            f.write(f"Global Data Split (Test Ratio):  {ts_ratio*100:.2f}%\n")
            f.write(f"--------------------------------------\n")
            f.write(f"TRAINING SET:\n")
            f.write(f"Total Train Samples:       {tr_total}\n")
            f.write(f"Train Balance (IBD Ratio): {tr_ibd_ratio*100:.2f}%\n")
            f.write(f"Train Classes:             {tr_ibd} IBD | {tr_healthy} Healthy\n")
            f.write(f"--------------------------------------\n")
            f.write(f"TESTING SET:\n")
            f.write(f"Total Test Samples:        {ts_total}\n")
            f.write(f"Test Balance (IBD Ratio):  {ts_ibd_ratio*100:.2f}%\n")
            f.write(f"Test Classes:              {ts_ibd} IBD | {ts_healthy} Healthy\n")
            f.write(f"======================================\n")
            
        print(f" -> Created {folder_name}/ (Train: {len(train_df)}, Test: {len(test_df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels", required=True, help="Path to labels.csv")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    args = parser.parse_args()
    
    generate_learning_curve_splits(args.labels, args.outdir)