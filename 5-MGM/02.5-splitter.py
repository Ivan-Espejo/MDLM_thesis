#!/usr/bin/env python3
"""
=============================================================================
Script: 02.5-splitter.py
Author: Microbiome ML Master Pipeline

Description:
    Performs dual-mode slicing of the global collapsed MGM OTU table and 
    labels. Isolates Train and Test subsets for MGM based on either 
    'loco' (study_id) or 'random' (Sample_Name) definitions.
=============================================================================
"""
import pandas as pd
import argparse
import os

def dual_mode_split_mgm(mode, otu_table_path, labels_path, outdir, train_txt, test_txt):
    print(f"\n[1/4] Loading MGM data for {mode.upper()} splitting...")
    print(f" - OTU Table: {otu_table_path}")
    print(f" - Labels:    {labels_path}")
    
    labels = pd.read_csv(labels_path)
    otu = pd.read_csv(otu_table_path, index_col=0)
    
    total_samples = labels.shape[0]
    print(f"Total samples loaded: {total_samples}")

    print(f"\n[2/4] Executing {mode.upper()} Split Strategy")
    
    # Read text lists (newline separated)
    with open(train_txt, 'r') as f:
        train_list = [line.strip() for line in f if line.strip()]
    with open(test_txt, 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]

    # Create boolean masks based on mode
    if mode == "loco":
        if "study_id" not in labels.columns:
            raise ValueError("labels.csv must contain 'study_id' for LOCO mode.")
        train_mask = labels['study_id'].isin(train_list).values
        test_mask = labels['study_id'].isin(test_list).values
    elif mode == "random":
        if "Sample_Name" not in labels.columns:
            raise ValueError("labels.csv must contain 'Sample_Name' for RANDOM mode.")
        train_mask = labels['Sample_Name'].isin(train_list).values
        test_mask = labels['Sample_Name'].isin(test_list).values
    else:
        raise ValueError("Invalid mode. Must be 'loco' or 'random'.")

    # Split labels (preserving Sample_Name and Label for MGM)
    train_labels = labels[train_mask][['Sample_Name', 'Label']]
    test_labels = labels[test_mask][['Sample_Name', 'Label']]
    
    # Extract sample IDs to slice the OTU columns
    train_samples = labels.loc[train_mask, 'Sample_Name'].astype(str).tolist()
    test_samples = labels.loc[test_mask, 'Sample_Name'].astype(str).tolist()
    
    # Safely slice OTU table (keeping columns present in the table)
    valid_train_samples = [s for s in train_samples if s in otu.columns]
    valid_test_samples = [s for s in test_samples if s in otu.columns]

    train_otu = otu[valid_train_samples]
    test_otu = otu[valid_test_samples]

    os.makedirs(outdir, exist_ok=True)

    print(f"\n[3/4] Split Diagnostics")
    print(f" -> Finetuning set (Train): {len(train_samples)} samples ({(len(train_samples)/total_samples)*100:.1f}%)")
    print(f" -> Testing set (Test):     {len(test_samples)} samples ({(len(test_samples)/total_samples)*100:.1f}%)")

    if len(test_samples) == 0 or len(train_samples) == 0:
        print("\n[WARNING] One of the sets is EMPTY. Check if IDs match labels.csv exactly.")

    print(f"\n[4/4] Exporting MGM input files...")
    train_labels.to_csv(os.path.join(outdir, "finetune_labels.csv"), index=False)
    test_labels.to_csv(os.path.join(outdir, "test_labels.csv"), index=False)
    
    train_otu.to_csv(os.path.join(outdir, "finetune_abundances.csv"))
    test_otu.to_csv(os.path.join(outdir, "test_abundances.csv"))
    
    print(f"Successfully saved 4 CSV files to: {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Mode dataset splitter for the MGM pipeline.")
    
    parser.add_argument("--mode", required=True, help="'loco' or 'random'")
    parser.add_argument("-i", "--otu_table", required=True, help="Path to collapsed MGM OTU table.")
    parser.add_argument("-l", "--labels", required=True, help="Path to labels.csv")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("--train_txt", required=True, help="Train IDs text file")
    parser.add_argument("--test_txt", required=True, help="Test IDs text file")
    
    args = parser.parse_args()
    
    dual_mode_split_mgm(
        mode=args.mode,
        otu_table_path=args.otu_table, 
        labels_path=args.labels, 
        outdir=args.outdir, 
        train_txt=args.train_txt, 
        test_txt=args.test_txt
    )