#!/usr/bin/env python3
"""
=============================================================================
Script: 1.5-mikropml_splitter.py
Author: Ivan Espejo

Description:
    Dynamically partitions global OTU tables and label matrices into strictly 
    isolated Train and Test subsets for the mikropml pipeline. 
    
    Triggered dynamically by the SLURM master batch script, it cross-references 
    the global metadata against a specific LOCO run's assigned project space 
    to construct temporary, run-specific input files. 
    
    Note: OTU table is expected to have Samples as Columns and OTUs as Rows.
=============================================================================
"""
import pandas as pd
import argparse
import os

def generate_mikropml_split(global_otu, global_labels, metadata, train_txt, test_txt, out_dir):
    print("Loading Global Files...")
    
    # Load labels and metadata
    labels = pd.read_csv(global_labels)
    meta = pd.read_csv(metadata, sep='\t')
    
    # Read project strings
    with open(train_txt, 'r') as f:
        train_projs = f.read().strip().split()
    with open(test_txt, 'r') as f:
        test_projs = f.read().strip().split()
        
    print(f"Train Projects: {len(train_projs)} | Test Projects: {len(test_projs)}")
    
    # Map samples to projects
    sample_col = labels.columns[0]
    merged = labels.merge(meta, how='left', left_on=sample_col, right_on='sample_id')
    
    # Generate Boolean masks
    train_mask = merged['study_id'].isin(train_projs).values
    test_mask = merged['study_id'].isin(test_projs).values
    
    # Slice Labels
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]
    
    # Extract string IDs to slice OTU columns safely
    train_sample_ids = train_labels[sample_col].astype(str).tolist()
    test_sample_ids = test_labels[sample_col].astype(str).tolist()

    print("Loading Global OTU Table...")
    # Load OTU table (index_col=0 keeps the OTU IDs as the index, Samples are columns)
    otu = pd.read_csv(global_otu, sep='\t', index_col=0)
    
    # Safely slice OTU columns (keeping only columns that actually exist)
    valid_train_samples = [s for s in train_sample_ids if s in otu.columns]
    valid_test_samples = [s for s in test_sample_ids if s in otu.columns]
    
    train_otu = otu[valid_train_samples]
    test_otu = otu[valid_test_samples]
    
    # Create the run-specific temporary output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Export datasets
    train_otu.to_csv(os.path.join(out_dir, "train_otu.tsv"), sep='\t')
    test_otu.to_csv(os.path.join(out_dir, "test_otu.tsv"), sep='\t')
    train_labels.to_csv(os.path.join(out_dir, "train_labels.csv"), index=False)
    test_labels.to_csv(os.path.join(out_dir, "test_labels.csv"), index=False)
    
    print(f"Success! {len(valid_train_samples)} Train Samples | {len(valid_test_samples)} Test Samples written to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_otu", required=True)
    parser.add_argument("--global_labels", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--test_txt", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    
    generate_mikropml_split(args.global_otu, args.global_labels, args.meta, 
                            args.train_txt, args.test_txt, args.outdir)