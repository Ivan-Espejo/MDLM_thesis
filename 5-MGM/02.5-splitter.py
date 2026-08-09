#!/usr/bin/env python3
"""
=============================================================================
Script: 02.5-lopo-split_data.py
Author: Ivan Espejo

Description:
    Performs a Leave-One-Project-Out (LOPO) split of the MGM-ready OTU 
    matrix and labels. It maps the sample IDs against a metadata file to 
    isolate specific projects into the testing set, and target projects into 
    the finetuning (training) set based on explicitly provided text lists.
=============================================================================
"""
import pandas as pd
import argparse
import os

def lopo_split_dataset(otu_table_path, labels_path, metadata_path, outdir, train_txt, test_txt):
    print(f"\n[1/4] Loading data and metadata...")
    print(f" - OTU Table: {otu_table_path}")
    print(f" - Labels:    {labels_path}")
    print(f" - Metadata:  {metadata_path}")
    
    # Load data
    labels = pd.read_csv(labels_path)
    otu = pd.read_csv(otu_table_path, index_col=0)
    
    try:
        # Assumes metadata is TSV. Change sep=',' if it is a CSV.
        metadata = pd.read_csv(metadata_path, sep='\t')
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
        
    total_samples = labels.shape[0]
    print(f"Total samples loaded: {total_samples}")

    print(f"\n[2/4] Executing LOPO Split Strategy")
    
    # Read project strings from text files provided by the master orchestrator
    with open(train_txt, 'r') as f:
        train_projects = f.read().strip().split()
    with open(test_txt, 'r') as f:
        test_projects = f.read().strip().split()

    # Assume column 0 in labels.csv is the Sample_Name
    sample_col_name = labels.columns[0]
    
    # Merge labels with metadata to fetch the study_id for each sample
    merged = labels.merge(metadata, how='left', left_on=sample_col_name, right_on='sample_id')
    
    # Check for unmatched samples just in case
    unmatched = merged['study_id'].isna().sum()
    if unmatched > 0:
        print(f"[WARNING] {unmatched} samples in labels.csv did not match any 'sample_id' in metadata.")

    # Create boolean masks strictly using the provided project lists
    train_mask = merged['study_id'].isin(train_projects).values
    test_mask = merged['study_id'].isin(test_projects).values

    # Split labels
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]
    
    # Split metadata for tracking
    train_metadata = merged[train_mask]
    test_metadata = merged[test_mask]

    # Extract sample IDs to slice the OTU columns
    train_samples = train_labels.iloc[:, 0].astype(str).tolist()
    test_samples = test_labels.iloc[:, 0].astype(str).tolist()
    
    # Safely slice (only keeping columns that actually exist in the OTU table)
    valid_train_samples = [s for s in train_samples if s in otu.columns]
    valid_test_samples = [s for s in test_samples if s in otu.columns]

    train_otu = otu[valid_train_samples]
    test_otu = otu[valid_test_samples]

    os.makedirs(outdir, exist_ok=True)

    print(f"\n[3/4] Split Diagnostics")
    print(f" -> Finetuning set (Train): {len(train_samples)} samples ({(len(train_samples)/total_samples)*100:.1f}%)")
    print(f" -> Testing set (Test):     {len(test_samples)} samples ({(len(test_samples)/total_samples)*100:.1f}%)")

    if len(test_samples) == 0 or len(train_samples) == 0:
        print("\n[WARNING] One of the sets is EMPTY. Check if the project IDs match the metadata exactly.")

    print(f"\n[4/4] Exporting files...")
    train_labels.to_csv(f"{outdir}/finetune_labels.csv", index=False)
    test_labels.to_csv(f"{outdir}/test_labels.csv", index=False)
    
    train_otu.to_csv(f"{outdir}/finetune_abundances.csv")
    test_otu.to_csv(f"{outdir}/test_abundances.csv")
    
    # Export metadata trackers (useful for downstream plotting and batch effect checks)
    train_metadata.to_csv(f"{outdir}/finetune_metadata.csv", index=False)
    test_metadata.to_csv(f"{outdir}/test_metadata.csv", index=False)
    
    print(f"Successfully saved 4 CSV data files and 2 tracker files to: {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOPO dataset splitter for the MGM pipeline.")
    
    parser.add_argument("-i", "--otu_table", required=True, help="Path to the collapsed MGM OTU table.")
    parser.add_argument("-l", "--labels", required=True, help="Path to the true labels CSV.")
    parser.add_argument("-m", "--metadata", required=True, help="Path to metadata TSV/CSV containing sample_id and study_id.")
    parser.add_argument("-o", "--outdir", required=True, help="Directory to save the split datasets.")
    parser.add_argument("--train_txt", required=True, help="Text file containing train projects separated by space")
    parser.add_argument("--test_txt", required=True, help="Text file containing test projects separated by space")
    
    args = parser.parse_args()
    
    lopo_split_dataset(
        otu_table_path=args.otu_table, 
        labels_path=args.labels, 
        metadata_path=args.metadata,
        outdir=args.outdir, 
        train_txt=args.train_txt,
        test_txt=args.test_txt
    )