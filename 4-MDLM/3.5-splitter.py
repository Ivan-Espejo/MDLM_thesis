#!/usr/bin/env python3
"""
=============================================================================
Script: 3.5-lopo-splitter.py
Author: Ivan Espejo

Description:
    Performs a Leave-One-Project-Out (LOPO) split of the preprocessed MDLM 
    numpy arrays. It maps the row-ordered arrays against a metadata file to 
    isolate specific projects into the testing set, and target projects into 
    the finetuning (training) set.
=============================================================================
"""
import numpy as np
import pandas as pd
import argparse
import os

def lopo_split_npy(data_path, labels_path, metadata_path, outdir, train_txt, test_txt):
    print(f"\n[1/4] Loading data arrays and metadata...")
    print(f" - Data Tensor: {data_path}")
    print(f" - Labels:      {labels_path}")
    print(f" - Metadata:    {metadata_path}")
    
    # Load the MDLM tensor format
    X = np.load(data_path)
    y = np.load(labels_path)
    
    try:
        metadata = pd.read_csv(metadata_path, sep='\t')
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    if X.shape[0] != len(metadata):
        raise ValueError(f"CRITICAL ERROR: Array length ({X.shape[0]}) does not match metadata length ({len(metadata)}).")

    print(f"\n[2/4] Executing LOPO Split Strategy")
    # Read project strings from text files
    with open(train_txt, 'r') as f:
        train_projects = f.read().strip().split()
    with open(test_txt, 'r') as f:
        test_projects = f.read().strip().split()

    # Create boolean masks strictly using the provided project lists
    train_mask = metadata['study_id'].isin(train_projects).values
    test_mask = metadata['study_id'].isin(test_projects).values

    # Split the arrays using the Boolean masks
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]

    os.makedirs(outdir, exist_ok=True)

    print(f"\n[3/4] Split Diagnostics")
    print(f" -> Finetuning set (Train): {X_train.shape[0]} samples ({(X_train.shape[0]/X.shape[0])*100:.1f}%)")
    print(f" -> Testing set (Test):     {X_test.shape[0]} samples ({(X_test.shape[0]/X.shape[0])*100:.1f}%)")
    
    if X_test.shape[0] == 0 or X_train.shape[0] == 0:
        print("\n[WARNING] One of the sets is EMPTY. Check if project IDs match the metadata exactly.")

    print(f"\n[4/4] Exporting tensors...")
    np.save(os.path.join(outdir, "finetune_samples.npy"), X_train)
    np.save(os.path.join(outdir, "test_samples.npy"), X_test)
    np.save(os.path.join(outdir, "finetune_labels.npy"), y_train)
    np.save(os.path.join(outdir, "test_labels.npy"), y_test)
    
    # Save the split labels/metadata for downstream tracking
    train_metadata = metadata[train_mask]
    test_metadata = metadata[test_mask]
    train_metadata.to_csv(os.path.join(outdir, "finetune_metadata.csv"), index=False)
    test_metadata.to_csv(os.path.join(outdir, "test_metadata.csv"), index=False)
    
    print(f"Successfully saved 4 .npy files and 2 .csv trackers to: {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOPO dataset splitter for MDLM numpy arrays.")
    
    parser.add_argument("-i", "--data", required=True, help="Path to the input samples .npy array")
    parser.add_argument("-l", "--labels", required=True, help="Path to the labels .npy array")
    parser.add_argument("-m", "--metadata", required=True, help="Path to metadata TSV/CSV containing study_id")
    parser.add_argument("-o", "--outdir", required=True, help="Directory to save the split arrays")
    parser.add_argument("--train_txt", required=True, help="Text file containing train projects separated by space")
    parser.add_argument("--test_txt", required=True, help="Text file containing test projects separated by space")

    args = parser.parse_args()

    lopo_split_npy(
        data_path=args.data,
        labels_path=args.labels,
        metadata_path=args.metadata,
        outdir=args.outdir,
        train_txt=args.train_txt,
        test_txt=args.test_txt
    )