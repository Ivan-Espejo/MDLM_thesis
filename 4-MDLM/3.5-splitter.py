#!/usr/bin/env python3
"""=============================================================================
Script: 3.5-splitter.py
Author: Microbiome ML Master Pipeline

Description:
    Maps preprocessed MDLM arrays to the master sample labels and selects the
    train/test rows listed by the LOCO study split or random sample split.

Inputs:
    --data and --labels: global feature and label NumPy arrays.
    --labels_csv: sample names and study IDs used to align array rows.
    --train_txt and --test_txt: split ID lists; --mode selects ID semantics.

Outputs:
    --outdir/finetune_samples.npy and finetune_labels.npy.
    --outdir/test_samples.npy, test_labels.npy, and test_sample_names.txt.
============================================================================="""
import numpy as np
import pandas as pd
import argparse
import os

def dual_mode_split_npy(mode, data_path, labels_path, labels_csv_path, outdir, train_txt, test_txt):
    print(f"\n[1/4] Loading data arrays for {mode.upper()} splitting...")
    print(f" - Data Tensor: {data_path}")
    print(f" - Global Labels: {labels_csv_path}")
    
    # Load the MDLM tensor format
    X = np.load(data_path)
    y = np.load(labels_path)
    
    try:
        # We exclusively use the global labels.csv now, not the raw metadata
        df_labels = pd.read_csv(labels_csv_path)
    except Exception as e:
        print(f"Error loading labels.csv: {e}")
        return

    if X.shape[0] != len(df_labels):
        raise ValueError(f"CRITICAL ERROR: Array length ({X.shape[0]}) does not match labels.csv length ({len(df_labels)}).")

    print(f"\n[2/4] Executing {mode.upper()} Split Strategy")
    
    # Read IDs from text files (newline separated)
    with open(train_txt, 'r') as f:
        train_list = [line.strip() for line in f if line.strip()]
    with open(test_txt, 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]

    # Create boolean masks based on execution mode
    if mode == "loco":
        if "study_id" not in df_labels.columns:
            raise ValueError("labels.csv must contain 'study_id' for LOCO mode.")
        train_mask = df_labels['study_id'].isin(train_list).values
        test_mask = df_labels['study_id'].isin(test_list).values
    elif mode == "random":
        if "Sample_Name" not in df_labels.columns:
            raise ValueError("labels.csv must contain 'Sample_Name' for RANDOM mode.")
        train_mask = df_labels['Sample_Name'].isin(train_list).values
        test_mask = df_labels['Sample_Name'].isin(test_list).values
    else:
        raise ValueError("Invalid mode. Must be 'loco' or 'random'.")

    # Split the Numpy arrays using the Boolean masks
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Extract the true string sample names of the test set for downstream inference predictions
    test_sample_names = df_labels.iloc[test_mask]['Sample_Name'].tolist()

    os.makedirs(outdir, exist_ok=True)

    print(f"\n[3/4] Split Diagnostics")
    print(f" -> Finetuning set (Train): {X_train.shape[0]} samples ({(X_train.shape[0]/X.shape[0])*100:.1f}%)")
    print(f" -> Testing set (Test):     {X_test.shape[0]} samples ({(X_test.shape[0]/X.shape[0])*100:.1f}%)")
    
    if X_test.shape[0] == 0 or X_train.shape[0] == 0:
        print("\n[WARNING] One of the sets is EMPTY. Check if IDs match the labels.csv exactly.")

    print(f"\n[4/4] Exporting isolated tensors...")
    np.save(os.path.join(outdir, "finetune_samples.npy"), X_train)
    np.save(os.path.join(outdir, "test_samples.npy"), X_test)
    np.save(os.path.join(outdir, "finetune_labels.npy"), y_train)
    np.save(os.path.join(outdir, "test_labels.npy"), y_test)
    
    # Save the test sample names as a raw text file so MDLM inference can append it to the probabilities!
    with open(os.path.join(outdir, "test_sample_names.txt"), "w") as f:
        f.write("\n".join(test_sample_names))
    
    print(f"Successfully saved 4 .npy files and sample mapping to: {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Mode dataset splitter for MDLM numpy arrays.")
    
    parser.add_argument("--mode", required=True, help="'loco' or 'random'")
    parser.add_argument("-i", "--data", required=True, help="Path to the input samples .npy array")
    parser.add_argument("-l", "--labels", required=True, help="Path to the labels .npy array")
    parser.add_argument("--labels_csv", required=True, help="Path to global labels.csv containing Sample_Name/study_id")
    parser.add_argument("-o", "--outdir", required=True, help="Directory to save the split arrays")
    parser.add_argument("--train_txt", required=True, help="Text file containing train IDs")
    parser.add_argument("--test_txt", required=True, help="Text file containing test IDs")

    args = parser.parse_args()

    dual_mode_split_npy(
        mode=args.mode,
        data_path=args.data,
        labels_path=args.labels,
        labels_csv_path=args.labels_csv,
        outdir=args.outdir,
        train_txt=args.train_txt,
        test_txt=args.test_txt
    )