#!/usr/bin/env python3
"""
=============================================================================
Script: 01-generate_loco_split.py
Description:
    Exhaustively searches all possible combinations of independent projects 
    to find validation splits that meet strict size and class-balance criteria.
    Outputs the exact projects for training/testing along with a detailed 
    information file for every valid permutation.
=============================================================================
"""
import pandas as pd
import argparse
import os
import itertools
import random

def generate_splits(labels_path, out_dir, num_runs=None):
    # Load labels
    df = pd.read_csv(labels_path)
    
    # Calculate project-level statistics
    # Since Label is binary (1=IBD, 0=Healthy), sum(Label) gives the exact IBD count
    stats = df.groupby('study_id').agg(
        Total=('Sample_Name', 'count'),
        IBD=('Label', 'sum')
    ).reset_index()
    
    # Healthy count is just Total minus IBD
    stats['Healthy'] = stats['Total'] - stats['IBD']
    
    global_total = stats['Total'].sum()
    projects = stats['study_id'].tolist()
    
    print(f"Total projects to combine: {len(projects)}")
    print(f"Global total samples: {global_total}")
    print("Initiating exhaustive combinatorial search...")
    
    valid_splits = []
    
    # Exhaustively test every possible grouping of projects
    for k in range(1, len(projects)):
        for test_cand in itertools.combinations(projects, k):
            test_cand = frozenset(test_cand)
            train_cand = frozenset(set(projects) - test_cand)
            
            # Check 1: Test Size Constraint (15% to 25% of global data)
            ts_stats = stats[stats['study_id'].isin(test_cand)]
            ts_total = ts_stats['Total'].sum()
            ts_ratio = ts_total / global_total
            
            if not (0.15 <= ts_ratio <= 0.25):
                continue
                
            # Check 2: Train Balance Constraint (42% to 58% IBD)
            tr_stats = stats[stats['study_id'].isin(train_cand)]
            tr_total = tr_stats['Total'].sum()
            tr_ibd = tr_stats['IBD'].sum()
            tr_healthy = tr_stats['Healthy'].sum()
            
            # Avoid division by zero
            if tr_total == 0:
                continue
                
            tr_ibd_ratio = tr_ibd / tr_total
            
            if not (0.42 <= tr_ibd_ratio <= 0.58):
                continue
                
            # If both pass, calculate testing balance for the info file
            ts_ibd = ts_stats['IBD'].sum()
            ts_healthy = ts_stats['Healthy'].sum()
            ts_ibd_ratio = ts_ibd / ts_total if ts_total > 0 else 0
            
            valid_splits.append({
                'train_projects': train_cand,
                'test_projects': test_cand,
                'ts_ratio': ts_ratio,
                'tr_ibd_ratio': tr_ibd_ratio,
                'ts_ibd_ratio': ts_ibd_ratio,
                'tr_total': tr_total,
                'ts_total': ts_total,
                'tr_ibd': tr_ibd,
                'tr_healthy': tr_healthy,
                'ts_ibd': ts_ibd,
                'ts_healthy': ts_healthy
            })
            
    print(f"Search complete. Found {len(valid_splits)} valid combinations.")
    
    if not valid_splits:
        print("[ERROR] No combinations met the constraints. Adjust ratio or balance limits.")
        return

    # Handle -r parameter logic
    if num_runs is not None:
        if num_runs < len(valid_splits):
            print(f"Randomly selecting {num_runs} splits as requested via '-r'...")
            random.seed(42) # Ensures reproducibility 
            valid_splits = random.sample(valid_splits, num_runs)
        else:
            print(f"Requested {num_runs} runs, but only found {len(valid_splits)}. Outputting all.")
    else:
        print("No '-r' limit provided. Exporting ALL valid combinations.")
        
    print(f"\nWriting files to {out_dir}/...")
    
    # Export the splits and info files
    for i, split in enumerate(valid_splits, 1):
        run_dir = os.path.join(out_dir, f"run{i}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Write project string files (NOW SEPARATED BY NEWLINE)
        with open(os.path.join(run_dir, "train_projects.txt"), "w") as f:
            f.write("\n".join(split['train_projects']))
        with open(os.path.join(run_dir, "test_projects.txt"), "w") as f:
            f.write("\n".join(split['test_projects']))
            
        # Write detailed info log
        info_path = os.path.join(run_dir, f"run{i}_info.txt")
        with open(info_path, "w") as f:
            f.write(f"======================================\n")
            f.write(f"LOCO Split Run {i} Diagnostics\n")
            f.write(f"======================================\n")
            f.write(f"Global Data Split (Test Ratio):  {split['ts_ratio']*100:.2f}%\n")
            f.write(f"--------------------------------------\n")
            f.write(f"TRAINING SET:\n")
            f.write(f"Total Train Samples:       {split['tr_total']}\n")
            f.write(f"Train Balance (IBD Ratio): {split['tr_ibd_ratio']*100:.2f}%\n")
            f.write(f"Train Classes:             {split['tr_ibd']} IBD | {split['tr_healthy']} Healthy\n")
            f.write(f"--------------------------------------\n")
            f.write(f"TESTING SET:\n")
            f.write(f"Total Test Samples:        {split['ts_total']}\n")
            f.write(f"Test Balance (IBD Ratio):  {split['ts_ibd_ratio']*100:.2f}%\n")
            f.write(f"Test Classes:              {split['ts_ibd']} IBD | {split['ts_healthy']} Healthy\n")
            f.write(f"======================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels", required=True, help="Path to labels.csv")
    parser.add_argument("-o", "--outdir", required=True, help="Root directory to generate run folders")
    
    # 'default=None' means if you don't use -r, it generates everything
    parser.add_argument("-r", "--runs", type=int, default=None, 
                        help="Number of split permutations to keep. If omitted, generates ALL valid combinations.")
    args = parser.parse_args()
    
    generate_splits(args.labels, args.outdir, args.runs)