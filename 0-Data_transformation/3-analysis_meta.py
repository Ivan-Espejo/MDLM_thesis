#!/usr/bin/env python3
"""
=============================================================================
Script: analyze_metadata.py
Author: Ivan Espejo
Function: Parses and summarizes clinical microbiome metadata, extracting 
          project-level statistics including sample counts, diagnoses 
          (Segregated by UC vs CD), demographics, and sequencing tech.

Usage: 
    python analyze_metadata.py --meta <metadata.tsv> [options]

Info: 
- Extracts nested JSON properties from 'additional_metadata'.
- Groups data by 'study_id'.
- Segregates IBD into specific Ulcerative Colitis (UC) and Crohn's (CD) counts.
=============================================================================
"""

import argparse
import os
import sys
import pandas as pd
import json

def extract_technology(meta_string):
    """Safely extracts sequencing technology from the JSON string."""
    if pd.isna(meta_string) or not isinstance(meta_string, str):
        return "Unknown"
    try:
        data = json.loads(meta_string)
        instrument = data.get("Instrument", "").strip()
        platform = data.get("Platform", "").strip()
        
        # Combine if both exist and differ, else return whichever is available
        if instrument and platform and instrument.lower() not in platform.lower():
            return f"{platform} ({instrument})"
        return instrument or platform or "Unknown"
    except json.JSONDecodeError:
        return "Unknown"

def main():
    args = parse_args()

    if not os.path.exists(args.meta):
        sys.exit(f"[!] Error: Metadata file '{args.meta}' not found.")

    print(f"[*] Loading metadata from: {args.meta}")
    
    # Load metadata
    df = pd.read_csv(args.meta, sep='\t')

    # Ensure required baseline columns exist
    for col in ['study_id', 'diagnosis']:
        if col not in df.columns:
            sys.exit(f"[!] Critical: Required column '{col}' missing from metadata.")

    # 1. Clean and Transform Columns
    # Extract Technology
    if 'additional_metadata' in df.columns:
        df['Technology'] = df['additional_metadata'].apply(extract_technology)
    else:
        df['Technology'] = "Unknown"

    # Map Diagnoses to Groupings (Segregating UC and CD)
    def map_diagnosis(d):
        d_str = str(d).strip().upper()
        if d_str == "IBD-UC": return "IBD-UC"
        if d_str == "IBD-CD": return "IBD-CD"
        if d in args.non_ibd: return "Healthy"
        if d in args.ibd: return "Other IBD" # Captures Colorectal Cancer or generic IBD
        return "Other"
    
    df['Diag_Group'] = df['diagnosis'].apply(map_diagnosis)

    # Clean Age
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Clean Gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('Unknown').astype(str).str.capitalize()

    # 2. Calculate Global Statistics
    total_samples = len(df)
    total_projects = df['study_id'].nunique()
    
    global_uc = (df['Diag_Group'] == 'IBD-UC').sum()
    global_cd = (df['Diag_Group'] == 'IBD-CD').sum()
    global_other_ibd = (df['Diag_Group'] == 'Other IBD').sum()
    global_healthy = (df['Diag_Group'] == 'Healthy').sum()
    global_other = (df['Diag_Group'] == 'Other').sum()

    print("\n=================================================")
    print("        === METADATA SUMMARY REPORT ===          ")
    print("=================================================")
    print(f" - Total Projects       : {total_projects}")
    print(f" - Total Samples        : {total_samples}")
    print(f" - Total IBD-UC         : {global_uc}")
    print(f" - Total IBD-CD         : {global_cd}")
    if global_other_ibd > 0:
        print(f" - Total Other IBD/CRC  : {global_other_ibd}")
    print(f" - Total Healthy        : {global_healthy}")
    if global_other > 0:
        print(f" - Total Other/Unknown  : {global_other}")
    print("=================================================\n")

    # 3. Calculate Project-Level Statistics
    for project, group in df.groupby('study_id'):
        print(f"[ Project: {project} ]")
        
        # Samples & Diagnosis
        p_samples = len(group)
        p_uc = (group['Diag_Group'] == 'IBD-UC').sum()
        p_cd = (group['Diag_Group'] == 'IBD-CD').sum()
        p_other_ibd = (group['Diag_Group'] == 'Other IBD').sum()
        p_healthy = (group['Diag_Group'] == 'Healthy').sum()
        p_other = (group['Diag_Group'] == 'Other').sum()
        
        diag_str = f"UC: {p_uc} | CD: {p_cd} | Healthy: {p_healthy}"
        if p_other_ibd > 0: diag_str += f" | Other IBD: {p_other_ibd}"
        if p_other > 0: diag_str += f" | Other: {p_other}"
        
        # Sequencing Region
        if 'sequencing_region' in group.columns:
            regions = [str(x) for x in group['sequencing_region'].dropna().unique()]
            region_str = ", ".join(regions) if regions else "Unknown"
        else:
            region_str = "Unknown"

        # Technology
        techs = [str(x) for x in group['Technology'].dropna().unique()]
        tech_str = ", ".join(techs) if techs else "Unknown"

        # Localization
        if 'geo_loc' in group.columns:
            locs = [str(x) for x in group['geo_loc'].dropna().unique()]
            loc_str = ", ".join(locs) if locs else "Unknown"
        else:
            loc_str = "Unknown"

        # Demographics
        if 'gender' in group.columns:
            females = (group['gender'] == 'Female').sum()
            males = (group['gender'] == 'Male').sum()
            gender_str = f"Female: {females} | Male: {males}"
        else:
            gender_str = "Unknown"

        if 'age' in group.columns:
            mean_age = group['age'].mean()
            age_str = f"{mean_age:.1f} years" if not pd.isna(mean_age) else "N/A"
        else:
            age_str = "Unknown"

        # Print Project Block
        print(f"  - Samples     : {p_samples}")
        print(f"  - Diagnosis   : {diag_str}")
        print(f"  - Seq Region  : {region_str}")
        print(f"  - Technology  : {tech_str}")
        print(f"  - Geography   : {loc_str}")
        print(f"  - Demographics: {gender_str} (Mean Age: {age_str})\n")

def parse_args():
    p = argparse.ArgumentParser(description="Analyze and summarize microbiome metadata.")
    p.add_argument('--meta', required=True, 
                   help="Path to the clinical metadata TSV file.")
    p.add_argument('--ibd', nargs='+', default=["IBD-UC", "IBD-CD", "Colorectal Cancer"], 
                   help="List of explicit diagnostic strings to classify as IBD-related diseases.")
    p.add_argument('--non_ibd', nargs='+', default=["Healthy"], 
                   help="List of explicit diagnostic strings to classify as Healthy.")
    return p.parse_args()

if __name__ == '__main__':
    main()