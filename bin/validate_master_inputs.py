#!/usr/bin/env python3
"""=============================================================================
Script: validate_master_inputs.py
Author: Ivan Espejo

Description:
    Checks that a master-workflow input bundle has valid labels, numeric OTU
    abundances, and feature IDs that match records in the sequence FASTA.

Inputs:
    --input-dir: directory containing labels.csv, raw_global_otu_dedup.tsv, and
    merged_seqs.fasta; --mode: loco or random validation rules.

Outputs:
    Prints validated sample, OTU, and FASTA-match counts; exits with an error
    when required files or relationships are invalid.
============================================================================="""

import argparse
import csv
import math
from pathlib import Path


def validate(input_dir, mode):
    labels_path = input_dir / "labels.csv"
    table_path = input_dir / "raw_global_otu_dedup.tsv"
    fasta_path = input_dir / "merged_seqs.fasta"
    for path in (labels_path, table_path, fasta_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required input is missing or empty: {path}")

    with labels_path.open(newline="") as source:
        labels_reader = csv.DictReader(source)
        required = {"Sample_Name", "Label"}
        if mode == "loco":
            required.add("study_id")
        if not required.issubset(labels_reader.fieldnames or ()):
            raise ValueError(f"{labels_path} must contain columns {sorted(required)}")
        labels = list(labels_reader)

    sample_names = [row["Sample_Name"] for row in labels]
    if not labels or any(not name for name in sample_names):
        raise ValueError("labels.csv must contain non-empty sample names")
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("labels.csv contains duplicate sample names")
    label_values = {row["Label"] for row in labels}
    if not label_values.issubset({"0", "1"}) or label_values != {"0", "1"}:
        raise ValueError("labels.csv must contain both binary labels 0 and 1")
    if mode == "loco":
        studies = {row["study_id"] for row in labels}
        if "" in studies or len(studies) < 2:
            raise ValueError("LOCO labels require at least two non-empty study_id values")

    feature_ids = set()
    with table_path.open(newline="") as source:
        table_reader = csv.DictReader(source, delimiter="\t")
        columns = table_reader.fieldnames or []
        if "UnifiedID" not in columns:
            raise ValueError(f"{table_path} must contain a UnifiedID column")
        missing_samples = sorted(set(sample_names) - set(columns))
        if missing_samples:
            raise ValueError(f"OTU table is missing labeled samples: {missing_samples[:5]}")
        value_columns = [name for name in sample_names]
        for row_number, row in enumerate(table_reader, start=2):
            feature_id = row.get("UnifiedID", "")
            if not feature_id or feature_id in feature_ids:
                raise ValueError(f"OTU table has a blank or duplicate UnifiedID at row {row_number}")
            feature_ids.add(feature_id)
            for sample in value_columns:
                try:
                    value = float(row[sample])
                except (TypeError, ValueError):
                    raise ValueError(f"Non-numeric abundance for {feature_id} / {sample}") from None
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid abundance for {feature_id} / {sample}: {value}")

    fasta_ids = set()
    with fasta_path.open() as fasta:
        for line in fasta:
            if line.startswith(">"):
                identifier = line[1:].strip().split()
                if identifier:
                    fasta_ids.add(identifier[0])
    if not feature_ids or not feature_ids.intersection(fasta_ids):
        raise ValueError("No OTU feature IDs match records in merged_seqs.fasta")

    print(
        f"Validated {len(sample_names)} samples, {len(feature_ids)} OTUs, "
        f"{len(feature_ids.intersection(fasta_ids))} FASTA matches ({mode} mode)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("loco", "random"), required=True)
    args = parser.parse_args()
    validate(args.input_dir, args.mode)


if __name__ == "__main__":
    main()