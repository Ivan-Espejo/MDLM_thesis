#!/usr/bin/env python3
"""Build a small, balanced input bundle for exercising the master workflows."""

import argparse
import csv
import re
from pathlib import Path


STUDIES = (
    "PRJNA1259137",
    "PRJNA368966",
    "PRJNA450340",
    "PRJNA610934",
    "PRJNA684584",
    "PRJNA753210",
    "PRJNA757573",
    "PRJEB13680",
)
PRIMER = re.compile(r"GTG[CT]CAGC[AC]GCCGCGGTAA")


def read_fasta(path):
    records = {}
    identifier = None
    sequence = []
    with path.open() as fasta:
        for line in fasta:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records[identifier] = "".join(sequence[1:])
                header = line[1:]
                identifier = header.split()[0]
                sequence = [f">{header}"]
            else:
                sequence.append(line)
        if identifier is not None:
            records[identifier] = "".join(sequence[1:])
    return records


def build(source_dir, output_dir, samples_per_class, feature_count):
    labels_path = source_dir / "labels.csv"
    table_path = source_dir / "raw_global_otu_dedup.tsv"
    fasta_path = source_dir / "merged_seqs.fasta"
    for path in (labels_path, table_path, fasta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")

    with labels_path.open(newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"Sample_Name", "Label", "study_id"}
        if not required_columns.issubset(reader.fieldnames or ()):
            raise ValueError(f"{labels_path} must contain {sorted(required_columns)}")
        labels = list(reader)

    selected_labels = []
    for study in STUDIES:
        for label in ("0", "1"):
            matching = sorted(
                (row for row in labels if row["study_id"] == study and row["Label"] == label),
                key=lambda row: row["Sample_Name"],
            )
            if len(matching) < samples_per_class:
                raise ValueError(f"Study {study} has fewer than {samples_per_class} samples for label {label}")
            selected_labels.extend(matching[:samples_per_class])

    sample_names = [row["Sample_Name"] for row in selected_labels]
    sequences = read_fasta(fasta_path)
    with table_path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or "UnifiedID" not in reader.fieldnames:
            raise ValueError(f"{table_path} must contain a UnifiedID column")
        missing_samples = sorted(set(sample_names) - set(reader.fieldnames))
        if missing_samples:
            raise ValueError(f"Samples missing from OTU table: {missing_samples[:5]}")
        has_taxon = "Taxon" in reader.fieldnames
        feature_rows = []
        for row in reader:
            feature_id = row["UnifiedID"]
            sequence = sequences.get(feature_id)
            if sequence is None or not PRIMER.search(sequence):
                continue
            values = [row[name] for name in sample_names]
            total = sum(float(value or 0) for value in values)
            if total > 0:
                feature_rows.append((total, feature_id, values, row.get("Taxon", ""), sequence))

    feature_rows.sort(key=lambda row: (-row[0], row[1]))
    feature_rows = feature_rows[:feature_count]
    if len(feature_rows) < feature_count:
        raise ValueError(
            f"Found only {len(feature_rows)} sequence-backed features with the V4 primer; "
            f"requested {feature_count}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_files = ("labels.csv", "raw_global_otu_dedup.tsv", "merged_seqs.fasta")
    if any((output_dir / name).exists() for name in expected_files):
        raise FileExistsError(f"Sample files already exist in {output_dir}; remove them before rebuilding")

    with (output_dir / "labels.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("Sample_Name", "Label", "study_id"))
        writer.writeheader()
        writer.writerows(selected_labels)

    table_columns = ["UnifiedID", *sample_names]
    if has_taxon:
        table_columns.append("Taxon")
    with (output_dir / "raw_global_otu_dedup.tsv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=table_columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for _, feature_id, values, taxon, _ in feature_rows:
            row = {"UnifiedID": feature_id, **dict(zip(sample_names, values))}
            if has_taxon:
                row["Taxon"] = taxon
            writer.writerow(row)

    with (output_dir / "merged_seqs.fasta").open("w") as output:
        for _, feature_id, _, _, sequence in feature_rows:
            output.write(f">{feature_id}\n")
            for offset in range(0, len(sequence), 80):
                output.write(sequence[offset : offset + 80] + "\n")

    print(f"Wrote {len(selected_labels)} samples across {len(STUDIES)} studies and {len(feature_rows)} features to {output_dir}")


def main():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=repo_dir / "data/input")
    parser.add_argument("--output-dir", type=Path, default=repo_dir / "data/sample_input")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--features", type=int, default=32)
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.features < 1:
        parser.error("--samples-per-class and --features must be positive")
    build(args.source_dir, args.output_dir, args.samples_per_class, args.features)


if __name__ == "__main__":
    main()