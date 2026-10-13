#!/usr/bin/env python3
"""=============================================================================
Script: 1-single_study.py
Author: Ays (adapted by Ivan Espejo)

Description:
    Processes each project's abundance table and FASTA independently. Imports
    sequence/count data into QIIME 2, maps features to Greengenes2 and SILVA,
    assigns unified feature IDs, and writes a project-level taxonomy table.

Inputs:
    --projects: one or more project directories.
    --table and --seqs: input filenames within each project directory.
    --outdir: destination for per-project results and logs.
    Greengenes2 QIIME 2 references and SILVA 138.2 FASTA under data/references.
    The SILVA taxonomy TSV is optional; without it, SILVA hits lack those
    taxonomy annotations but sequence matching can still run.

Outputs:
    <project>_unified_tax.tsv for each processed project and logs_<project>/
    diagnostics beneath --outdir. Temporary QIIME 2/BIOM files are cleaned up.
============================================================================="""
import os, sys, subprocess, hashlib, argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../data"))
REF_DIR = os.path.join(DATA_DIR, "references/gg2_data")
REF_BACKBONE, REF_TAX_MAP = "2024.09.backbone.full-length.fna.qza", "2024.09.taxonomy.asv.nwk.qza"
SILVA_DIR = os.path.join(DATA_DIR, "references/silva_138")
SILVA_FASTA = os.path.join(SILVA_DIR, "silva-138.2-ssu-nr99-seqs.fasta")
SILVA_TAX_TSV = os.path.join(SILVA_DIR, "silva_138.2_ssu_nr99_taxonomy.tsv")
USE_QIIME2_SILVA_TAXONOMY = True
SILVA_CACHE_DIR = os.path.join(REF_DIR, "_silva_cache")
SILVA_READS_QZA, SILVA_TAX_QZA = os.path.join(SILVA_CACHE_DIR, "silva_reads.qza"), os.path.join(SILVA_CACHE_DIR, "silva_tax.qza")

GG2_PERC_IDENTITY, VSEARCH_GG2_ID = 0.998, 0.99
SILVA_IDENTITY, SILVA_IDDEF, SILVA_QUERY_COV = 0.99, 2, 0.80
THREADS = 9

def log(msg): print(msg, flush=True)
def step(title): log(f"\n{'='*78}\n[STEP] {title}\n{'='*78}")
def run_cmd(cmd, label=""): 
    log(f"\n[EXEC:{label}] {cmd}" if label else f"\n[EXEC] {cmd}")
    subprocess.run(cmd, shell=True, check=True)
def safe_rm(path): 
    if os.path.exists(path): os.remove(path)
def md5_seq(seq): return hashlib.md5("".join(seq.split()).upper().encode("utf-8")).hexdigest()

def read_fasta(path):
    seqs, cur, buf = {}, None, []
    with open(path, "r") as f:
        for line in (l.strip() for l in f if l.strip()):
            if line.startswith(">"):
                if cur: seqs[cur] = "".join(buf)
                cur, buf = line[1:].split()[0], []
            else: buf.append(line)
        if cur: seqs[cur] = "".join(buf)
    return seqs

def write_fasta(seqs, ids, out_path):
    with open(out_path, "w") as out:
        for sid in (i for i in ids if i in seqs): 
            out.write(f">{sid}\n" + "\n".join(seqs[sid][i:i+80] for i in range(0, len(seqs[sid]), 80)) + "\n")

def parse_uc(uc_path, label):
    if not os.path.exists(uc_path): return pd.DataFrame(columns=["Original_ZOTU", f"{label}_TargetID", "PctID", "Strand"])
    rows = [{"Query": p[8], "Target": p[9], "PctID": float(p[3]) if p[3].replace('.','',1).isdigit() else None, "Strand": p[4]} 
            for p in (line.rstrip("\n").split("\t") for line in open(uc_path) if line.startswith("H"))]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Query", "Target", "PctID", "Strand"])
    if df.empty: return pd.DataFrame(columns=["Original_ZOTU", f"{label}_TargetID", "PctID", "Strand"])
    return df.sort_values(["Query", "PctID"], ascending=[True, False]).drop_duplicates(subset=["Query"]).rename(columns={"Query": "Original_ZOTU", "Target": f"{label}_TargetID"})

def load_tax(path):
    if not path or not os.path.exists(path): return {}
    df = pd.read_csv(path, sep="\t")
    cols = [c.lower() for c in df.columns]
    fid_col = df.columns[cols.index("feature id")] if "feature id" in cols else df.columns[cols.index("id")]
    tax_col = df.columns[cols.index("taxon")] if "taxon" in cols else df.columns[cols.index("taxonomy")]
    return df.set_index(fid_col)[tax_col].fillna("").astype(str).to_dict()

def load_counts(path):
    df = pd.read_csv(path, sep="," if path.lower().endswith(".csv") else "\t", dtype=str).set_index(pd.read_csv(path, nrows=0, sep="," if path.lower().endswith(".csv") else "\t").columns[0])
    df_num = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_num[[c for c in df_num.columns if df_num[c].any()]].copy()

def prep_silva():
    if not USE_QIIME2_SILVA_TAXONOMY or not os.path.exists(SILVA_FASTA) or not os.path.exists(SILVA_TAX_TSV): return
    os.makedirs(SILVA_CACHE_DIR, exist_ok=True)
    if not os.path.exists(SILVA_READS_QZA): run_cmd(f"qiime tools import --input-path {SILVA_FASTA} --output-path {SILVA_READS_QZA} --type 'FeatureData[Sequence]'")
    if not os.path.exists(SILVA_TAX_QZA): run_cmd(f"qiime tools import --input-path {SILVA_TAX_TSV} --output-path {SILVA_TAX_QZA} --type 'FeatureData[Taxonomy]'")

def silva_lca(query_fasta, log_dir, tag):
    if not USE_QIIME2_SILVA_TAXONOMY or not os.path.exists(SILVA_READS_QZA): return {}
    qza, out_qza, hits_qza, exp_dir = [os.path.join(log_dir, f"{tag}_{s}") for s in ["seqs.qza", "silva_tax.qza", "silva_hits.qza", "silva_exp"]]
    for f in [qza, out_qza, hits_qza]: safe_rm(f)
    run_cmd(f"qiime tools import --input-path {query_fasta} --output-path {qza} --type 'FeatureData[Sequence]'")
    run_cmd(f"qiime feature-classifier classify-consensus-vsearch --i-query {qza} --i-reference-reads {SILVA_READS_QZA} --i-reference-taxonomy {SILVA_TAX_QZA} --p-perc-identity {SILVA_IDENTITY} --p-maxaccepts 5 --p-strand both --p-threads {THREADS} --o-classification {out_qza} --o-search-results {hits_qza}")
    os.makedirs(exp_dir, exist_ok=True)
    run_cmd(f"qiime tools export --input-path {out_qza} --output-path {exp_dir}")
    tsv = os.path.join(exp_dir, "taxonomy.tsv")
    return pd.read_csv(tsv, sep="\t").set_index("Feature ID")["Taxon"].fillna("").astype(str).to_dict() if os.path.exists(tsv) else {}

def process(name, table, seqs, outdir):
    step(f"Init & Validate: {name}")
    log_dir = os.path.join(outdir, f"logs_{name}")
    os.makedirs(log_dir, exist_ok=True)
    df_counts, all_seqs = load_counts(table), read_fasta(seqs)
    
    feats = df_counts.index.astype(str).tolist()
    missing = [z for z in feats if z not in all_seqs or not all_seqs.get(z,"").strip()]
    kept = [z for z in feats if z not in missing]
    
    pd.DataFrame([{"Original_ZOTU": z, "Status": "dropped" if z in missing else "kept", "Kept_From": "PENDING"} for z in feats]).to_csv(os.path.join(log_dir, "zotu_status.tsv"), sep="\t", index=False)
    if not kept: return log(f"[{name}] No kept features.")

    kept_fasta, clean_table = os.path.join(log_dir, "kept.fasta"), os.path.join(log_dir, "num.tsv")
    write_fasta(all_seqs, kept, kept_fasta)
    with open(clean_table, "w") as f: f.write("#OTU ID\t" + "\t".join(df_counts.columns) + "\n")
    df_counts.loc[kept].to_csv(clean_table, sep="\t", mode="a", header=False)

    biom, seq_qza, tab_qza = os.path.join(log_dir, "tab.biom"), os.path.join(log_dir, "seq.qza"), os.path.join(log_dir, "tab.qza")
    run_cmd(f"biom convert -i {clean_table} -o {biom} --to-hdf5 --table-type='OTU table'")
    for f in [seq_qza, tab_qza]: safe_rm(f)
    run_cmd(f"qiime tools import --input-path {kept_fasta} --output-path {seq_qza} --type 'FeatureData[Sequence]'")
    run_cmd(f"qiime tools import --input-path {biom} --output-path {tab_qza} --type 'FeatureTable[Frequency]'")

    step(f"GG2 Map: {name}")
    map_qza, reps_qza, tax_qza = [os.path.join(log_dir, f"gg2_{s}.qza") for s in ["map", "reps", "tax"]]
    for f in [map_qza, reps_qza, tax_qza]: safe_rm(f)
    run_cmd(f"qiime greengenes2 non-v4-16s --i-table {tab_qza} --i-sequences {seq_qza} --i-backbone {os.path.join(REF_DIR, REF_BACKBONE)} --p-perc-identity {GG2_PERC_IDENTITY} --o-mapped-table {map_qza} --o-representatives {reps_qza}")
    run_cmd(f"qiime greengenes2 taxonomy-from-table --i-reference-taxonomy {os.path.join(REF_DIR, REF_TAX_MAP)} --i-table {map_qza} --o-classification {tax_qza}")

    reps_dir, tax_dir = os.path.join(log_dir, "gg2_reps"), os.path.join(log_dir, "gg2_tax")
    os.makedirs(reps_dir, exist_ok=True); os.makedirs(tax_dir, exist_ok=True)
    run_cmd(f"qiime tools export --input-path {reps_qza} --output-path {reps_dir}")
    run_cmd(f"qiime tools export --input-path {tax_qza} --output-path {tax_dir}")
    
    gg2_tax_map = load_tax(os.path.join(tax_dir, "taxonomy.tsv"))
    gg2_uc = os.path.join(log_dir, "gg2.uc")
    run_cmd(f"vsearch --usearch_global {kept_fasta} --db {os.path.join(reps_dir, 'dna-sequences.fasta')} --id {VSEARCH_GG2_ID} --strand both --maxaccepts 10 --maxrejects 0 --uc {gg2_uc} --threads {THREADS} --quiet")
    
    gg2_df = parse_uc(gg2_uc, "GG2")
    gg2_map, gg2_uid_tax = {}, {}
    if not gg2_df.empty:
        gg2_df["UnifiedID"] = gg2_df["GG2_TargetID"].apply(lambda x: f"GG2:{x}")
        gg2_df["Taxon_GG2"] = gg2_df["GG2_TargetID"].map(gg2_tax_map).fillna("")
        gg2_map, gg2_uid_tax = dict(zip(gg2_df["Original_ZOTU"], gg2_df["UnifiedID"])), dict(zip(gg2_df["UnifiedID"], gg2_df["Taxon_GG2"]))

    step(f"SILVA Map: {name}")
    unmap_gg2 = [z for z in kept if z not in gg2_map]
    unmap_gg2_fa = os.path.join(log_dir, "unmapped_gg2.fasta")
    write_fasta(all_seqs, unmap_gg2, unmap_gg2_fa)
    
    silva_tax_map = silva_lca(unmap_gg2_fa, log_dir, "unmap_gg2") if unmap_gg2 else {}
    silva_uc, silva_ref_map, silva_uid_tax = os.path.join(log_dir, "silva.uc"), {}, {}
    
    if unmap_gg2 and os.path.exists(SILVA_FASTA):
        run_cmd(f"vsearch --usearch_global {unmap_gg2_fa} --db {SILVA_FASTA} --id {SILVA_IDENTITY} --iddef {SILVA_IDDEF} --strand both --maxaccepts 10 --maxrejects 0 --uc {silva_uc} --threads {THREADS} --quiet")
        silva_df = parse_uc(silva_uc, "SILVA")
        if not silva_df.empty:
            silva_df["UnifiedID"] = silva_df["SILVA_TargetID"].apply(lambda tid: f"SILVA:{tid}")
            silva_df["Taxon"] = silva_df["SILVA_TargetID"].map(load_tax(SILVA_TAX_TSV)).fillna("")
            silva_ref_map, silva_uid_tax = dict(zip(silva_df["Original_ZOTU"], silva_df["UnifiedID"])), dict(zip(silva_df["UnifiedID"], silva_df["Taxon"]))

    step(f"Final Merging & Cleanup: {name}")
    mapped, kept_from, tax_final = [], {}, {}
    for z in kept:
        if z in gg2_map:
            uid, kf, tax = gg2_map[z], "GG2", gg2_uid_tax.get(gg2_map[z], "")
        elif z in silva_ref_map:
            uid, kf, tax = silva_ref_map[z], "SILVA_REF", silva_uid_tax.get(silva_ref_map[z], silva_tax_map.get(z, ""))
        elif z in silva_tax_map and silva_tax_map[z].strip():
            uid, kf, tax = f"SILVATAX:{md5_seq(all_seqs.get(z,''))}", "SILVA_TAX", silva_tax_map[z]
        else:
            uid, kf, tax = f"NOVEL:{md5_seq(all_seqs.get(z,''))}", "NOVEL", ""
        mapped.append({"Original_ZOTU": z, "UnifiedID": uid})
        kept_from[z], tax_final[uid] = kf, tax

    df_out = df_counts.loc[kept].copy()
    z_to_u = dict(zip([m["Original_ZOTU"] for m in mapped], [m["UnifiedID"] for m in mapped]))
    df_out["UnifiedID"] = df_out.index.map(lambda x: z_to_u.get(str(x)))
    merged = df_out.groupby("UnifiedID").sum(numeric_only=True)
    merged.insert(0, "Taxon", pd.Series(tax_final).reindex(merged.index).fillna(""))
    
    # Save target table specific to project name
    final_tsv = os.path.join(outdir, f"{name}_unified_tax.tsv")
    merged.to_csv(final_tsv, sep="\t")

    # Cleanup heavy intermediates for this project
    run_cmd(f"rm -rf {log_dir}/*.qza {log_dir}/*.biom", f"clean-heavy-files-{name}")
    log(f"\n[{name}] Done. Kept: {len(kept)} | GG2: {len(gg2_map)} | SILVA_REF: {len(silva_ref_map)} | NOVEL: {list(kept_from.values()).count('NOVEL')}")
    log(f"[{name}] Saved to: {final_tsv}")

def main():
    args = parse_args()
    prep_silva()
    
    for proj in args.projects:
        name = os.path.basename(os.path.normpath(proj))
        table_path = os.path.join(proj, args.table)
        seqs_path = os.path.join(proj, args.seqs)
        
        if not os.path.exists(table_path) or not os.path.exists(seqs_path):
            log(f"\n[WARN] Skipping project '{name}': Missing '{table_path}' or '{seqs_path}'")
            continue
            
        try:
            print(f"\n\n=== Starting with Project: {name} ===\n")
            process(name, table_path, seqs_path, args.outdir)
            print(f"=== Finished Project: {name} ===\n\n")
        except Exception as e:
            log(f"\n[ERROR] Pipeline failed for project '{name}': {e}")

def parse_args():
    ap = argparse.ArgumentParser(description="GG2 + SILVA mapping script with multi-project loop processing.")
    ap.add_argument("-p", "--projects", nargs='+', required=True, help="List of project directories (e.g., PRJNAXXX/ PRJNAYYY/).")
    ap.add_argument("--table", required=False, default="ZOTUs-Table.tab", help="Name of the abundance table inside each project directory (e.g., ZOTUs-Table.tab).")
    ap.add_argument("--seqs", required=False, default="ZOTUs-Seqs.fasta", help="Name of the FASTA sequence file inside each project directory (e.g., ZOTUs-Seqs.fasta).")
    ap.add_argument("--outdir", required=False, default="tmp", help="Directory to save the final merged TSVs and temporary logs.")
    return ap.parse_args()

if __name__ == "__main__":
    try: main()
    except Exception as e: log(f"[ERROR] {e}"); sys.exit(1)
