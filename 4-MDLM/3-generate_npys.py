#!/usr/bin/env python3
"""
=============================================================================
Script: 5-generate_512_npys.py
Author: Ivan Espejo
Function: Filters ASV tables via BLAST hits, collapses taxa to Pope et al. 
          embedding IDs, applies TSS, and generates the final padded 3D 
          Numpy arrays (features) and 1D label arrays for the MDLM model.

Usage: 
    python 5-generate_512_npys.py [options]

Info: 
- Drops taxa without a valid BLAST hit (Out-of-Vocabulary filtering).
- Applies strict Total Sum Scaling (TSS = 1.0).
- Truncates and pads sequences with 0.0 to exactly 512 sequence length.
- Extracts binary labels directly mapped to the output.
- Generates a mapping CSV linking Embeddings -> Tags -> Taxonomy -> Sequence.
=============================================================================
"""


import argparse, sys, os
import pandas as pd
import numpy as np

def process(args):
    # 1. Load BLAST hits & setup dictionaries
    blast = pd.read_csv(args.blast, sep="\t", header=None, usecols=[0, 1], names=["Taxa", "Pope"])
    t2i = {r["Taxa"]: float(int(r["Pope"].replace("embed", "")) - 1) for _, r in blast.iterrows()}
    t2e = {r["Taxa"]: r["Pope"] for _, r in blast.iterrows()}

    # 2. Load ASV Table & Filter
    otu = pd.read_csv(args.asv, sep="\t", index_col=0)
    otu = otu.drop(columns=["Taxon"]) if "Taxon" in otu.columns else otu
    otu_mapped = otu[otu.index.isin(t2i.keys())].copy()
    if otu_mapped.empty: sys.exit("[!] Critical: No taxa matched the BLAST dictionary.")

    # 3. Merge Abundances & Apply Total Sum Scaling (TSS = 1.0)
    otu_mapped.index = otu_mapped.index.map(t2i)
    otu_raw = otu_mapped.groupby(otu_mapped.index).sum()
    otu_tss = (otu_raw / otu_raw.sum(axis=0).replace(0, 1)) * args.tss

    # 4. Truncate to 512 & Pad Arrays with 0.0
    otu_t, mask_t = otu_tss.T, (otu_raw > 0).T
    samples = otu_t.index.tolist()
    
    d_512 = np.zeros((len(samples), args.seq_len, 2), dtype=np.float32)
    base_df = pd.DataFrame(np.nan, index=otu_tss.index, columns=otu_tss.columns)

    for i, s in enumerate(samples):
        s_tss = otu_t.loc[s]
        pairs = sorted([[p, ab] for p, ab in s_tss[mask_t.loc[s]].items()], key=lambda x: x[1], reverse=True)[:args.seq_len]
        
        for j in range(args.seq_len):
            if j < len(pairs):
                d_512[i, j, 0], d_512[i, j, 1] = pairs[j]
                base_df.at[pairs[j][0], s] = pairs[j][1]
            else:
                d_512[i, j, 0], d_512[i, j, 1] = 0.0, 0.0
        base_df[s] = base_df[s].fillna(0.0)

    # 5. Extract Labels ensuring perfect row-by-row alignment with sample order
    lbl_df = pd.read_csv(args.labels, header=0)
    l_dict = dict(zip(lbl_df.iloc[:, 0].astype(str), lbl_df.iloc[:, 1].astype(int)))
    
    labels_list = []
    for s in samples:
        if s not in l_dict: sys.exit(f"[!] Error: Sample '{s}' missing in metadata labels.")
        labels_list.append(l_dict[s])
    labels = np.array(labels_list, dtype=np.int32)

    # 6. Generate Embed -> Tag -> Sequence -> Taxonomy Mapping
    seq_map = []
    if os.path.exists(args.fasta):
        seqs_data = {}
        with open(args.fasta, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    parts = line[1:].strip().split(' ', 1)
                    tag = parts[0]
                    tax = parts[1] if len(parts) > 1 else "Unknown"
                    seqs_data[tag] = (next(f).strip(), tax)
                    
        for tag, emb in t2e.items():
            seq, tax = seqs_data.get(tag, ("Seq_Not_Found", "Tax_Not_Found"))
            seq_map.append({
                "Embedding": emb, 
                "Tag": tag, 
                "Sequence": seq,
                "Taxonomy": tax
            })
        pd.DataFrame(seq_map).to_csv(args.out_map, index=False)

    # 7. Write Outputs
    os.makedirs(os.path.dirname(args.out_npy) or '.', exist_ok=True)
    np.save(args.out_npy, d_512)
    np.save(args.out_lbl, labels)
    
    i2e = {v: t2e[k] for k, v in t2i.items()}
    df_emb = base_df.copy()
    df_emb.index = df_emb.index.map(i2e)
    df_emb.dropna(how='all').to_csv(args.out_tsv, sep="\t")

    print(f"[+] Output Matrices Generated:\n -> NPY Features: {d_512.shape}\n -> NPY Labels: {labels.shape}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--asv", default="output/preprocessing/global_V4_otu.tsv")
    p.add_argument("--blast", default="output/preprocessing/tmp/best_hits.tsv")
    p.add_argument("--labels", default="input/labels.csv")
    p.add_argument("--fasta", default="output/preprocessing/global_V4.fasta")
    p.add_argument("--out_npy", default="output/preprocessing/512_embed_otu.npy")
    p.add_argument("--out_lbl", default="output/preprocessing/label.npy")
    p.add_argument("--out_tsv", default="output/preprocessing/embed_otu.tsv")
    p.add_argument("--out_map", default="output/preprocessing/embed_seq_mapping.csv")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--tss", type=float, default=1.0)
    return p.parse_args()

if __name__ == "__main__":
    process(parse_args())