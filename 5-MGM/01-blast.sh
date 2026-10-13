#!/bin/bash
# =============================================================================
# Script: 01-blast.sh
# Author: Ivan Espejo
#
# Description:
#     Aligns each query sequence to SILVA 132 using VSEARCH, retains the best
#     hit, and extracts its taxonomy from the reference FASTA header.
#
# Inputs:
#     Positional arguments: query FASTA and SILVA 132 reference FASTA.
#     Defaults use the MGM global query FASTA and data/references/silva_132/.
#
# Outputs:
#     output/preprocessing/tmp/query_to_taxonomy.tsv for 02-mgm_otu_table.py.
#     Temporary hit tables and dropped-sequence log in the same tmp folder.
# =============================================================================

source /opt/mambaforge/etc/profile.d/conda.sh
conda activate vsearch_env

QUERY_FASTA="${1:-./output/preprocessing/MGM_Global/mgm_input.fasta}"
SILVA_FASTA="${2:-../data/references/silva_132/SILVA_132_SSURef_NR99_DNA.fasta}"

OUT_DIR="output/preprocessing/tmp"
RAW_HITS="$OUT_DIR/vsearch_hits.tsv"
BEST_HITS="$OUT_DIR/best_hits.tsv"
LOG_FILE="$OUT_DIR/pipeline_drops.log"

mkdir -p "$OUT_DIR"

echo "Running global alignment for $QUERY_FASTA with 98.5% identity..."
vsearch --usearch_global "$QUERY_FASTA" \
        --db "$SILVA_FASTA" \
        --id 0.985 \
        --blast6out "$RAW_HITS" \
        --threads 8

# VSEARCH blast6out format: query, target, id, alnlen, mism, opens, qlo, qhi, tlo, thi, evalue, bits
sort -t$'\t' -k1,1 -k12,12nr "$RAW_HITS" | awk -F'\t' '!seen[$1]++' > "$BEST_HITS"

TOTAL_SEQS=$(grep -c "^>" "$QUERY_FASTA")
ALIGNED_SEQS=$(wc -l < "$BEST_HITS")
UNALIGNED_SEQS=$((TOTAL_SEQS - ALIGNED_SEQS))
UNIQUE_HITS=$(awk '{print $2}' "$BEST_HITS" | sort -u | wc -l)
DEDUPLICATIONS=$((ALIGNED_SEQS - UNIQUE_HITS))

echo "=== Alignment Summary ==="
echo "1. Sequences before alignment: $TOTAL_SEQS"
echo "2. Sequences aligned: $ALIGNED_SEQS"
echo "3. Sequences merged into a single SILVA tag: $DEDUPLICATIONS"
echo "4. Sequences that could not be aligned: $UNALIGNED_SEQS"
echo "========================="

grep "^>" "$QUERY_FASTA" | sed 's/^>//' | sort > "$OUT_DIR/query_ids.txt"
awk '{print $1}' "$BEST_HITS" | sort > "$OUT_DIR/aligned_ids.txt"

echo -e "\n=== TAXA DROPPED (NO VSEARCH ALIGNMENT) ===" >> "$LOG_FILE"
DROPPED=$(comm -23 "$OUT_DIR/query_ids.txt" "$OUT_DIR/aligned_ids.txt")

if [ -z "$DROPPED" ]; then
    echo "None" >> "$LOG_FILE"
else
    echo "$DROPPED" >> "$LOG_FILE"
fi

rm "$RAW_HITS" "$OUT_DIR"/query_ids.txt "$OUT_DIR"/aligned_ids.txt


awk '
NR==FNR { 
    # File 1: best_hits.tsv (extract Query ID and SILVA ID)
    query_to_silva[$1] = $2; 
    needed_silva[$2] = 1; 
    next 
}
/^>/ {
    # File 2: SILVA FASTA (extract ID and match)
    id = $1; 
    sub(/^>/, "", id);
    if (id in needed_silva) {
        # Extract everything after the first space as taxonomy
        tax = $0; 
        sub(/^>[^ ]+ /, "", tax);
        silva_to_tax[id] = tax;
    }
}
END {
    print "Query_ID\tSILVA_Taxonomy"
    for (q in query_to_silva) {
        s = query_to_silva[q];
        if (s in silva_to_tax) {
            print q "\t" silva_to_tax[s];
        }
    }
}' output/preprocessing/tmp/best_hits.tsv "$SILVA_FASTA" > output/preprocessing/tmp/query_to_taxonomy.tsv

echo "Extracting taxonomic lineages for the aligned hits..."
MAPPING_FILE="$OUT_DIR/query_to_taxonomy.tsv"

# AWK Pass 1: Load target hits into memory
# AWK Pass 2: Scan SILVA FASTA for those specific IDs and extract the taxonomy string
awk '
NR==FNR { 
    needed_silva[$2] = 1; 
    query_to_silva[$1] = $2; 
    next 
}
/^>/ {
    id = $1; sub(/^>/, "", id);
    if (id in needed_silva) {
        tax = $0; sub(/^>[^ ]+ /, "", tax);
        silva_to_tax[id] = tax;
    }
}
END {
    print "Query_ID\tSILVA_Taxonomy"
    for (q in query_to_silva) {
        s = query_to_silva[q];
        if (s in silva_to_tax) {
            print q "\t" silva_to_tax[s];
        }
    }
}' "$BEST_HITS" "$SILVA_FASTA" > "$MAPPING_FILE"

echo "Taxonomy mapping file generated at: $MAPPING_FILE"


conda deactivate