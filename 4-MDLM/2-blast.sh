#!/bin/bash
# =============================================================================
# Script: 2-blast.sh
# Author: Ivan Espejo
#
# Description:
#     Builds or reuses a BLAST database from the MDLM embedding FASTA and maps
#     extracted V4 sequences to that reference vocabulary.
#
# Inputs:
#     Positional arguments: query V4 FASTA and reference embedding FASTA.
#     Defaults point to the generated global V4 FASTA and shared GloVe reference.
#
# Outputs:
#     output/preprocessing/tmp/best_hits.tsv for the NumPy-generation step.
#     BLAST database files beside the reference FASTA when first built.
# =============================================================================

source /opt/mambaforge/etc/profile.d/conda.sh
conda activate 4-blast_env

QUERY_FASTA="${1:-./output/preprocessing/global_V4.fasta}"
EMBED_FASTA="${2:-../data/references/GloVe_data/seqs_.07_embed.fasta}"

OUT_DIR="output/preprocessing/tmp"
RAW_HITS="$OUT_DIR/blast_hits.tsv"
BEST_HITS="$OUT_DIR/best_hits.tsv"
LOG_FILE="$OUT_DIR/pipeline_drops.log"

mkdir -p "$OUT_DIR"

echo "Building database..."
makeblastdb -in "$EMBED_FASTA" -dbtype nucl -out "$OUT_DIR/embed_db"

echo "Running alignment for $QUERY_FASTA with 98.5% identity and 100% coverage!"
echo "..."
blastn -db "$OUT_DIR/embed_db" \
       -query "$QUERY_FASTA" \
       -perc_identity 98.5 \
       -qcov_hsp_perc 100 \
       -outfmt "6 qseqid sseqid qseq sseq evalue bitscore" \
       -out "$RAW_HITS"

sort -t$'\t' -k1,1 -k6,6nr "$RAW_HITS" | awk -F'\t' '!seen[$1]++' > "$BEST_HITS"

TOTAL_SEQS=$(grep -c "^>" "$QUERY_FASTA")
ALIGNED_SEQS=$(wc -l < "$BEST_HITS")
UNALIGNED_SEQS=$((TOTAL_SEQS - ALIGNED_SEQS))
UNIQUE_EMBEDS=$(awk '{print $2}' "$BEST_HITS" | sort -u | wc -l)
DEDUPLICATIONS=$((ALIGNED_SEQS - UNIQUE_EMBEDS))

echo "=== Alignment Summary ==="
echo "1. Sequences before alignment: $TOTAL_SEQS"
echo "2. Sequences aligned: $ALIGNED_SEQS"
echo "3. Sequences merged into a single embed tag: $DEDUPLICATIONS"
echo "4. Sequences that could not be aligned: $UNALIGNED_SEQS"
echo "========================="

# --- LOGGING ADDITION ---
grep "^>" "$QUERY_FASTA" | sed 's/^>//' | sort > "$OUT_DIR/query_ids.txt"
awk '{print $1}' "$BEST_HITS" | sort > "$OUT_DIR/aligned_ids.txt"

echo -e "\n=== TAXA DROPPED (NO 100% EMBED ALIGNMENT) ===" >> "$LOG_FILE"
DROPPED=$(comm -23 "$OUT_DIR/query_ids.txt" "$OUT_DIR/aligned_ids.txt")

if [ -z "$DROPPED" ]; then
    echo "None" >> "$LOG_FILE"
else
    echo "$DROPPED" >> "$LOG_FILE"
fi

rm "$RAW_HITS" "$OUT_DIR"/embed_db.* "$OUT_DIR"/query_ids.txt "$OUT_DIR"/aligned_ids.txt

conda deactivate