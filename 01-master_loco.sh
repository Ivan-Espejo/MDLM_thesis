#!/bin/bash

# Parse arguments to determine which models to execute
MODELS_TO_RUN=("$@")
if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then
    echo "No models specified. Defaulting to ALL: MIKROPML DEEPMICRO MDLM MGM"
    MODELS_TO_RUN=("MIKROPML" "DEEPMICRO" "MDLM" "MGM")
fi

# Helper function to check if a model is in the execution list
run_model() {
    for m in "${MODELS_TO_RUN[@]}"; do
        if [ "${m^^}" == "${1^^}" ]; then return 0; fi
    done
    return 1
}

echo "========================================="
echo "Starting Global LOCO Pipeline..."
echo "Selected Models: ${MODELS_TO_RUN[*]}"
echo "========================================="

# ---------------------------------------------------------
# Phase 1: Global Setup & Data Splitting
# ---------------------------------------------------------
if [ -d "./output/split_data/run1" ]; then
    echo "[Phase 1] Splits already exist in ./output/split_data/. Skipping..."
else
    echo "[Phase 1] Generating unique LOCO split combinations..."
    python3 01-generate_loco_split.py \
    -m ./input/metadata_clean_2.txt \
    -o ./output/split_data/ \
    -r 10
fi

# ---------------------------------------------------------
# Phase 2A: Global MDLM Preprocessing
# ---------------------------------------------------------
if run_model "MDLM"; then
    if [ -f "./output/preprocessing/MDLM_Global/512_embed_otu.npy" ]; then
        echo "[Phase 2A] MDLM Preprocessing files found. Skipping..."
    else
        echo "[Phase 2A] Executing Global MDLM Preprocessing..."
        cd 4-MDLM/
        mkdir -p ../output/preprocessing/MDLM_Global/
        source /opt/mambaforge/etc/profile.d/conda.sh
        conda activate 2-merge_env

        python 1-generate_fasta.py \
        --table ../input/raw_global_otu_dedup.tsv \
        --fasta ../input/merged_seqs.fasta \
        --out_fasta ../output/preprocessing/MDLM_Global/global_V4.fasta \
        --out_tsv ../output/preprocessing/MDLM_Global/global_V4_otu.tsv \
        --primer GTGYCAGCMGCCGCGGTAA

        bash 2-blast.sh ../output/preprocessing/MDLM_Global/global_V4.fasta \
        ../input/GloVe_data/seqs_.07_embed.fasta

        python 3-generate_npys.py \
        --asv ../output/preprocessing/MDLM_Global/global_V4_otu.tsv \
        --blast output/preprocessing/tmp/best_hits.tsv \
        --labels ../input/labels.csv \
        --fasta ../output/preprocessing/MDLM_Global/global_V4.fasta \
        --out_npy ../output/preprocessing/MDLM_Global/512_embed_otu.npy \
        --out_lbl ../output/preprocessing/MDLM_Global/label.npy \
        --out_tsv ../output/preprocessing/MDLM_Global/embed_otu.tsv \
        --out_map ../output/preprocessing/MDLM_Global/embed_seq_mapping.csv
        
        rm -rf output/preprocessing/tmp/
        conda deactivate
        cd ..
    fi
fi

# ---------------------------------------------------------
# Phase 2B: Global MGM Preprocessing
# ---------------------------------------------------------
if run_model "MGM"; then
    if [ -f "./5-MGM/output/preprocessing/MGM_Global/mgm_ready_abundances.csv" ]; then
        echo "[Phase 2B] MGM Preprocessing files found. Skipping..."
    else
        echo "[Phase 2B] Executing Global MGM Preprocessing..."
        cd 5-MGM/
        mkdir -p output/preprocessing/MGM_Global/ output/preprocessing/tmp/
        source ~/.bashrc
        
        conda activate vsearch_env
        cp ../input/merged_seqs.fasta output/preprocessing/MGM_Global/mgm_input.fasta

        bash 01-blast.sh output/preprocessing/MGM_Global/mgm_input.fasta \
        ./data/silva132/SILVA_132_SSURef_NR99_DNA.fasta
        
        conda deactivate
        conda activate mgm_env

        python3 02-mgm_otu_table.py -i ../input/raw_global_otu_dedup.tsv \
        -m output/preprocessing/tmp/query_to_taxonomy.tsv \
        -o output/preprocessing/MGM_Global/mgm_ready_abundances.csv \
        -l output/preprocessing/MGM_Global/pipeline_drops.log
        
        rm -rf output/preprocessing/tmp/
        conda deactivate
        cd ..
    fi
fi

# ---------------------------------------------------------
# Phase 3: Slurm Dispatch Loop
# ---------------------------------------------------------
echo "[Phase 3] Dispatching parallel model training jobs..."

for run_path in ./output/split_data/run*/ ; do
    RUN_NAME=$(basename ${run_path})
    
    echo "-----------------------------------------"
    echo "Dispatching models for Split: ${RUN_NAME}"
    
    TRAIN_FILE=$(realpath "${run_path}train_projects.txt")
    TEST_FILE=$(realpath "${run_path}test_projects.txt")
    
    if run_model "MIKROPML"; then
        cd 1-mikropml/
        sbatch mikropml_sbatch.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}"
        cd ..
    fi

    if run_model "DEEPMICRO"; then
        cd 2-DeepMicro/
        sbatch deep_sbatch.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}"
        cd ..
    fi

    if run_model "MDLM"; then
        cd 4-MDLM/
        sbatch mdlm_finetune.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}"
        cd ..
    fi

    if run_model "MGM"; then
        cd 5-MGM/
        sbatch mgm_sbatch.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}"
        cd ..
    fi
done

echo "========================================="
echo "All requested LOCO folds successfully dispatched!"
echo "Check queue status using 'squeue -u <username>'"
echo "========================================="
exit