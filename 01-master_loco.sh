#!/bin/bash
# =============================================================================
# Script: 01-master_loco.sh
# Description: Global Master Orchestrator for Microbiome Machine Learning
# =============================================================================
# EXPLANATION & METHODOLOGY:
# -----------------------------------------------------------------------------
# 1. PURPOSE:
#    This master bash script dictates the execution of the entire multi-model 
#    microbiome pipeline (Alpha, mikropml, DeepMicro, MDLM, MGM). It manages 
#    global data splitting, global preprocessing, and automated SLURM dispatch.
#
# 2. DUAL ARCHITECTURE MODES (--mode):
#    A. LOCO Mode (Leave-One-Cohort-Out): [DEFAULT]
#       Tests cross-cohort generalization. The pipeline holds out entire studies
#       or geographical cohorts during training to strictly evaluate whether the
#       model learns biological truth or merely technical batch effects.
#
#    B. RANDOM Mode (Learning Curve Efficiency):
#       Tests data efficiency. Rather than cross-cohort validation, it performs 
#       stratified random splits incrementally (10%, 20%... 90% training data)
#       to generate learning curves and assess performance saturation.
#
# 3. GLOBAL PREPROCESSING:
#    To prevent redundant computation across 10+ folds, deep learning models 
#    (MDLM & MGM) perform their heavy lifting (e.g., FASTA extraction, 
#    BLAST alignments, GloVe embedding generation) once globally before 
#    dispatching parallel training jobs on the compute cluster.
#
# 4. USAGE EXAMPLES:
#    - Standard LOCO (All models):  bash 01-master_loco.sh --mode loco --runs 10
#    - Specific Models:              bash 01-master_loco.sh --mode loco --models MDLM MGM
#    - Learning Curves:              bash 01-master_loco.sh --mode random --models MIKROPML
# =============================================================================

set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
INPUT_DIR="$SCRIPT_DIR/data/input"

submit_job() {
    sbatch --export=ALL,PIPELINE_INPUT_DIR="$INPUT_DIR" "$@"
}

# ---------------------------------------------------------
# Helper Function: Print Usage
# ---------------------------------------------------------
print_intro() {
    echo "============================================================================="
    echo '      __  ___ ____   __    __  ___          ______          __            '
    echo '     /  |/  // __ \ / /   /  |/  / ___     /_  __/__  ___  / /_  ___      '
    echo '    / /|_/ // /_/ // /__ / /|_/ / |_-<      / / / -_ |_-< / __/ |_-<      '
    echo '   /_/  /_//_____//____//_/  /_/____/      /_/  \__/____/ \__/____/       '
    echo "                                                                          "
    echo "                    Microbiome ML Master Pipeline                       "
    echo "============================================================================="
}

print_usage() {
    echo "Usage: bash 01-master_loco.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message and exit."
    echo "  --mode, -m       Set pipeline mode. Options: 'loco' (default) or 'random'."
    echo "  --models, -mo    Space-separated list of models to execute."
    echo "                   Options: ALPHA MIKROPML DEEPMICRO MDLM MGM."
    echo "                   (Default: ALL models)."
    echo "  --input-dir      Directory containing labels.csv, raw_global_otu_dedup.tsv, and merged_seqs.fasta."
    echo "  --validate-only  Validate the inputs and exit without creating splits or submitting jobs."
    echo "  --runs, -r       Number of LOCO cross-validation folds to run (Default: 10)."
    echo ""
    echo "Examples:"
    echo "  bash 01-master_loco.sh --mode loco --runs 10"
    echo "  bash 01-master_loco.sh --mode loco --models MDLM MGM ALPHA"
    echo "  bash 01-master_loco.sh --mode random --models MIKROPML"
    echo "============================================================================="
}

# Default values
MODE="loco"
MODELS_TO_RUN=()
LOCO_RUNS=10
VALIDATE_ONLY=false

print_intro

# ---------------------------------------------------------
# Parse Arguments
# ---------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        --mode|-m)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "ERROR: --mode requires a value ('loco' or 'random')."
                exit 1
            fi
            MODE="${2,,}" # Convert to lowercase
            shift 2
            ;;
        --models|-mo)
            shift # Move past the flag
            # Consume everything until the next flag or end of arguments
            while [[ "$#" -gt 0 && ! "$1" =~ ^- ]]; do
                MODELS_TO_RUN+=("$1")
                shift
            done
            ;;
        --input-dir)
            if [[ -z "${2:-}" || "$2" =~ ^- ]]; then
                echo "ERROR: --input-dir requires a directory path." >&2
                exit 1
            fi
            INPUT_DIR="$2"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --runs|-r)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "ERROR: --runs requires a numeric value."
                exit 1
            fi
            LOCO_RUNS="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown parameter passed: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ "$INPUT_DIR" != /* ]]; then
    INPUT_DIR="$SCRIPT_DIR/$INPUT_DIR"
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR" >&2
    exit 1
fi
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
for required_file in labels.csv raw_global_otu_dedup.tsv merged_seqs.fasta; do
    if [[ ! -s "$INPUT_DIR/$required_file" ]]; then
        echo "ERROR: Required input file missing or empty: $INPUT_DIR/$required_file" >&2
        exit 1
    fi
done

# Validate Mode
if [[ "${MODE}" != "loco" && "${MODE}" != "random" ]]; then
    echo "ERROR: Invalid mode specified. Use --mode loco or --mode random."
    exit 1
fi
if [[ ! "$LOCO_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --runs must be a positive integer." >&2
    exit 1
fi

python3 bin/validate_master_inputs.py --input-dir "$INPUT_DIR" --mode "$MODE"
if [[ "$VALIDATE_ONLY" == true ]]; then
    echo "Input bundle is valid."
    exit 0
fi
INPUTS_SIGNATURE="$(cd "$INPUT_DIR" && sha256sum labels.csv raw_global_otu_dedup.tsv merged_seqs.fasta | sha256sum | cut -d' ' -f1)"

# Apply Defaults if no models provided
if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then
    echo "No models specified. Defaulting to ALL: ALPHA MIKROPML DEEPMICRO MDLM MGM"
    MODELS_TO_RUN=("ALPHA" "MIKROPML" "DEEPMICRO" "MDLM" "MGM")
fi

# Helper function to check if a model is in the execution list
run_model() {
    for m in "${MODELS_TO_RUN[@]}"; do
        if [ "${m^^}" == "${1^^}" ]; then return 0; fi
    done
    return 1
}

echo "========================================="
echo "Starting Master Pipeline..."
echo "Mode: ${MODE^^}"
if [ "${MODE}" == "loco" ]; then
    echo "LOCO Runs: ${LOCO_RUNS}"
fi
echo "Selected Models: ${MODELS_TO_RUN[*]}"
echo "========================================="

# ---------------------------------------------------------
# Phase 1: Global Setup & Data Splitting
# ---------------------------------------------------------
# We name the output directory based on the mode so they don't overwrite each other
SPLIT_DIR="./output/${MODE}/split_data_${MODE}"
SPLIT_INPUT_SIGNATURE="${SPLIT_DIR}/.input_signature"
SPLIT_CONFIG_FILE="${SPLIT_DIR}/.split_config"
SPLIT_CONFIG="${MODE}:${LOCO_RUNS}"

if compgen -G "${SPLIT_DIR}/run*" > /dev/null; then
    if [[ ! -f "$SPLIT_INPUT_SIGNATURE" || "$(<"$SPLIT_INPUT_SIGNATURE")" != "$INPUTS_SIGNATURE" || ! -f "$SPLIT_CONFIG_FILE" || "$(<"$SPLIT_CONFIG_FILE")" != "$SPLIT_CONFIG" ]]; then
        echo "ERROR: Existing splits in ${SPLIT_DIR} do not match the selected inputs/configuration." >&2
        echo "Remove that generated split directory before starting a new run." >&2
        exit 1
    fi
    echo "[Phase 1] Splits already exist in ${SPLIT_DIR}. Skipping generation..."
else
    echo "[Phase 1] Generating ${MODE^^} split combinations..."
    mkdir -p "${SPLIT_DIR}"
    
    if [ "${MODE}" == "loco" ]; then
        python3 bin/01-generate_loco_split.py \
        -l "$INPUT_DIR/labels.csv" \
        -o "${SPLIT_DIR}/" \
        -r "${LOCO_RUNS}"
        
    elif [ "${MODE}" == "random" ]; then
        # This python script will generate run10 to run90, representing 
        # the 10%, 20%... 90% training splits for the learning curve.
        python3 bin/01-generate_random_split.py \
        -l "$INPUT_DIR/labels.csv" \
        -o "${SPLIT_DIR}/"
    fi
fi

# Stop instead of submitting a literal wildcard when the labels cannot form splits.
shopt -s nullglob
RUN_DIRS=("${SPLIT_DIR}"/run*/)
shopt -u nullglob
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No valid splits were created in ${SPLIT_DIR}." >&2
    exit 1
fi
printf '%s\n' "$INPUTS_SIGNATURE" > "$SPLIT_INPUT_SIGNATURE"
printf '%s\n' "$SPLIT_CONFIG" > "$SPLIT_CONFIG_FILE"

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
        --table "$INPUT_DIR/raw_global_otu_dedup.tsv" \
        --fasta "$INPUT_DIR/merged_seqs.fasta" \
        --out_fasta ../output/preprocessing/MDLM_Global/global_V4.fasta \
        --out_tsv ../output/preprocessing/MDLM_Global/global_V4_otu.tsv \
        --primer GTGYCAGCMGCCGCGGTAA

        bash 2-blast.sh ../output/preprocessing/MDLM_Global/global_V4.fasta \
        ../data/references/GloVe_data/seqs_.07_embed.fasta

        python 3-generate_npys.py \
        --asv ../output/preprocessing/MDLM_Global/global_V4_otu.tsv \
        --blast output/preprocessing/tmp/best_hits.tsv \
        --labels "$INPUT_DIR/labels.csv" \
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
        source /opt/mambaforge/etc/profile.d/conda.sh
        
        conda activate vsearch_env
        cp "$INPUT_DIR/merged_seqs.fasta" output/preprocessing/MGM_Global/mgm_input.fasta

        bash 01-blast.sh output/preprocessing/MGM_Global/mgm_input.fasta \
        ../data/references/silva_132/SILVA_132_SSURef_NR99_DNA.fasta
        
        conda deactivate
        conda activate mgm_env

        python3 02-mgm_otu_table.py -i "$INPUT_DIR/raw_global_otu_dedup.tsv" \
        -m output/preprocessing/tmp/query_to_taxonomy.tsv \
        -o output/preprocessing/MGM_Global/mgm_ready_abundances.csv \
        -l output/preprocessing/MGM_Global/pipeline_drops.log
        
        rm -rf output/preprocessing/tmp/
        conda deactivate
        cd ..
    fi
fi

# ---------------------------------------------------------
# Phase 2C: Global DeepMicro Preprocessing (Autoencoder)
# ---------------------------------------------------------
DEEPMICRO_HALT_REQUIRED=false

if run_model "DEEPMICRO"; then
    if [ -f "./2-DeepMicro/DeepMicro/results/AE[20]_global_input_otu_rep.csv" ]; then
        echo "[Phase 2C] DeepMicro Global Embeddings found. Proceeding..."
    else
        echo "[Phase 2C] Missing Global DeepMicro Embeddings."
        echo "Dispatching 'global_deepmicro.sh' to the GPU node..."
        cd 2-DeepMicro/
        mkdir -p slurm_logs
        submit_job global_deepmicro.sh
        cd ..
        
        echo "======================================================="
        echo "NOTICE: The DeepMicro classifier cannot be run until the "
        echo "Autoencoder finishes generating embeddings on the GPU."
        echo "======================================================="
        
        DEEPMICRO_HALT_REQUIRED=true
    fi
fi


# ---------------------------------------------------------
# Phase 3: Slurm Dispatch Loop
# ---------------------------------------------------------
echo "[Phase 3] Dispatching parallel model training jobs..."

for run_path in "${RUN_DIRS[@]}"; do
    RUN_NAME=$(basename "${run_path}")
    
    echo "-----------------------------------------"
    echo "Dispatching models for Split: ${RUN_NAME} (${MODE^^})"
    
    # Conditionally grab the correct text files based on mode
    if [ "${MODE}" == "loco" ]; then
        TRAIN_FILE=$(realpath "${run_path}train_projects.txt")
        TEST_FILE=$(realpath "${run_path}test_projects.txt")
    else
        # For random efficiency mode, we pass sample lists instead of project lists
        TRAIN_FILE=$(realpath "${run_path}train_samples.txt")
        TEST_FILE=$(realpath "${run_path}test_samples.txt")
    fi
    
    if run_model "ALPHA"; then
        cd 0-alpha_diversity/
        mkdir -p slurm_logs
        submit_job alpha_sbatch.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}" "${MODE}"
        cd ..
    fi

    if run_model "MIKROPML"; then
        cd 1-mikropml/
        mkdir -p output/slurm_outputs
        submit_job mikropml_sbatch.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}" "${MODE}"
        cd ..
    fi

    if run_model "DEEPMICRO"; then
        if [ "$DEEPMICRO_HALT_REQUIRED" = true ]; then
            echo "[SKIPPED] DeepMicro classification skipped (waiting on GPU embeddings)."
        else
            cd 2-DeepMicro/
            mkdir -p slurm_logs
            submit_job deep_sbatch.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}" "${MODE}"
            cd ..
        fi
    fi

    if run_model "MDLM"; then
        cd 4-MDLM/
        mkdir -p slurm_logs
        submit_job mdlm_finetune.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}" "${MODE}"
        cd ..
    fi

    if run_model "MGM"; then
        cd 5-MGM/
        mkdir -p slurm_logs
        submit_job mgm_sbatch.sh "${TRAIN_FILE}" "${TEST_FILE}" "${RUN_NAME}" "${MODE}"
        cd ..
    fi
done

echo "========================================="
if [ "$DEEPMICRO_HALT_REQUIRED" = true ]; then
    echo "Pipeline partially dispatched."
    echo "DeepMicro was paused. Once the global_ae job finishes,"
    echo "re-run the next script to also dispatch the DeepMicro classifiers:"
    echo "bash 01-master_loco.sh --mode ${MODE} --models DEEPMICRO"
else
    echo "All requested ${MODE^^} folds successfully dispatched!"
fi
echo "Check queue status using 'squeue -u <username>'"
echo "========================================="
exit