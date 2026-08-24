#!/bin/bash
#SBATCH --job-name=dm_eval
#SBATCH --partition=shared-cpu         # Switched to CPU pool since GPU work is done!
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # 8 CPU cores for fast caret tuning
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/dm_%j.log
#SBATCH --error=slurm_logs/dm_%j.err

mkdir -p slurm_logs

# ==============================================================================
# DEEPMICRO FAST CLASSIFIER 
# Autoencoder generation is handled globally. This script only performs 
# Cross-Validation / Classification on the generated embeddings.
# ==============================================================================

# Parse arguments from master.sh
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3
MODE=$4  # "loco" or "random"

echo "[1/3] Job initialized for ${RUN_NAME} (${MODE^^}) at: $(date)"
echo "      Running on node: $SLURMD_NODENAME"

# Define the final R output directory exactly as requested
FINAL_DEST="../output/${MODE}/results/${RUN_NAME}/deepmicro"
mkdir -p "${FINAL_DEST}"

# Initialize environment
source ~/.bashrc
conda activate mikropml

echo "[2/3] Executing Mikropml Classifier on global DeepMicro embeddings..."

# Pass the pre-computed global embeddings directly into the R classification script
$CONDA_PREFIX/bin/Rscript deep_mikropml.R \
    --mode "${MODE}" \
    --embeddings "../data/models/AE[20]_global_input_otu_rep.csv" \
    --train_file "${TRAIN_FILE}" \
    --test_file "${TEST_FILE}" \
    --labels "../data/input/labels.csv" \
    --output_dir "${FINAL_DEST}"

echo "[3/3] Pipeline complete."
echo "======================================================="
echo "[SUCCESS] DeepMicro ${MODE^^} pipeline for ${RUN_NAME} finished at: $(date)"
echo "Predictions successfully generated in ${FINAL_DEST}"
echo "======================================================="