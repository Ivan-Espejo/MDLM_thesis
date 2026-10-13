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

# =============================================================================
# Script: deep_sbatch.sh
# Author: Ivan Espejo
#
# Description:
#     Slurm wrapper for one DeepMicro embedding-classification fold. Uses the
#     globally generated autoencoder embeddings and delegates classification to
#     deep_mikropml.R.
#
# Inputs:
#     Positional arguments: train-ID file, test-ID file, run name, and mode.
#     DeepMicro/results/AE[20]_global_input_otu_rep.csv and labels.csv under
#     PIPELINE_INPUT_DIR (default: ../data/input).
#
# Outputs:
#     ../output/<mode>/results/<run>/deepmicro/: predictions and metrics.
#     slurm_logs/: Slurm standard output and error logs.
# =============================================================================

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
INPUT_DIR="${PIPELINE_INPUT_DIR:-../data/input}"

echo "[1/3] Job initialized for ${RUN_NAME} (${MODE^^}) at: $(date)"
echo "      Running on node: $SLURMD_NODENAME"

# Define the final R output directory exactly as requested
FINAL_DEST="../output/${MODE}/results/${RUN_NAME}/deepmicro"
mkdir -p "${FINAL_DEST}"

# Initialize environment
source /opt/mambaforge/etc/profile.d/conda.sh
conda activate mikropml

echo "[2/3] Executing Mikropml Classifier on global DeepMicro embeddings..."

# Pass the pre-computed global embeddings directly into the R classification script
$CONDA_PREFIX/bin/Rscript deep_mikropml.R \
    --mode "${MODE}" \
    --embeddings "./DeepMicro/results/AE[20]_global_input_otu_rep.csv" \
    --train_file "${TRAIN_FILE}" \
    --test_file "${TEST_FILE}" \
    --labels "${INPUT_DIR}/labels.csv" \
    --output_dir "${FINAL_DEST}"

echo "[3/3] Pipeline complete."
echo "======================================================="
echo "[SUCCESS] DeepMicro ${MODE^^} pipeline for ${RUN_NAME} finished at: $(date)"
echo "Predictions successfully generated in ${FINAL_DEST}"
echo "======================================================="