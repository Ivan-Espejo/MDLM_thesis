#!/bin/bash
#SBATCH --job-name=alpha_eval
#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=slurm_logs/alpha_%j.log
#SBATCH --error=slurm_logs/alpha_%j.err

echo "=== Job Started at $(date) ==="
echo "Allocated Node: $SLURMD_NODENAME"

# Read the arguments passed by 01-master.sh
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3
MODE=$4  # "loco" or "random"

mkdir -p slurm_logs

echo "Initiating Alpha Diversity Baseline | Mode: ${MODE^^} | Run: ${RUN_NAME}"

# 1. Initialize Conda environment
source ~/.bashrc
conda activate mikropml

# 2. Define structured output directories exactly matching the master architecture
R_OUT_DIR="../output/${MODE}/results/${RUN_NAME}/alpha"
mkdir -p "${R_OUT_DIR}"

# 3. Execute the dual-mode R script directly on global data
echo "Executing Alpha Diversity R pipeline..."
$CONDA_PREFIX/bin/Rscript lopo_execution_alpha.R \
  --mode "${MODE}" \
  --train_file "${TRAIN_FILE}" \
  --test_file "${TEST_FILE}" \
  --otu "../data/input/raw_global_otu_dedup.tsv" \
  --labels "../data/input/labels.csv" \
  --output_dir "${R_OUT_DIR}"

echo "=== Job Finished at $(date) ==="