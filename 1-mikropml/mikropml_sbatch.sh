#!/bin/bash
#SBATCH --job-name=mikropml_eval
#SBATCH --partition=shared-cpu         # Uses the cluster's main CPU node pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # Request 8 CPU cores for parallel tuning
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=output/slurm_outputs/mikropml_%j.log
#SBATCH --error=output/slurm_outputs/mikropml_%j.err

echo "=== Job Started at $(date) ==="
echo "Allocated Node: $SLURM_NODELIST"

# Read the arguments passed by 01-master.sh
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3
MODE=$4  # "loco" or "random"

mkdir -p output/slurm_outputs

echo "Initiating Mikropml Pipeline | Mode: ${MODE^^} | Run: ${RUN_NAME}"

# 1. Initialize Conda environment
source ~/.bashrc
conda activate mikropml

# 2. Define structured output directories
R_OUT_DIR="../output/${MODE}/results/${RUN_NAME}/mikropml"
mkdir -p "${R_OUT_DIR}"

# 3. Execute the dual-mode R script directly on global data
echo "Executing mikropml R pipeline..."
$CONDA_PREFIX/bin/Rscript lopo_execution.R \
  --mode "${MODE}" \
  --train_file "${TRAIN_FILE}" \
  --test_file "${TEST_FILE}" \
  --otu "../data/input/raw_global_otu_dedup.tsv" \
  --labels "../data/input/labels.csv" \
  --output_dir "${R_OUT_DIR}"

echo "=== Job Finished at $(date) ==="