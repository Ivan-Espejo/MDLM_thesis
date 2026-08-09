#!/bin/bash
#SBATCH --job-name=mikropml_eval
#SBATCH --partition=shared-cpu         # Uses the cluster's main CPU node pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # Request 8 CPU cores for parallel tuning
#SBATCH --mem=16G                      # 16G is plenty for N=66
#SBATCH --time=05:00:00
#SBATCH --output=output/slurm_outputs/mikropml_%j.log
#SBATCH --error=output/slurm_outputs/mikropml_%j.err

mkdir -p output/slurm_outputs

echo "=== Job Started at $(date) ==="
echo "Allocated Node: $SLURM_NODELIST"

# Read the arguments passed by master_loco.sh
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3

echo "Initiating Mikropml LOCO Pipeline for ${RUN_NAME}"

# Define temporary data directory for this run's split files
# We put it in /tmp or a local subfolder so runs don't overwrite each other
TMP_SPLIT_DIR="./tmp_${RUN_NAME}"
mkdir -p ${TMP_SPLIT_DIR}

# 1. Initialize Conda environment
source ~/.bashrc
conda activate mikropml

# 2. Slice the global data into Train/Test subsets using Python
echo "Executing Python Data Splitter..."
python3 1.5-mikropml_splitter.py \
    --global_otu ../input/raw_global_otu_dedup.tsv \
    --global_labels ../input/labels.csv \
    --meta ../input/metadata_clean_2.txt \
    --train_txt "${TRAIN_FILE}" \
    --test_txt "${TEST_FILE}" \
    --outdir "${TMP_SPLIT_DIR}"

# 3. Run the R script on the newly split data
# Note: --output_dir creates 'models', 'predictions', and 'metrics' subfolders automatically inside R
R_OUT_DIR="./output_${RUN_NAME}"

echo "Executing mikropml R pipeline..."
$CONDA_PREFIX/bin/Rscript lopo_execution.R \
  --train_otu "${TMP_SPLIT_DIR}/train_otu.tsv" \
  --train_labels "${TMP_SPLIT_DIR}/train_labels.csv" \
  --test_otu "${TMP_SPLIT_DIR}/test_otu.tsv" \
  --test_labels "${TMP_SPLIT_DIR}/test_labels.csv" \
  --output_dir "${R_OUT_DIR}"

# 4. Consolidate Results into the Master Directory
# Creating the final destination folder required by your architecture
FINAL_DEST="../output/results/predictions/${RUN_NAME}/mikropml"
mkdir -p "${FINAL_DEST}"

echo "Moving predictions to master structure: ${FINAL_DEST}"
# Copy just the prediction CSVs to the final benchmarking directory
cp ${R_OUT_DIR}/predictions/Predictions_*.csv "${FINAL_DEST}/"

# (Optional) Clean up temporary split data to save disk space
# rm -rf ${TMP_SPLIT_DIR}
echo "=== Job Finished at $(date) ==="