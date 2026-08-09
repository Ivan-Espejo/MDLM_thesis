#!/bin/bash
#SBATCH --job-name=dm_loco_pipeline
#SBATCH --partition=exbio-gpu
#SBATCH --nodelist=gpu01.exbio.wzw.tum.de
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/slurm_%j.log
#SBATCH --error=slurm_logs/slurm_%j.err

# ==============================================================================
# HARDWARE & COMPATIBILITY NOTES (CUDA 10 / TITAN REQUIRED)
# ==============================================================================

# Parse arguments from master_loco.sh
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3

echo "[1/8] Job initialized for ${RUN_NAME} at: $(date)"
echo "[2/8] Running on node: $SLURMD_NODENAME"

# Define temporary execution directories to prevent parallel Slurm jobs from crashing into each other
TMP_SPLIT_DIR="./tmp_${RUN_NAME}"
DM_RUN_DIR="./DeepMicro_${RUN_NAME}"
R_OUT_DIR="./output_${RUN_NAME}"
FINAL_DEST="../output/results/predictions/${RUN_NAME}/deepmicro"

mkdir -p ${TMP_SPLIT_DIR} ${R_OUT_DIR} "${FINAL_DEST}"

# Verify Base DeepMicro Installation Exists
if [ ! -d ./DeepMicro/ ] ; then
    echo "[WARNING] Base DeepMicro directory not found. Cloning repository..."
    git clone https://github.com/minoh0201/DeepMicro
fi

# Clone the DeepMicro folder for this specific run to prevent parallel overwriting
cp -r ./DeepMicro/ ${DM_RUN_DIR}/

# Creating necessary subdirectories for DeepMicro to run properly
mkdir -p ${DM_RUN_DIR}/data/
mkdir -p ${DM_RUN_DIR}/results/

# Initialize environment
source ~/.bashrc
conda activate mikropml

echo "[3/8] Slicing data for ${RUN_NAME} using the unified Python splitter..."
# Reuse the exact same python splitter to guarantee identical data splits across models
python3 2.5-deep_splitter.py \
    --global_otu ../input/raw_global_otu_dedup.tsv \
    --global_labels ../input/labels.csv \
    --meta ../input/metadata_clean_2.txt \
    --train_txt "${TRAIN_FILE}" \
    --test_txt "${TEST_FILE}" \
    --outdir "${TMP_SPLIT_DIR}"

# Switch to the deep learning environment
conda deactivate
conda activate deep_env

echo "[4/8] Running R preprocessing for DeepMicro format..."
# Convert the beautifully sliced TSVs into the single custom CSV that DeepMicro requires
$CONDA_PREFIX/bin/Rscript preprocessing.R \
    --train_otu "${TMP_SPLIT_DIR}/train_otu.tsv" \
    --test_otu "${TMP_SPLIT_DIR}/test_otu.tsv" \
    --output "${TMP_SPLIT_DIR}/deep_input_otu.csv"


echo "[5/8] Initiating DeepMicro Autoencoder pipeline..."

# Copying to directory of data for DM.py
cp ${TMP_SPLIT_DIR}/deep_input_otu.csv ${DM_RUN_DIR}/data/deep_input_otu.csv

cd ${DM_RUN_DIR}

python DM.py --custom_data deep_input_otu.csv \
            -s 2026 \
            -r 1 \
            --ae \
            -dm 20 \
            --save_rep \
            --no_clf

cd ..

echo "[6/8] DeepMicro completed. Extracting latent representations..."
# Move the embeddings safely back to our run's isolated temp folder
cp ${DM_RUN_DIR}/results/AE[20]_deep_input_otu_rep.csv ${TMP_SPLIT_DIR}/


echo "[7/8] Activating mikropml environment for classification..."
conda deactivate
conda activate mikropml

# Run the R classification script, passing the newly generated AE embeddings and the isolated labels
$CONDA_PREFIX/bin/Rscript mikropml.R \
    --embeddings "${TMP_SPLIT_DIR}/AE[20]_deep_input_otu_rep.csv" \
    --train_labels "${TMP_SPLIT_DIR}/train_labels.csv" \
    --test_labels "${TMP_SPLIT_DIR}/test_labels.csv" \
    --output_dir "${R_OUT_DIR}"


echo "[8/8] Pipeline complete. Moving predictions to master architecture..."
# Move just the prediction CSVs to the final benchmarking directory
cp ${R_OUT_DIR}/predictions/Predictions_*.csv "${FINAL_DEST}/"

# Clean up temporary execution files to save disk space
rm -rf ${TMP_SPLIT_DIR} ${DM_RUN_DIR} 

echo "======================================================="
echo "[SUCCESS] DeepMicro LOCO folds for ${RUN_NAME} finished at: $(date)"
echo "Predictions successfully aggregated in ${FINAL_DEST}"
echo "======================================================="