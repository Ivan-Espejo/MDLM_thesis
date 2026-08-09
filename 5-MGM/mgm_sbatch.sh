#!/bin/bash
#SBATCH --job-name=mgm_loco
#SBATCH --partition=shared-gpu        
#SBATCH --qos=limitgpus
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/mgm_%j.out
#SBATCH --error=slurm_logs/mgm_%j.err

START_TIME=$(date +%s)

# ---------------------------------------------------------
# Argument Parsing from master_loco.sh
# ---------------------------------------------------------
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3

echo "=== MGM LOCO Job Started for ${RUN_NAME} at $(date) ==="
echo "Allocated Node: $SLURM_NODELIST"

# Setup Run-Specific Output Directories
TMP_SPLIT_DIR="./tmp_${RUN_NAME}"
MODEL_OUT_DIR="./output_${RUN_NAME}/finetuned_model"
PREDS_OUT_DIR="./output_${RUN_NAME}/predictions"
FINAL_DEST="../output/results/predictions/${RUN_NAME}/mgm"

mkdir -p ${TMP_SPLIT_DIR} ${MODEL_OUT_DIR} ${PREDS_OUT_DIR} "${FINAL_DEST}"

source /opt/mambaforge/etc/profile.d/conda.sh
conda activate 2-merge_env

# ---------------------------------------------------------
# Local Data Splitting
# ---------------------------------------------------------
echo "Step 1: Splitting Local MGM arrays into Train/Test subsets for ${RUN_NAME}..."

# Points directly to the centralized preprocessing directory inside 5-MGM/
python 02.5-splitter.py \
    -i output/preprocessing/MGM_Global/mgm_ready_abundances.csv \
    -l ../input/labels.csv \
    -m ../input/metadata_clean_2.txt \
    --train_txt "${TRAIN_FILE}" \
    --test_txt "${TEST_FILE}" \
    -o "${TMP_SPLIT_DIR}"


# ---------------------------------------------------------
# 3. MGM PIPELINE (Python / PyTorch)
# ---------------------------------------------------------
conda deactivate
conda activate mgm_env

# CD into temp directory to isolate PyTorch checkpoints per run
cd ${TMP_SPLIT_DIR}

echo "Step 2: Constructing MicroCorpus objects..."
mgm construct \
    -i finetune_abundances.csv \
    -o finetune_corpus.pkl

mgm construct \
    -i test_abundances.csv \
    -o test_corpus.pkl

echo "Step 2.5: Generating custom MGM configuration file with..."
echo """
[finetune]
learning_rate = 5e-5
warmup_steps = 100
weight_decay = 0.1
per_device_train_batch_size = 16
num_train_epochs = 100
logging_steps = 5
"""
cat <<EOF > mgm_config.ini
[finetune]
learning_rate = 5e-5
warmup_steps = 100
weight_decay = 0.1
per_device_train_batch_size = 16
num_train_epochs = 100
logging_steps = 5
EOF


echo "Step 3: Fine-tuning MGM on Training Set..."
mgm finetune \
    -i finetune_corpus.pkl \
    -l finetune_labels.csv \
    -o ../${MODEL_OUT_DIR} \
    -c mgm_config.ini \
    -s 0.1 \
    --seed 42


echo "Step 4: Predicting on Unseen Test Set..."
mgm predict -E \
    -i test_corpus.pkl \
    -l test_labels.csv \
    -m ../${MODEL_OUT_DIR} \
    -o ../${PREDS_OUT_DIR}

python3 ../03-results.py \
    -s ../${PREDS_OUT_DIR}/y_score.csv \
    -l test_labels.csv \
    -o ../${PREDS_OUT_DIR}/figures/

cd ..

# ---------------------------------------------------------
# Results Aggregation & Teardown
# ---------------------------------------------------------
conda deactivate

echo "Step 5: Moving predictions to master architecture..."
cp ${PREDS_OUT_DIR}/y_score.csv "${FINAL_DEST}/"

rm -rf ${TMP_SPLIT_DIR} ./output_${RUN_NAME}

END_TIME=$(date +%s)
ELAPSED_TOTAL=$((END_TIME - START_TIME))
echo "=========================================="
echo "=== ${RUN_NAME} Job Finished at $(date) ==="
echo "Total Execution Time: $((ELAPSED_TOTAL / 60)) min and $((ELAPSED_TOTAL % 60)) sec."
echo "=========================================="
exit