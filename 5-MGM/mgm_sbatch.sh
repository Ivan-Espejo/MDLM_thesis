#!/bin/bash
#SBATCH --job-name=mgm_eval
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
# Argument Parsing from 01-master.sh
# ---------------------------------------------------------
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3
MODE=$4  # "loco" or "random"

echo "=== MGM Classifier Job Started | Mode: ${MODE^^} | Run: ${RUN_NAME} at $(date) ==="
echo "Allocated Node: $SLURMD_NODENAME"

# Setup Run-Specific Output Directories
TMP_SPLIT_DIR="./tmp_${MODE}_${RUN_NAME}"
MODEL_OUT_DIR="./output_${MODE}_${RUN_NAME}/finetuned_model"
PREDS_OUT_DIR="./output_${MODE}_${RUN_NAME}/predictions"
FINAL_DEST="../output/${MODE}/results/${RUN_NAME}/mgm/predictions"

mkdir -p ${TMP_SPLIT_DIR} ${MODEL_OUT_DIR} ${PREDS_OUT_DIR} "${FINAL_DEST}"

source ~/.bashrc
conda activate mgm_env

# ---------------------------------------------------------
# 1. Local Data Splitting
# ---------------------------------------------------------
echo "Step 1: Slicing MGM OTU table into Train/Test subsets for ${RUN_NAME}..."

python3 02.5-splitter.py \
    --mode "${MODE}" \
    -i output/preprocessing/MGM_Global/mgm_ready_abundances.csv \
    -l ../data/input/labels.csv \
    --train_txt "${TRAIN_FILE}" \
    --test_txt "${TEST_FILE}" \
    -o "${TMP_SPLIT_DIR}"


# ---------------------------------------------------------
# 2. MGM PIPELINE (PyTorch Fine-Tuning & Inference)
# ---------------------------------------------------------
cd ${TMP_SPLIT_DIR}

echo "Step 2: Constructing MicroCorpus objects..."
mgm construct \
    -i finetune_abundances.csv \
    -o finetune_corpus.pkl

mgm construct \
    -i test_abundances.csv \
    -o test_corpus.pkl

echo "Step 2.5: Generating MGM configuration file..."
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
# 3. Results Aggregation & Teardown
# ---------------------------------------------------------
echo "Step 5: Moving predictions to master architecture..."
cp ${PREDS_OUT_DIR}/y_score.csv "${FINAL_DEST}/Predictions_MGM.csv"
if [ -d "${PREDS_OUT_DIR}/figures" ]; then
    cp -r ${PREDS_OUT_DIR}/figures "${FINAL_DEST}/"
fi

# Clean up isolated temporary run folders
rm -rf ${TMP_SPLIT_DIR} ./output_${MODE}_${RUN_NAME}

END_TIME=$(date +%s)
ELAPSED_TOTAL=$((END_TIME - START_TIME))
echo "=========================================="
echo "=== ${RUN_NAME} (${MODE^^}) Job Finished at $(date) ==="
echo "Total Execution Time: $((ELAPSED_TOTAL / 60)) min and $((ELAPSED_TOTAL % 60)) sec."
echo "=========================================="
exit