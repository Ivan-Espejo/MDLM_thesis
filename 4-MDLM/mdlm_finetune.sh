#!/bin/bash
#SBATCH --job-name=mdlm_eval
#SBATCH --partition=shared-gpu        
#SBATCH --qos=limitgpus
#SBATCH --gres=gpu:1
#SBATCH --exclude=compms-gpu-1.exbio.wzw.tum.de
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/mdlm_%j.out
#SBATCH --error=slurm_logs/mdlm_%j.err

START_TIME=$(date +%s)

# ---------------------------------------------------------
# Argument Parsing from 01-master.sh
# ---------------------------------------------------------
TRAIN_FILE=$1
TEST_FILE=$2
RUN_NAME=$3
MODE=$4  # "loco" or "random"

echo "=== MDLM Classifier Job Started | Mode: ${MODE^^} | Run: ${RUN_NAME} ==="
echo "Allocated Node: $SLURM_NODELIST"

# Setup Run-Specific Sandboxed Output Directories
TMP_SPLIT_DIR="./tmp_${MODE}_${RUN_NAME}"
FINAL_DEST="../output/${MODE}/results/${RUN_NAME}/mdlm/predictions"
LOGS_DIR="../output/${MODE}/results/${RUN_NAME}/mdlm/logs"


mkdir -p ${TMP_SPLIT_DIR} ${MODEL_OUT_DIR} ${LOGS_DIR} ${FINAL_DEST}
touch ${LOGS_DIR}/finetune_metrics.txt

#################################
#### DATA SPLITTING ####
#################################

source ~/.bashrc
conda activate mdlm_gpu

echo "Step 1: Slicing Global .npy arrays based on ${MODE^^} definition..."
# We slice the global array specifically for this run.
# The python splitter now uses the labels.csv to track the index mapping.
python 3.5-splitter.py \
    --mode "${MODE}" \
    -i ../output/preprocessing/MDLM_Global/512_embed_otu.npy \
    -l ../output/preprocessing/MDLM_Global/label.npy \
    --labels_csv ../data/input/labels.csv \
    --train_txt "${TRAIN_FILE}" \
    --test_txt "${TEST_FILE}" \
    -o "${TMP_SPLIT_DIR}"

#################################
######## MDLM EXECUTION #########
#################################

echo "Step 2: Starting MDLM FINE TUNING for ${RUN_NAME}..."

cd microbiome_transformers/finetune_discriminator/

# Execute fine-tuning strictly on the temporary split arrays
python begin.py \
    --samples ../../${TMP_SPLIT_DIR}/finetune_samples.npy \
    --sample_labels ../../${TMP_SPLIT_DIR}/finetune_labels.npy \
    --vocab_path ../../data/references/GloVe_data/vocab_embeddings.npy \
    --output_path ../../${MODEL_OUT_DIR} \
    --log_file ../../${LOGS_DIR}/finetune_metrics \
    --batch_size 16 \
    --layers 5 \
    --epochs 30 \
    --attn_heads 5 \
    --seq_len 513 \
    --num_labels 2 \
    --freeze_opt 2 \
    --freeze_encoders 5 \
    --adam \
    --class_weights \
    --adam_weight_decay 0.01 \
    --lr 0.0001 \
    --cuda \
    --load_disc ../../data/models/MDLM_Models/ensemble/run10_epoch0_disc/pytorch_model.bin
    
cd ../../

# ---------------------------------------------------------
# Inference & Attribution
# ---------------------------------------------------------

echo "Step 3: Running Inference using Fine-Tuned Model on ${RUN_NAME} Test Set..."

# CRITICAL: Prevent HuggingFace from attempting internet connections on HPC nodes
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Execute Python in unbuffered mode to guarantee log output
python -u 4-Attribution_calculations.py \
    --data_path "${TMP_SPLIT_DIR}/test_samples.npy" \
    --label_path "${TMP_SPLIT_DIR}/test_labels.npy" \
    --sample_names_path "${TMP_SPLIT_DIR}/test_sample_names.txt" \
    --vocab_path ./data/references/GloVe_data/vocab_embeddings.npy \
    --models_path "${MODEL_OUT_DIR}" \
    --output_dir "${TMP_SPLIT_DIR}"

# ---------------------------------------------------------
# Results Aggregation
# ---------------------------------------------------------

echo "Step 4: Moving predictions to master architecture..."
# Verify the output was actually created before copying to prevent silent failures
if [ -f "${TMP_SPLIT_DIR}/sample_predictions.csv" ]; then
    cp "${TMP_SPLIT_DIR}/sample_predictions.csv" "${FINAL_DEST}/Predictions_MDLM.csv"
    echo "  [OK] Successfully copied Predictions_MDLM.csv"
else
    echo "  [ERROR] sample_predictions.csv was not generated! Halting cleanup to preserve logs."
    exit 1
fi

# Safely remove the massive temporary arrays and local logs to prevent disk clutter
rm -rf "${TMP_SPLIT_DIR}"

END_TIME=$(date +%s)
ELAPSED_TOTAL=$((END_TIME - START_TIME))
echo "=========================================="
echo "=== ${RUN_NAME} (${MODE^^}) Job Finished at $(date) ==="
echo "Total Execution Time: $((ELAPSED_TOTAL / 60)) min and $((ELAPSED_TOTAL % 60)) sec."
echo "=========================================="
exit