#!/bin/bash
#SBATCH --job-name=dm_global_ae
#SBATCH --partition=exbio-gpu
#SBATCH --nodelist=gpu01.exbio.wzw.tum.de
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/global_ae_%j.log
#SBATCH --error=slurm_logs/global_ae_%j.err

set -eo pipefail
INPUT_DIR="${PIPELINE_INPUT_DIR:-../data/input}"

mkdir -p slurm_logs

echo "[1/5] Starting Global DeepMicro Autoencoder Job at $(date)"

# Verify Base DeepMicro Installation
if [ ! -f ./DeepMicro/DM.py ]; then
    echo "[ERROR] DeepMicro/DM.py is missing. Initialize the pinned DeepMicro submodule before submitting this job." >&2
    exit 1
fi

mkdir -p DeepMicro/data/ DeepMicro/results/
source /opt/mambaforge/etc/profile.d/conda.sh

echo "[2/5] Formatting and normalizing global TSV into DeepMicro CSV..."
conda activate mikropml
$CONDA_PREFIX/bin/Rscript preprocessing.R \
    --input_otu "${INPUT_DIR}/raw_global_otu_dedup.tsv" \
    --output "DeepMicro/data/global_input_otu.csv"


echo "[3/5] Activating deep_env and executing DM.py on TITAN GPU..."
conda deactivate
conda activate deep_env

cd DeepMicro/
# Note: DM.py expects the raw matrix without Sample IDs
python DM.py --custom_data global_input_otu.csv \
            -s 2026 \
            -r 1 \
            --ae \
            -dm 20 \
            --save_rep \
            --no_clf

echo "[4/5] Restoring Sample ID metadata to embeddings..."
# DM.py outputs a headerless CSV of just embeddings.
# We securely paste our extracted Sample Names as the first column.
RAW_EMBEDS="./results/AE[20]_global_input_otu_rep.csv"
TMP_EMBEDS="./results/tmp_embeds.csv"

# paste puts the two files side-by-side separated by a comma
paste -d, ./data/global_input_samples.txt ${RAW_EMBEDS} > ${TMP_EMBEDS}

# Overwrite the original output with the fixed version
mv ${TMP_EMBEDS} ${RAW_EMBEDS}

cd ..

echo "[5/5] Global Embeddings successfully generated and mapped!"
echo "Target file ready for mikropml: 2-DeepMicro/DeepMicro/results/AE[20]_global_input_otu_rep.csv"