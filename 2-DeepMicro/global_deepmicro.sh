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

mkdir -p slurm_logs

echo "[1/5] Starting Global DeepMicro Autoencoder Job at $(date)"

# Verify Base DeepMicro Installation
if [ ! -d ./DeepMicro/ ] ; then
    echo "[WARNING] Base DeepMicro directory not found. Cloning repository..."
    git clone https://github.com/minoh0201/DeepMicro
fi

mkdir -p ../data/models/ DeepMicro/data/ DeepMicro/results/
source ~/.bashrc

echo "[2/5] Formatting and normalizing global TSV into DeepMicro CSV..."
conda activate mikropml
$CONDA_PREFIX/bin/Rscript preprocessing.R \
    --input_otu "../data/input/raw_global_otu_dedup.tsv" \
    --output "../data/models/global_input_otu.csv"


echo "[3/5] Activating deep_env and executing DM.py on TITAN GPU..."
conda deactivate
conda activate deep_env

cd DeepMicro/
# Note: DM.py expects the raw matrix without Sample IDs
ln -sfn ../../data/models/global_input_otu.csv DeepMicro/data/global_input_otu.csv
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
RAW_EMBEDS="../../data/models/AE[20]_global_input_otu_rep.csv"
TMP_EMBEDS="../../data/models/tmp_embeds.csv"

mv ./results/AE[20]_global_input_otu_rep.csv "${RAW_EMBEDS}"

# paste puts the two files side-by-side separated by a comma
paste -d, ../../data/models/global_input_samples.txt ${RAW_EMBEDS} > ${TMP_EMBEDS}

# Overwrite the original output with the fixed version
mv ${TMP_EMBEDS} ${RAW_EMBEDS}

cd ..

echo "[5/5] Global Embeddings successfully generated and mapped!"
echo "Target file ready for mikropml: data/models/AE[20]_global_input_otu_rep.csv"