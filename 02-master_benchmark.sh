#!/bin/bash
#SBATCH --job-name=benchmark_eval
#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=output/slurm_logs/benchmark_%j.log
#SBATCH --error=output/slurm_logs/benchmark_%j.err

START_TIME=$(date +%s)
mkdir -p output/slurm_logs

# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
MODE="loco" # Default mode
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode|-m)
            MODE="${2,,}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--mode loco|random]"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [[ "${MODE}" != "loco" && "${MODE}" != "random" ]]; then
    echo "ERROR: Invalid mode specified. Use --mode loco or --mode random."
    exit 1
fi

echo "=== Benchmarking Job Started | Mode: ${MODE^^} at $(date) ==="
echo "Allocated Node: $SLURMD_NODENAME"

source /opt/mambaforge/etc/profile.d/conda.sh
conda activate benchmark_env

# ---------------------------------------------------------
# Paths Setup
# ---------------------------------------------------------
RESULTS_DIR="./output/${MODE}/results"
BENCHMARK_DIR="./output/${MODE}/benchmark"
GLOBAL_DIR="${BENCHMARK_DIR}/global_stability"

mkdir -p "${BENCHMARK_DIR}" "${GLOBAL_DIR}"

echo "========================================="
echo "Starting Parallel Benchmarking Pipeline..."
echo "Targeting: ${RESULTS_DIR}"
echo "========================================="

# Enable nullglob to handle empty directories safely
shopt -s nullglob
RUN_DIRS=("${RESULTS_DIR}"/run*/)
shopt -u nullglob

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No run directories found in ${RESULTS_DIR}/"
    exit 1
fi

# ---------------------------------------------------------
# Step 1: Individual Run Benchmarking
# ---------------------------------------------------------
for run_path in "${RUN_DIRS[@]}"; do
    RUN_NAME=$(basename "${run_path}")
    
    echo "-----------------------------------------"
    echo "Dispatching Benchmark for Split: ${RUN_NAME}"
    
    RUN_OUT_DIR="${BENCHMARK_DIR}/${RUN_NAME}"
    mkdir -p "${RUN_OUT_DIR}"

    # The python script will now search inside the run folder to find the nested model CSVs
    python3 bin/02-benchmarking.py \
        --input_dir "${run_path}" \
        --labels data/input/labels.csv \
        --outdir "${RUN_OUT_DIR}" \
        --mode "${MODE}"
done

echo "-----------------------------------------"
echo "All Individual Benchmarks Completed."
echo "Executing Global Stability Aggregation..."
echo "-----------------------------------------"

# ---------------------------------------------------------
# Step 2: Global Stability Aggregation
# ---------------------------------------------------------
# Run the stability script targeting the main benchmark folder
python3 bin/03-benchmarking_stability.py \
    --mode "${MODE}" \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --outdir "${GLOBAL_DIR}"

# ---------------------------------------------------------
# Step 3: Cohort Presence Impact Analysis (LOCO Only)
# ---------------------------------------------------------
if [ "${MODE}" == "loco" ]; then
    echo "-----------------------------------------"
    echo "Executing Cohort Allocation Impact Analysis..."
    echo "-----------------------------------------"

    python3 bin/04-fold_comparison.py \
        --benchmark_root_dir "${BENCHMARK_DIR}" \
        --split_data_dir "./output/${MODE}/split_data_${MODE}" \
        --global_out_dir "${GLOBAL_DIR}" \
        --cohort_of_interest "PRJNA1259137" \
        --metric "ROC_AUC"

    echo "Skipping Learning Curves (only applicable to RANDOM mode)."
else
    echo "Skipping Cohort Presence Analysis (only applicable to LOCO mode)."

    echo "-----------------------------------------"
    echo "Executing Learning Curve Generation..."
    echo "-----------------------------------------"

    python3 bin/05-learning_curve.py \
        --benchmark_root_dir "${BENCHMARK_DIR}" \
        --global_out_dir "${GLOBAL_DIR}"

fi

echo "Pipeline Finished Successfully!"

conda deactivate

END_TIME=$(date +%s)
ELAPSED_TOTAL=$((END_TIME - START_TIME))
echo "=========================================="
echo "=== ${MODE^^} Benchmarking Finished at $(date) ==="
echo "Total Execution Time: $((ELAPSED_TOTAL / 60)) min and $((ELAPSED_TOTAL % 60)) sec."
echo "=========================================="
exit