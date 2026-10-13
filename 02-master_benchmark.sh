#!/bin/bash
#SBATCH --job-name=benchmark_eval
#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err

set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
INPUT_DIR="$SCRIPT_DIR/data/input"

START_TIME=$(date +%s)
mkdir -p output/slurm_logs

# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
MODE="loco" # Default mode
VALIDATE_ONLY=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode|-m)
            if [[ -z "${2:-}" || "$2" =~ ^- ]]; then
                echo "ERROR: --mode requires 'loco' or 'random'." >&2
                exit 1
            fi
            MODE="${2,,}"
            shift 2
            ;;
        --input-dir)
            if [[ -z "${2:-}" || "$2" =~ ^- ]]; then
                echo "ERROR: --input-dir requires a directory path." >&2
                exit 1
            fi
            INPUT_DIR="$2"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
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

if [[ "$INPUT_DIR" != /* ]]; then
    INPUT_DIR="$SCRIPT_DIR/$INPUT_DIR"
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR" >&2
    exit 1
fi
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
if [[ ! -s "$INPUT_DIR/labels.csv" ]]; then
    echo "ERROR: Required labels file missing or empty: $INPUT_DIR/labels.csv" >&2
    exit 1
fi

if [[ "${MODE}" != "loco" && "${MODE}" != "random" ]]; then
    echo "ERROR: Invalid mode specified. Use --mode loco or --mode random."
    exit 1
fi

python3 bin/validate_master_inputs.py --input-dir "$INPUT_DIR" --mode "$MODE"
if [[ "$VALIDATE_ONLY" == true ]]; then
    echo "Input bundle is valid."
    exit 0
fi
INPUTS_SIGNATURE="$(cd "$INPUT_DIR" && sha256sum labels.csv raw_global_otu_dedup.tsv merged_seqs.fasta | sha256sum | cut -d' ' -f1)"
SPLIT_INPUT_SIGNATURE="${SCRIPT_DIR}/output/${MODE}/split_data_${MODE}/.input_signature"
if [[ ! -f "$SPLIT_INPUT_SIGNATURE" || "$(<"$SPLIT_INPUT_SIGNATURE")" != "$INPUTS_SIGNATURE" ]]; then
    echo "ERROR: No matching split provenance found. Run 01-master_loco.sh with this input directory first." >&2
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
        --labels "$INPUT_DIR/labels.csv" \
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
        --metric "ROC_AUC" \
        --best_fold_dir "output/loco/results/run4" \
        --worst_fold_dir "output/loco/results/run10" \
        --labels_path "$INPUT_DIR/labels.csv"

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