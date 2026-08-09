#!/bin/bash
#SBATCH --job-name=benchmark_loco
#SBATCH --partition=shared-cpu         # Uses the cluster's main CPU node pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # Request 8 CPU cores for parallel tuning
#SBATCH --mem=16G                      # 16G is plenty for N=66
#SBATCH --time=05:00:00
#SBATCH --output=output/slurm_outputs/benchmark_lopo_%j.log
#SBATCH --error=output/slurm_outputs/benchmark_lopo_%j.err

mkdir -p output/slurm_outputs

echo "=== Job Started at $(date) ==="
echo "Allocated Node: $SLURM_NODELIST"

echo "========================================="
echo "Starting Global Benchmarking Pipeline..."
echo "========================================="

# Activating Env
source /opt/mambaforge/etc/profile.d/conda.sh
conda activate benchmark_env

# ---------------------------------------------------------
# Python Loop
# ---------------------------------------------------------

echo "[Phase 3] Dispatching parallel benchmarking jobs..."

for run_path in ./output/split_data/run*/ ; do

    RUN_NAME=$(basename ${run_path})
    
    echo "-----------------------------------------"
    echo "Dispatching Benchmark for Split: ${RUN_NAME}"
    
    mkdir -p output/results/benchmark/${RUN_NAME}/

    python3 02-benchmarking.py \
    --input output/results/predictions_with_alpha/${RUN_NAME}/ \
    --labels input/labels.csv \
    --outdir output/results/benchmark_with_alpha/${RUN_NAME}/


done


echo "-----------------------------------------"
echo "All Individual Benchmarks Completed."
echo "Executing Global Stability Aggregation..."
echo "-----------------------------------------"

# Run the stability script targeting the main benchmark folder
python3 02.5-benchmarking_stability.py \
    --benchmark_dir output/results/benchmark_with_alpha/ \
    --outdir output/results/benchmark_with_alpha/global/

echo "Pipeline Finished Successfully!"

conda deactivate

END_TIME=$(date +%s)
ELAPSED_TOTAL=$((END_TIME - START_TIME))
echo "=========================================="
echo "=== Benchmarking Finished at $(date) ==="
echo "Total Execution Time: $((ELAPSED_TOTAL / 60)) minutes and $((ELAPSED_TOTAL % 60)) seconds."
echo "=========================================="
exit