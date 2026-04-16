#!/bin/bash
# ============================================================
# Evaluate a single trained checkpoint.
# Usage:
#   sbatch --export=ALL,MODEL_PATH=outputs/ablations/ccsr_full/seed_42/best_model slurm/06_evaluate_single.sbatch
# ============================================================
#SBATCH --job-name=ccsr-eval
#SBATCH --account=$CCSR_ACCOUNT
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/eval_%j.out
#SBATCH --error=slurm/logs/eval_%j.err

source slurm/env.sh
mkdir -p slurm/logs

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH via --export}"
EVAL_DIR="${MODEL_PATH%best_model}eval_results"

echo "=== Evaluating: ${MODEL_PATH} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m src.scripts.evaluate \
    --model_path "$MODEL_PATH" \
    --output_dir "$EVAL_DIR" \
    --batch_size 8

echo "=== Results saved to ${EVAL_DIR}/results.json ==="
