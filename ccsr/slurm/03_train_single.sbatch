#!/bin/bash
# ============================================================
# Train a single CCSR variant with a single seed.
# Usage:
#   sbatch --export=ALL,VARIANT=ccsr_full,SEED=42 slurm/03_train_single.sbatch
#   sbatch --export=ALL,VARIANT=sft_only,SEED=42,EXTRA_ARGS="--no_contrastive --no_curriculum --no_focal" slurm/03_train_single.sbatch
# ============================================================
#SBATCH --job-name=ccsr-train
#SBATCH --account=$CCSR_ACCOUNT
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/train_%x_%j.out
#SBATCH --error=slurm/logs/train_%x_%j.err

source slurm/env.sh
mkdir -p slurm/logs

VARIANT="${VARIANT:-ccsr_full}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== Training: variant=${VARIANT}, seed=${SEED} ==="
echo "Node: $(hostname), GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

python -m src.scripts.train \
    --config configs/default.yaml \
    --output_dir "outputs/ablations/${VARIANT}" \
    --seed "$SEED" \
    --difficulty_scores outputs/difficulty_scores/difficulty_scores.json \
    --rationale_sim outputs/rationale_sim/rationale_sim.json \
    $EXTRA_ARGS

echo "=== Training complete ==="
