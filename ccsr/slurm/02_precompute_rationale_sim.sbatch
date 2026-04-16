#!/bin/bash
#SBATCH --job-name=ccsr-ratsim
#SBATCH --account=$CCSR_ACCOUNT
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/ratsim_%j.out
#SBATCH --error=slurm/logs/ratsim_%j.err

source slurm/env.sh
mkdir -p slurm/logs

echo "=== Precompute rationale similarity matrix ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m src.scripts.precompute_rationale_sim \
    --output_dir outputs/rationale_sim \
    --threshold 0.85 \
    --batch_size 256

echo "=== Done ==="
