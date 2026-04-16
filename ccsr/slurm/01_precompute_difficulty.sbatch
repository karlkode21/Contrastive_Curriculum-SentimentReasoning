#!/bin/bash
#SBATCH --job-name=ccsr-difficulty
#SBATCH --account=$CCSR_ACCOUNT
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/difficulty_%j.out
#SBATCH --error=slurm/logs/difficulty_%j.err

source slurm/env.sh
mkdir -p slurm/logs

echo "=== Precompute difficulty scores ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m src.scripts.precompute_difficulty \
    --output_dir outputs/difficulty_scores \
    --bert_model bert-base-uncased \
    --epochs 3 \
    --batch_size 64

echo "=== Done ==="
