#!/bin/bash
# ============================================================
# Submit all 7 ablation variants x 3 seeds = 21 jobs.
# Each job trains independently; SLURM schedules them.
# ============================================================
set -euo pipefail
source slurm/env.sh

SEEDS=(42 123 456)

# variant_name -> extra args for train.py
declare -A ABLATIONS=(
    ["sft_only"]="--no_contrastive --no_curriculum --no_focal"
    ["focal_only"]="--no_contrastive --no_curriculum"
    ["contrastive_only"]="--no_curriculum --no_focal"
    ["curriculum_only"]="--no_contrastive --no_focal"
    ["contrastive_focal"]="--no_curriculum"
    ["curriculum_focal"]="--no_contrastive"
    ["ccsr_full"]=""
)

echo "Submitting 21 ablation jobs..."

for variant in "${!ABLATIONS[@]}"; do
    extra="${ABLATIONS[$variant]}"
    for seed in "${SEEDS[@]}"; do
        job_name="ccsr-${variant}-s${seed}"
        echo "  Submitting: ${job_name}"
        sbatch \
            --job-name="$job_name" \
            --export="ALL,VARIANT=${variant},SEED=${seed},EXTRA_ARGS=${extra}" \
            slurm/03_train_single.sbatch
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
