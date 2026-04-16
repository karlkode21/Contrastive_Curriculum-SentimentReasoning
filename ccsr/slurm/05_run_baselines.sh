#!/bin/bash
# ============================================================
# Submit baseline jobs: SFT label-only and SFT label+rationale.
# ============================================================
set -euo pipefail
source slurm/env.sh

SEEDS=(42 123 456)

declare -A BASELINES=(
    ["llama3_sft_label_only"]="--label_only --no_contrastive --no_curriculum --no_focal"
    ["llama3_sft_rationale"]="--no_contrastive --no_curriculum --no_focal"
)

echo "Submitting 6 baseline jobs..."

for variant in "${!BASELINES[@]}"; do
    extra="${BASELINES[$variant]}"
    for seed in "${SEEDS[@]}"; do
        job_name="base-${variant}-s${seed}"
        echo "  Submitting: ${job_name}"
        sbatch \
            --job-name="$job_name" \
            --export="ALL,VARIANT=${variant},SEED=${seed},EXTRA_ARGS=${extra}" \
            slurm/03_train_single.sbatch
    done
done

echo ""
echo "All baseline jobs submitted. Monitor with: squeue -u \$USER"
