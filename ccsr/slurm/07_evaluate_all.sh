#!/bin/bash
# ============================================================
# Submit evaluation jobs for ALL trained checkpoints.
# Run this after all training jobs are complete.
# ============================================================
set -euo pipefail
source slurm/env.sh

count=0
for model_dir in outputs/ablations/*/seed_*/best_model outputs/baselines/*/seed_*/best_model; do
    if [ -d "$model_dir" ]; then
        job_name="eval-$(basename $(dirname $(dirname $model_dir)))-$(basename $(dirname $model_dir))"
        echo "  Submitting eval: ${model_dir}"
        sbatch \
            --job-name="$job_name" \
            --export="ALL,MODEL_PATH=${model_dir}" \
            slurm/06_evaluate_single.sbatch
        count=$((count + 1))
    fi
done

if [ $count -eq 0 ]; then
    echo "No best_model directories found. Are the training jobs complete?"
    exit 1
fi

echo ""
echo "Submitted ${count} evaluation jobs. Monitor with: squeue -u \$USER"
