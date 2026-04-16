#!/bin/bash
# ============================================================
# Shared environment for all SLURM jobs.
# Edit these values ONCE to match your cluster setup.
# ============================================================

# -- SLURM account (required on your cluster) --
export CCSR_ACCOUNT="${ACCOUNT:-your_account_here}"

# -- Partition --
# Your cluster has gpuH200x8-interactive; for batch jobs the
# non-interactive variant is typical. Change if needed.
export CCSR_PARTITION="${CCSR_PARTITION:-gpuH200x8}"

# -- Conda / virtualenv activation --
# Uncomment and edit ONE of these:
# source activate ccsr
# module load cuda/12.1 python/3.10
# conda activate ccsr

# -- Project root (set this to wherever you cloned the repo) --
export CCSR_ROOT="${CCSR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# -- HuggingFace cache (optional, avoids re-downloading models) --
export HF_HOME="${HF_HOME:-$CCSR_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"

cd "$CCSR_ROOT"
