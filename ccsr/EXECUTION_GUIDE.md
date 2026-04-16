# CCSR Execution Guide

Step-by-step instructions for running the Contrastive Curriculum Sentiment Reasoning experiments on a SLURM cluster with H200 GPUs.

---

## Prerequisites

- SLURM cluster access with GPU partition
- HuggingFace account with Llama 3 8B-Instruct access (accept Meta's license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- Python 3.10+

---

## Step 0: Setup

### Transfer to cluster

Copy the entire project directory to your cluster home or scratch space.

### Configure SLURM environment

Edit `slurm/env.sh` — this is the **only file you need to modify**:

```bash
vi slurm/env.sh
```

Set these three things:
1. `CCSR_ACCOUNT` — your SLURM account name (the `$ACCOUNT` you use with `srun`)
2. `CCSR_PARTITION` — your GPU partition (default: `gpuH200x8`; change if your batch partition name differs from the interactive one)
3. Uncomment the conda/module activation line matching your setup

### Install dependencies

```bash
cd ccsr
pip install -e ".[dev]"
```

### Verify tests pass

```bash
python -m pytest tests/ -v
```

All 25 tests should pass.

### Log into HuggingFace

```bash
huggingface-cli login
```

Paste your access token when prompted.

---

## Step 1: Precompute Offline Artifacts

These must complete **before** any training jobs. They produce files that the training scripts depend on.

```bash
cd ccsr
mkdir -p slurm/logs

sbatch slurm/01_precompute_difficulty.sbatch    # ~30 min, 1 GPU
sbatch slurm/02_precompute_rationale_sim.sbatch  # ~1-2 hours, 1 GPU
```

**Wait for both to finish:**
```bash
squeue -u $USER
```

**Verify outputs exist:**
```bash
ls outputs/difficulty_scores/difficulty_scores.json
ls outputs/rationale_sim/rationale_sim.json
```

---

## Step 2: Run Baselines

Submits 6 jobs (2 baseline variants x 3 random seeds):

```bash
bash slurm/05_run_baselines.sh
```

Baselines:
- `llama3_sft_label_only` — Llama 3 fine-tuned on labels only (no rationale)
- `llama3_sft_rationale` — Llama 3 fine-tuned on labels + rationale (SFT-only, no CCSR)

---

## Step 3: Run Ablations

Submits 21 jobs (7 ablation variants x 3 random seeds):

```bash
bash slurm/04_run_all_ablations.sh
```

Ablation variants:
| Variant | Contrastive | Curriculum | Focal Loss |
|---------|:-----------:|:----------:|:----------:|
| `sft_only` | - | - | - |
| `focal_only` | - | - | yes |
| `contrastive_only` | yes | - | - |
| `curriculum_only` | - | yes | - |
| `contrastive_focal` | yes | - | yes |
| `curriculum_focal` | - | yes | yes |
| `ccsr_full` | yes | yes | yes |

**Wait for ALL training jobs to finish** (~4-6 hours per job on 2x H200):
```bash
squeue -u $USER
```

---

## Step 4: Evaluate All Checkpoints

Submits one eval job per trained checkpoint (up to 27 jobs):

```bash
bash slurm/07_evaluate_all.sh
```

Each eval job runs all metrics:
- Classification: accuracy, per-class F1, macro-F1, confusion matrix
- Rationale: ROUGE-1/2/L/Lsum, BERTScore
- Faithfulness: NLI-based entailment check
- Attention entropy: keyword reliance measurement

---

## Step 5: Collect Results

All results are saved as JSON:

```bash
# View a single result
cat outputs/ablations/ccsr_full/seed_42/eval_results/results.json | python -m json.tool

# Quick summary across all variants
for f in outputs/ablations/*/seed_*/eval_results/results.json; do
    variant=$(echo $f | cut -d'/' -f3)
    seed=$(echo $f | cut -d'/' -f4)
    acc=$(python -c "import json; print(json.load(open('$f'))['classification']['accuracy'])")
    f1=$(python -c "import json; print(json.load(open('$f'))['classification']['macro_f1'])")
    echo "$variant $seed acc=$acc f1=$f1"
done
```

---

## Step 6: Generate Paper Figures

Open the notebook on a node with a display or use JupyterHub:

```bash
jupyter notebook notebooks/generate_tables_figures.ipynb
```

This produces:
- `paper/figures/confusion_matrices.pdf`
- `paper/figures/attention_entropy.pdf`
- LaTeX-formatted tables for copy-paste into `paper/main.tex`

---

## Step 7: Update the Paper

The LaTeX paper is at `paper/main.tex`. Mock results are marked in blue with `\mock{}`. To find them all:

```bash
grep -n '\\mock' paper/main.tex
```

Replace each `\mock{value}` with the real number from your results.

Figure placeholders are marked with `\TODO{}`:

```bash
grep -n '\\TODO' paper/main.tex
```

Replace these with `\includegraphics` pointing to the PDFs generated in Step 6.

Compile the paper:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex  # run twice for references
```

---

## Troubleshooting

### Job fails immediately
Check the SLURM log:
```bash
cat slurm/logs/train_<jobname>_<jobid>.err
```

Common causes:
- Wrong partition name → edit `slurm/env.sh`
- Missing conda env → uncomment activation line in `slurm/env.sh`
- Llama 3 access denied → run `huggingface-cli login` and accept the license

### Out of memory
Reduce batch size in `configs/default.yaml`:
```yaml
training:
  per_device_batch_size: 4        # reduce from 8
  gradient_accumulation_steps: 8  # increase to keep effective batch = 32
```

### Training is slow
Verify GPUs are being used:
```bash
srun --jobid=<JOBID> nvidia-smi
```

### Want to run a single experiment manually
```bash
sbatch --export=ALL,VARIANT=ccsr_full,SEED=42 slurm/03_train_single.sbatch
```

### Want to use a different backbone model
Edit `configs/default.yaml`:
```yaml
model:
  backbone: "mistralai/Mistral-7B-Instruct-v0.3"  # or any HF model
```

---

## File Map

```
ccsr/
├── configs/
│   ├── default.yaml          # All hyperparameters
│   └── ds_config.json        # DeepSpeed ZeRO Stage 2
├── slurm/
│   ├── env.sh                # ← EDIT THIS FIRST
│   ├── 01-02_*.sbatch        # Precompute jobs
│   ├── 03_train_single.sbatch # Single training job
│   ├── 04-05_*.sh            # Submit all ablations/baselines
│   ├── 06_evaluate_single.sbatch
│   └── 07_evaluate_all.sh
├── src/
│   ├── data/                 # Dataset loading, prompts, curriculum
│   ├── models/               # Focal loss, contrastive head, CCSRTrainer
│   ├── evaluation/           # All metrics
│   └── scripts/              # Entrypoints (train, evaluate, precompute)
├── tests/                    # 25 unit tests
├── outputs/                  # Created at runtime (gitignored)
└── notebooks/                # Paper figure generation
```
