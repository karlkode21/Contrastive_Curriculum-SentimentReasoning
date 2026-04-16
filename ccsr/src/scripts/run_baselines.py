"""Run baseline experiments: Llama 3 SFT label-only and label+rationale."""

import subprocess
import sys

BASELINES = {
    "llama3_sft_label_only": {
        "desc": "Llama 3 8B SFT (label only)",
        "args": ["--label_only", "--no_contrastive", "--no_curriculum", "--no_focal"],
    },
    "llama3_sft_rationale": {
        "desc": "Llama 3 8B SFT (label + rationale)",
        "args": ["--no_contrastive", "--no_curriculum", "--no_focal"],
    },
}

SEEDS = [42, 123, 456]


def main():
    for name, baseline in BASELINES.items():
        for seed in SEEDS:
            print(f"\n{'=' * 60}")
            print(f"Running: {baseline['desc']} (seed={seed})")
            print(f"{'=' * 60}\n")

            cmd = [
                sys.executable, "-m", "src.scripts.train",
                "--config", "configs/default.yaml",
                "--output_dir", f"outputs/baselines/{name}",
                "--seed", str(seed),
            ] + baseline["args"]

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"WARNING: {name} seed={seed} failed (exit {result.returncode})")

    print("\nAll baselines complete. Run evaluate.py on each output directory.")


if __name__ == "__main__":
    main()
