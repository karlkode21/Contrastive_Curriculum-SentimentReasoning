"""Run all 7 ablation variants across 3 seeds."""

import subprocess
import sys

ABLATIONS = {
    "sft_only":          {"no_contrastive": True,  "no_curriculum": True,  "no_focal": True},
    "focal_only":        {"no_contrastive": True,  "no_curriculum": True,  "no_focal": False},
    "contrastive_only":  {"no_contrastive": False, "no_curriculum": True,  "no_focal": True},
    "curriculum_only":   {"no_contrastive": True,  "no_curriculum": False, "no_focal": True},
    "contrastive_focal": {"no_contrastive": False, "no_curriculum": True,  "no_focal": False},
    "curriculum_focal":  {"no_contrastive": True,  "no_curriculum": False, "no_focal": False},
    "ccsr_full":         {"no_contrastive": False, "no_curriculum": False, "no_focal": False},
}

SEEDS = [42, 123, 456]


def main():
    for name, flags in ABLATIONS.items():
        for seed in SEEDS:
            print(f"\n{'=' * 60}")
            print(f"Ablation: {name} (seed={seed})")
            print(f"{'=' * 60}\n")

            cmd = [
                sys.executable, "-m", "src.scripts.train",
                "--config", "configs/default.yaml",
                "--output_dir", f"outputs/ablations/{name}",
                "--seed", str(seed),
                "--difficulty_scores", "outputs/difficulty_scores/difficulty_scores.json",
                "--rationale_sim", "outputs/rationale_sim/rationale_sim.json",
            ]

            for flag_name, is_disabled in flags.items():
                if is_disabled:
                    cmd.append(f"--{flag_name}")

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"WARNING: {name} seed={seed} failed (exit {result.returncode})")

    print("\nAll ablations complete. Run evaluate.py on each output directory.")


if __name__ == "__main__":
    main()
