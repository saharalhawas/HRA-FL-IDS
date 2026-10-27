"""
Unified HRA Configuration Experiment
================================================================
Runs ONE fixed HRA configuration against BOTH attacks, plus the
cross-configuration matrix (each attack-specific config vs each attack),
over multiple seeds.

Configs:
  unified : alpha=0.35, tau=0.05, fallback, warm-up 8  (used for ALL attacks)
  lf_tuned: alpha=0.10, no exclusion                  (old label-flipping config)

Usage:
  python run_unified_defense.py --seeds 42              # quick check (4 runs)
  python run_unified_defense.py                         # full: 10 seeds (40 runs)

Results are saved after every run, so the script can be stopped and resumed.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from attack_config import ALL_ATTACK_TYPES
from data_loader import FLDataset
from defense_config import DefenseConfig
from main import run_experiment

RESULTS_DIR = os.path.join("results", "unified_defense")
OUT_JSON = os.path.join(RESULTS_DIR, "unified_defense_results.json")

DEFAULT_SEEDS = [42, 123, 7, 99, 500, 2024, 31, 77, 1001, 256]

CONFIGS = {
    "unified": DefenseConfig.unified(),
    "lf_tuned": DefenseConfig(),  # old label-flipping config (alpha=0.1, no exclusion)
}


def extra_metrics(test_metrics):
    """Imbalance-aware metrics computed from the saved test predictions."""
    y = test_metrics["test_targets"]
    p = test_metrics["test_probs"]
    pred = (p >= test_metrics["best_threshold"]).astype(int)
    return {
        "mcc": float(matthews_corrcoef(y, pred) * 100),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred) * 100),
        "pr_auc": float(average_precision_score(y, p) * 100),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "best_threshold": float(test_metrics["best_threshold"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--malicious-ratio", type=float, default=0.4)
    ap.add_argument("--num-clients", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            results = json.load(f)

    dataset = FLDataset(
        "data/cic", None, dataset_type="cicids2017",
        model_type="mlp", val_ratio=0.1, seed=42, subsample_size=500000,
    )

    for attack in ALL_ATTACK_TYPES:
        for cfg_name, cfg in CONFIGS.items():
            for seed in args.seeds:
                key = f"{attack}|{cfg_name}|{seed}"
                if key in results:
                    print(f"  [Skip] {key} (already done)")
                    continue

                dataset.reshuffle(seed)
                _, _, m = run_experiment(
                    num_clients=args.num_clients,
                    num_rounds=args.rounds,
                    malicious_ratio=args.malicious_ratio,
                    attack_type=attack,
                    defense=True,
                    defense_config=cfg,
                    model_type="mlp",
                    batch_size=64,
                    local_epochs=5,
                    lr=0.01,
                    seed=seed,
                    dataset=dataset,
                    patience=args.rounds,
                )
                results[key] = {
                    "attack": attack, "config": cfg_name, "seed": seed,
                    "accuracy": float(m["accuracy"]),
                    "f1_binary": float(m["f1_binary"]),
                    "recall_binary": float(m["recall_binary"]),
                    "precision_binary": float(m["precision_binary"]),
                    "roc_auc": float(m["roc_auc"]),
                    **extra_metrics(m),
                }
                with open(OUT_JSON, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  [Saved] {key}: F1={m['f1_binary']:.2f}%")

    # Summary: mean ± sample std (ddof=1) per attack × config
    print(f"\n{'=' * 78}")
    print(f"  {'Attack':<18}{'Config':<11}{'n':>3}{'F1':>16}{'AUC':>16}{'MCC':>14}")
    print(f"  {'-' * 76}")
    for attack in ALL_ATTACK_TYPES:
        for cfg_name in CONFIGS:
            rows = [r for r in results.values()
                    if r["attack"] == attack and r["config"] == cfg_name]
            if not rows:
                continue

            def ms(k):
                v = np.array([r[k] for r in rows])
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                return f"{v.mean():6.2f}±{sd:4.2f}"

            print(f"  {attack:<18}{cfg_name:<11}{len(rows):>3}"
                  f"{ms('f1_binary'):>16}{ms('roc_auc'):>16}{ms('mcc'):>14}")
    print("=" * 78)


if __name__ == "__main__":
    main()