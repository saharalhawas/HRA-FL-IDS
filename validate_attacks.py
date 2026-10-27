"""
Quick validation: verify both attacks degrade FedAvg vs baseline threshold.

Success criteria (per attack, no defense, 15 rounds):
  - test f1_binary drop >= 40 pp vs baseline mean, OR
  - roc_auc <= 60%

Usage:
  python validate_attacks.py
  python validate_attacks.py --quick   # 8 rounds only
"""

import argparse
import json
import os

import numpy as np

from attack_config import ALL_ATTACK_TYPES, AttackConfig
from data_loader import FLDataset
from main import run_experiment
from results_paths import BASELINE_JSON, ensure_dirs, load_baseline_means


def evaluate_attack_impact(attack_type, dataset, baseline_f1, rounds, malicious_ratio, seed):
    """Run short attack experiment and return metrics + pass/fail."""
    dataset.reshuffle(seed)
    _, _, metrics = run_experiment(
        num_clients=10,
        num_rounds=rounds,
        malicious_ratio=malicious_ratio,
        attack_type=attack_type,
        defense=False,
        model_type='mlp',
        batch_size=64,
        local_epochs=5,
        lr=0.01,
        seed=seed,
        dataset=dataset,
        patience=rounds,
    )
    f1_drop = baseline_f1 - metrics['f1_binary']
    auc = metrics['roc_auc']
    passed = f1_drop >= 40.0 or auc <= 60.0
    return {
        'attack_type': attack_type,
        'f1_binary': metrics['f1_binary'],
        'roc_auc': auc,
        'f1_drop_pp': f1_drop,
        'passed': passed,
    }


def sweep_model_poisoning(dataset, baseline_f1, rounds, malicious_ratio, seed):
    """Try default config first, then scale/noise grid until impact criteria pass."""
    default_cfg = AttackConfig.for_attack('model_poisoning')

    def _run(cfg):
        dataset.reshuffle(seed)
        _, _, metrics = run_experiment(
            num_clients=10,
            num_rounds=rounds,
            malicious_ratio=malicious_ratio,
            attack_type='model_poisoning',
            defense=False,
            model_type='mlp',
            batch_size=64,
            local_epochs=5,
            lr=0.01,
            seed=seed,
            dataset=dataset,
            patience=rounds,
            attack_config=cfg,
        )
        f1_drop = baseline_f1 - metrics['f1_binary']
        auc = metrics['roc_auc']
        passed = f1_drop >= 40.0 or auc <= 60.0
        return {
            'scale_factor': cfg.scale_factor,
            'noise_std': cfg.noise_std,
            'f1_binary': metrics['f1_binary'],
            'roc_auc': auc,
            'f1_drop_pp': f1_drop,
            'passed': passed,
        }

    print("  Trying default model_poisoning config...")
    best = _run(default_cfg)
    print(
        f"  default: F1={best['f1_binary']:.1f}% drop={best['f1_drop_pp']:.1f}pp "
        f"AUC={best['roc_auc']:.1f}% {'PASS' if best['passed'] else 'FAIL'}"
    )
    if best['passed']:
        return best

    for scale in [3.0, 8.0]:
        for noise in [3.0, 10.0]:
            cfg = AttackConfig(
                name="model_poisoning",
                lr_multiplier=1.0,
                epoch_multiplier=1.0,
                scale_factor=scale,
                noise_std=noise,
                flip_sign=-1.0,
            )
            result = _run(cfg)
            print(
                f"  scale={scale} noise={noise}: F1={result['f1_binary']:.1f}% "
                f"drop={result['f1_drop_pp']:.1f}pp AUC={result['roc_auc']:.1f}% "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )
            if result['passed'] and result['f1_drop_pp'] > best.get('f1_drop_pp', -1):
                best = result
            if result['passed']:
                return result
    return best if best.get('passed') else best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Use 8 rounds instead of 15')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--malicious-ratio', type=float, default=0.4)
    args = parser.parse_args()

    ensure_dirs()
    rounds = 8 if args.quick else 15

    baseline = load_baseline_means(BASELINE_JSON)
    if not baseline:
        print(f"  [Warning] No baseline at {BASELINE_JSON}; using F1=93.5% reference")
        baseline_f1 = 93.5
    else:
        baseline_f1 = baseline.get('f1_binary_mean', 93.5)
        print(f"  Baseline reference F1: {baseline_f1:.2f}%")

    dataset = FLDataset(
        "data/cic", None, dataset_type='cicids2017',
        model_type='mlp', val_ratio=0.1, seed=42, subsample_size=500000,
    )

    results = []
    print(f"\n{'=' * 60}\n  ATTACK IMPACT VALIDATION ({rounds} rounds)\n{'=' * 60}")

    for attack_type in ALL_ATTACK_TYPES:
        print(f"\n--- {attack_type} ---")
        if attack_type == 'model_poisoning':
            r = sweep_model_poisoning(
                dataset, baseline_f1, rounds, args.malicious_ratio, args.seed
            )
            if r is None:
                r = {'attack_type': attack_type, 'passed': False,
                     'note': 'No parameter combo passed; using defaults'}
            else:
                r['attack_type'] = attack_type
        else:
            r = evaluate_attack_impact(
                attack_type, dataset, baseline_f1, rounds,
                args.malicious_ratio, args.seed,
            )
            print(
                f"  F1={r['f1_binary']:.1f}% drop={r['f1_drop_pp']:.1f}pp "
                f"AUC={r['roc_auc']:.1f}% {'PASS' if r['passed'] else 'FAIL'}"
            )
        results.append(r)

    out_path = os.path.join('results', 'summary', 'attack_validation.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'baseline_f1': baseline_f1, 'rounds': rounds, 'results': results}, f, indent=2)
    print(f"\n  [Saved] {out_path}")

    all_pass = all(r.get('passed', False) for r in results)
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
