"""
Attack-Intensity Sensitivity (Table 13, Fig. 11)
===============================================================
Model poisoning (default) or label flipping (--attack label_flipping),
one factor changed at a time (all others as in the paper):

  ratio : malicious-client ratio m/K in {0.1, 0.2, 0.3, 0.5}   (s = 5, sigma = 5)
  scale : scaling factor s        in {1.5, 2, 10}              (m/K = 0.4, sigma = 5)
  noise : Gaussian noise sigma    in {0, 2, 10}                (m/K = 0.4, s = 5)

The paper setting (m/K = 0.4, s = 5, sigma = 5) is already covered by
run_defense_comparison_seeds.py, so it is not repeated here.

Defenses: HRA (unified config), Multi-Krum, Krum, Trimmed Mean.
Krum / Multi-Krum / Trimmed Mean keep their fixed assumption of f = 40 %,
because in practice the true number of attackers is unknown.

Results are saved after every run; re-run the same command to resume.

Usage:
  python3 run_attack_sensitivity.py --part ratio            # ~17 h
  python3 run_attack_sensitivity.py --part scale            # ~13 h
  python3 run_attack_sensitivity.py --part noise            # ~13 h
  python3 run_attack_sensitivity.py --part ratio --rounds 2 --seeds 42   # quick test
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef
from tqdm import tqdm

from attack_config import AttackConfig
from data_loader import FLDataset
from defense_config import DefenseConfig
from evaluation import evaluate_with_threshold, get_probabilities, sweep_thresholds
from federated_client import FederatedClient
from federated_server import FederatedServer
from main import set_seed
from model import MLPModel

RESULTS_DIR = os.path.join("results", "attack_sensitivity")
SEEDS = [42, 123, 7]
DEFENSES = ["hra", "multi_krum", "krum", "trimmed_mean"]
PARTS = {
    "ratio": [0.1, 0.2, 0.3, 0.5],
    "scale": [1.5, 2.0, 10.0],
    "noise": [0.0, 2.0, 10.0],
}


ATTACK = "model_poisoning"   # set from --attack in main()


def attack_cfg(scale, noise):
    base = AttackConfig.for_attack(ATTACK)
    base.scale_factor = scale
    base.noise_std = noise
    return base


def run_one(defense, seed, ratio, cfg_attack, dataset, args):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.reshuffle(seed)
    client_data = dataset.split_data(args.num_clients, iid=False)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    cfg = DefenseConfig.unified() if defense == "hra" else DefenseConfig()
    server = FederatedServer(MLPModel, device, args.num_clients,
                             defense_config=cfg, input_dim=dataset.input_dim)

    n_mal = int(round(args.num_clients * ratio))
    malicious = set(range(n_mal))
    clients = []
    for i in range(args.num_clients):
        is_mal = i in malicious
        clients.append(FederatedClient(
            i, client_data[i], device, model_type="mlp",
            is_malicious=is_mal, attack_type=ATTACK if is_mal else None,
            batch_size=args.batch_size, loss_type="ce",
            global_class_weights=getattr(dataset, "global_class_weights", None),
            attack_config=cfg_attack if is_mal else None,
        ))

    excluded_per_round = []
    for r in tqdm(range(args.rounds), desc=f"{defense} m/K={ratio} s={cfg_attack.scale_factor} "
                                           f"sigma={cfg_attack.noise_std} seed={seed}"):
        weights, sizes = [], []
        for c in clients:
            weights.append(c.train_local_model(server.get_global_model(),
                                               epochs=args.local_epochs, lr=args.lr))
            sizes.append(c.get_update_size())
        if defense == "hra":
            info = server.aggregate_hybrid_robust(weights, sizes)
            reps = np.array(info["ema_reputations"])
            excluded_per_round.append([int(i) for i in np.where(reps < cfg.hard_exclude_threshold)[0]])
        elif defense == "multi_krum":
            server.aggregate_multi_krum(weights, sizes)
        elif defense == "krum":
            server.aggregate_krum(weights, sizes)
        elif defense == "trimmed_mean":
            server.aggregate_trimmed_mean(weights, sizes)
        server.checkpoint_if_best(server.evaluate(val_loader), r + 1)

    server.restore_best_model()
    model = server.get_global_model()
    vt, vp = get_probabilities(model, val_loader, device)
    _, thr, _ = sweep_thresholds(vt, vp)
    tt, tp = get_probabilities(model, test_loader, device)
    m = evaluate_with_threshold(tt, tp, threshold=thr)
    pred = (tp >= thr).astype(int)
    out = {
        "defense": defense, "seed": seed, "malicious_ratio": ratio,
        "scale": cfg_attack.scale_factor, "noise": cfg_attack.noise_std,
        "best_round": int(server.best_round), "threshold": float(thr),
        "accuracy": float(m["accuracy"]), "f1_binary": float(m["f1_binary"]),
        "precision_binary": float(m["precision_binary"]),
        "recall_binary": float(m["recall_binary"]), "roc_auc": float(m["roc_auc"]),
        "mcc": float(matthews_corrcoef(tt, pred) * 100),
        "pr_auc": float(average_precision_score(tt, tp) * 100),
        "confusion_matrix": m["confusion_matrix"],
    }
    if defense == "hra" and excluded_per_round and malicious:
        honest = set(range(args.num_clients)) - malicious
        last = set(excluded_per_round[-1])
        out.update({
            "malicious_excluded_final_pct": len(malicious & last) / len(malicious) * 100,
            "honest_wrongly_excluded_final_pct": len(honest & last) / len(honest) * 100,
            "first_round_all_malicious_excluded": next(
                (i + 1 for i, e in enumerate(excluded_per_round) if malicious <= set(e)), None),
            "excluded_per_round": excluded_per_round,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(PARTS))
    ap.add_argument("--attack", default="model_poisoning", choices=["model_poisoning", "label_flipping"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--defenses", nargs="+", default=DEFENSES)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    global ATTACK
    ATTACK = args.attack
    if ATTACK == "label_flipping" and args.part == "noise":
        raise SystemExit("Label flipping has no Gaussian noise (sigma=0); use --part ratio or scale.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if args.rounds == 50 else f"_test{args.rounds}rounds"
    prefix = "lf_" if ATTACK == "label_flipping" else ""
    out_json = os.path.join(RESULTS_DIR, f"{prefix}{args.part}{suffix}.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}

    dataset = FLDataset("data/cic", None, dataset_type="cicids2017",
                        model_type="mlp", val_ratio=0.1, seed=42, subsample_size=500000)

    for value in PARTS[args.part]:
        ratio, scale, noise = 0.4, 5.0, (0.0 if ATTACK == "label_flipping" else 5.0)
        if args.part == "ratio":
            ratio = value
        elif args.part == "scale":
            scale = value
        else:
            noise = value
        for seed in args.seeds:
            for d in args.defenses:
                key = f"{value}|{d}|{seed}"
                if key in results:
                    print(f"  [Skip] {args.part}={key} (already done)")
                    continue
                res = run_one(d, seed, ratio, attack_cfg(scale, noise), dataset, args)
                if os.path.exists(out_json):   # re-read so several terminals can share one file
                    results.update(json.load(open(out_json)))
                results[key] = res
                tmp = out_json + f".tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(tmp, out_json)
                print(f"  [Saved] {args.part}={value} {d} seed={seed}: "
                      f"F1={res['f1_binary']:.2f}% AUC={res['roc_auc']:.2f}%")

    print(f"\n{'=' * 64}\n  {args.part} — mean F1 (%) over completed seeds\n{'-' * 64}")
    print("  " + f"{args.part:<8}" + "".join(f"{d:>14}" for d in args.defenses))
    for value in PARTS[args.part]:
        row = f"  {value:<8}"
        for d in args.defenses:
            v = [x["f1_binary"] for k, x in results.items() if k.startswith(f"{value}|{d}|")]
            row += f"{np.mean(v):>14.2f}" if v else f"{'-':>14}"
        print(row)
    print("=" * 64)


if __name__ == "__main__":
    main()