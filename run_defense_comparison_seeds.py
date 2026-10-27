"""
Multi-Seed Defense Comparison (Table 7)
================================================================
Compares FedAvg, Coordinate Median, Trimmed Mean, Krum, Multi-Krum,
FoolsGold and HRA (unified configuration) under the same attack,
over 10 seeds. Identical data split, model, attack and seeds for every
defense. Results are saved after every run, so the script can be
stopped and resumed with the same command.

For HRA it also records client-level detection: how many malicious
clients were excluded and how many honest clients were wrongly excluded.

Usage:
  python3 run_defense_comparison_seeds.py --seeds 42 --rounds 2   # quick test (minutes)
  python3 run_defense_comparison_seeds.py                         # full run (70 runs)
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

RESULTS_DIR = os.path.join("results", "defense_comparison_seeds")
SEEDS = [42, 123, 7, 99, 500, 2024, 31, 77, 1001, 256]
DEFENSES = ["fedavg", "coordinate_median", "trimmed_mean", "krum",
            "multi_krum", "foolsgold", "hra"]


def run_one(defense, attack, seed, dataset, args):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.reshuffle(seed)
    client_data = dataset.split_data(args.num_clients, iid=False)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    cfg = DefenseConfig.unified() if defense == "hra" else DefenseConfig()
    server = FederatedServer(MLPModel, device, args.num_clients,
                             defense_config=cfg, input_dim=dataset.input_dim)

    n_mal = int(args.num_clients * args.malicious_ratio)
    malicious = set(range(n_mal))
    clients = []
    for i in range(args.num_clients):
        is_mal = i in malicious
        clients.append(FederatedClient(
            i, client_data[i], device, model_type="mlp",
            is_malicious=is_mal, attack_type=attack if is_mal else None,
            batch_size=args.batch_size, loss_type="ce",
            global_class_weights=getattr(dataset, "global_class_weights", None),
            attack_config=AttackConfig.for_attack(attack) if is_mal else None,
        ))

    excluded_per_round = []
    for r in tqdm(range(args.rounds), desc=f"{defense} {attack} seed={seed}"):
        weights, sizes = [], []
        for c in clients:
            weights.append(c.train_local_model(server.get_global_model(),
                                               epochs=args.local_epochs, lr=args.lr))
            sizes.append(c.get_update_size())

        if defense == "fedavg":
            server.aggregate_fedavg(weights, sizes)
        elif defense == "coordinate_median":
            server.aggregate_coordinate_median(weights, sizes)
        elif defense == "trimmed_mean":
            server.aggregate_trimmed_mean(weights, sizes)
        elif defense == "krum":
            server.aggregate_krum(weights, sizes)
        elif defense == "multi_krum":
            server.aggregate_multi_krum(weights, sizes)
        elif defense == "foolsgold":
            server.aggregate_foolsgold(weights, sizes)
        elif defense == "hra":
            info = server.aggregate_hybrid_robust(weights, sizes)
            reps = np.array(info["ema_reputations"])
            excluded_per_round.append([int(i) for i in np.where(reps < cfg.hard_exclude_threshold)[0]])

        server.checkpoint_if_best(server.evaluate(val_loader), r + 1)

    server.restore_best_model()
    model = server.get_global_model()
    vt, vp = get_probabilities(model, val_loader, device)
    _, thr, _ = sweep_thresholds(vt, vp)
    tt, tp = get_probabilities(model, test_loader, device)
    m = evaluate_with_threshold(tt, tp, threshold=thr)
    pred = (tp >= thr).astype(int)

    out = {
        "defense": defense, "attack": attack, "seed": seed,
        "best_round": int(server.best_round), "threshold": float(thr),
        "accuracy": float(m["accuracy"]), "f1_binary": float(m["f1_binary"]),
        "precision_binary": float(m["precision_binary"]),
        "recall_binary": float(m["recall_binary"]), "roc_auc": float(m["roc_auc"]),
        "mcc": float(matthews_corrcoef(tt, pred) * 100),
        "pr_auc": float(average_precision_score(tt, tp) * 100),
        "confusion_matrix": m["confusion_matrix"],
    }

    if defense == "hra" and excluded_per_round:
        honest = set(range(args.num_clients)) - malicious
        det = [len(malicious & set(e)) / len(malicious) * 100 for e in excluded_per_round]
        fex = [len(honest & set(e)) / len(honest) * 100 for e in excluded_per_round]
        first_full = next((i + 1 for i, e in enumerate(excluded_per_round)
                           if malicious <= set(e)), None)
        out.update({
            "malicious_excluded_pct_mean": float(np.mean(det)),
            "honest_wrongly_excluded_pct_mean": float(np.mean(fex)),
            "first_round_all_malicious_excluded": first_full,
            "excluded_per_round": excluded_per_round,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="model_poisoning",
                    choices=["model_poisoning", "label_flipping"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--defenses", nargs="+", default=DEFENSES)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--malicious-ratio", type=float, default=0.4)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if args.rounds == 50 else f"_test{args.rounds}rounds"
    out_json = os.path.join(RESULTS_DIR, f"{args.attack}{suffix}.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}

    dataset = FLDataset("data/cic", None, dataset_type="cicids2017",
                        model_type="mlp", val_ratio=0.1, seed=42, subsample_size=500000)

    for seed in args.seeds:            # seed-major order: every defense finishes a seed together
        for d in args.defenses:
            key = f"{d}|{seed}"
            if key in results:
                print(f"  [Skip] {key} (already done)")
                continue
            res = run_one(d, args.attack, seed, dataset, args)
            results[key] = res
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
            extra = ""
            if "malicious_excluded_pct_mean" in res:
                extra = (f" | malicious excluded {res['malicious_excluded_pct_mean']:.0f}%"
                         f" | honest wrongly excluded {res['honest_wrongly_excluded_pct_mean']:.0f}%")
            print(f"  [Saved] {key}: F1={res['f1_binary']:.2f}% AUC={res['roc_auc']:.2f}%{extra}")

    print(f"\n{'=' * 72}\n  {args.attack} — mean ± SD over completed seeds\n{'-' * 72}")
    print(f"  {'Defense':<20}{'n':>3}{'F1':>16}{'AUC':>16}{'MCC':>16}")
    for d in args.defenses:
        rows = [v for v in results.values() if v["defense"] == d]
        if not rows:
            continue

        def ms(k):
            v = np.array([x[k] for x in rows])
            return f"{v.mean():6.2f}±{(v.std(ddof=1) if len(v) > 1 else 0):5.2f}"
        print(f"  {d:<20}{len(rows):>3}{ms('f1_binary'):>16}{ms('roc_auc'):>16}{ms('mcc'):>16}")
    print("=" * 72)


if __name__ == "__main__":
    main()