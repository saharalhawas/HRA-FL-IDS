"""
Client-dropout experiment (Section 5.7.10)
=====================================================================
In every round, a random 20% of the clients (2 of 10) are offline and send no
update. Everything else is as in the main experiment: model poisoning (s = 5,
sigma = 5), 40% malicious clients (0-3), Dirichlet Non-IID, 50 rounds,
unified HRA configuration, seeds 42 / 123 / 7.

  HRA        : reputation is stored per client and updated only when the client
               participates; an absent client keeps its last reputation.
               (Same HRA code as the paper, applied to the participating clients.)
  Multi-Krum : applied to the participating clients (f = 40% of them).

The offline clients are drawn from a fixed random generator per seed, so HRA and
Multi-Krum see exactly the same participation pattern.

Results: results/final_checks/dropout.json (saved after every run; re-run the
same command to continue). Safe to run next to another experiment.

Usage (same folder as run_remaining.py):
  caffeinate -dims python3 run_dropout.py
  python3 run_dropout.py --rounds 2 --seeds 42        # quick test
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef
from tqdm import tqdm

import run_remaining as rr
from data_loader import FLDataset
from defense_config import DefenseConfig
from evaluation import evaluate_with_threshold, get_probabilities, sweep_thresholds
from federated_client import FederatedClient
from federated_server import FederatedServer
from main import set_seed
from model import MLPModel

RESULTS_DIR = os.path.join("results", "final_checks")
DROP_RATE = 0.2


def hra_partial(server, weights, sizes, ids):
    """Run the unchanged HRA aggregation on the participating clients only.
    Reputations of absent clients are left untouched."""
    full = server.reputations.copy()
    server.reputations = full[ids].copy()
    info = server.aggregate_hybrid_robust(weights, sizes)
    full[ids] = server.reputations
    server.reputations = full
    return info


def run_one(defense, seed, dataset, args):
    set_seed(seed)
    device = torch.device("cpu")
    dataset.reshuffle(seed)
    client_data = dataset.split_data(args.num_clients, iid=False)
    val_loader, test_loader = dataset.get_val_loader(), dataset.get_test_loader()
    dcfg = DefenseConfig.unified() if defense == "hra" else DefenseConfig()
    server = FederatedServer(MLPModel, device, args.num_clients, defense_config=dcfg,
                             input_dim=dataset.input_dim)
    acfg = rr.mp_cfg()
    malicious = set(range(int(round(args.num_clients * 0.4))))
    clients = [FederatedClient(i, client_data[i], device, model_type="mlp", is_malicious=i in malicious,
                               attack_type="model_poisoning" if i in malicious else None,
                               batch_size=args.batch_size, loss_type="ce",
                               global_class_weights=getattr(dataset, "global_class_weights", None),
                               attack_config=acfg if i in malicious else None)
               for i in range(args.num_clients)]

    n_off = int(round(args.num_clients * DROP_RATE))
    rng = np.random.default_rng(seed + 10_000)          # same pattern for every defense
    participants, excluded, fallback = [], [], []
    for r in tqdm(range(args.rounds), desc=f"dropout {defense} seed={seed}"):
        ids = np.sort(rng.choice(args.num_clients, args.num_clients - n_off, replace=False))
        w, s = [], []
        for i in ids:
            w.append(clients[i].train_local_model(server.get_global_model(), epochs=args.local_epochs, lr=args.lr))
            s.append(clients[i].get_update_size())
        participants.append([int(i) for i in ids])
        if defense == "hra":
            info = hra_partial(server, w, s, ids)
            excluded.append([int(i) for i in np.where(server.reputations < dcfg.hard_exclude_threshold)[0]])
            fallback.append(bool(info["used_median_fallback"]))
        else:
            server.aggregate_multi_krum(w, s)
        server.checkpoint_if_best(server.evaluate(val_loader), r + 1)

    server.restore_best_model()
    model = server.get_global_model()
    vt, vp = get_probabilities(model, val_loader, device)
    _, th, _ = sweep_thresholds(vt, vp)
    tt, tp = get_probabilities(model, test_loader, device)
    m = evaluate_with_threshold(tt, tp, threshold=th)
    pred = (tp >= th).astype(int)
    out = {"defense": defense, "seed": seed, "drop_rate": DROP_RATE, "best_round": int(server.best_round),
           "threshold": float(th), "accuracy": float(m["accuracy"]), "f1_binary": float(m["f1_binary"]),
           "precision_binary": float(m["precision_binary"]), "recall_binary": float(m["recall_binary"]),
           "roc_auc": float(m["roc_auc"]), "mcc": float(matthews_corrcoef(tt, pred) * 100),
           "pr_auc": float(average_precision_score(tt, tp) * 100),
           "confusion_matrix": m["confusion_matrix"], "participants_per_round": participants,
           "malicious_participants_per_round": [len(malicious & set(p)) for p in participants]}
    if defense == "hra":
        honest = set(range(args.num_clients)) - malicious
        last = set(excluded[-1])
        out.update({"excluded_per_round": excluded, "fallback_rounds": int(sum(fallback)),
                    "malicious_excluded_final_pct": len(malicious & last) / len(malicious) * 100,
                    "honest_wrongly_excluded_final_pct": len(honest & last) / len(honest) * 100})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=rr.SEEDS3)
    ap.add_argument("--defenses", nargs="+", default=["hra", "multi_krum"])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if args.rounds == 50 else f"_test{args.rounds}rounds"
    out_json = os.path.join(RESULTS_DIR, f"dropout{suffix}.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    dataset = FLDataset("data/cic", None, dataset_type="cicids2017",
                        model_type="mlp", val_ratio=0.1, seed=42, subsample_size=500000)

    for seed in args.seeds:
        for d in args.defenses:
            key = f"{d}|{seed}"
            if key in results:
                print(f"  [Skip] {key}")
                continue
            res = run_one(d, seed, dataset, args)
            if os.path.exists(out_json):
                results.update(json.load(open(out_json)))
            results[key] = res
            tmp = out_json + f".tmp{os.getpid()}"
            json.dump(results, open(tmp, "w"), indent=1)
            os.replace(tmp, out_json)
            print(f"  [Saved] {key}: F1={res['f1_binary']:.2f}%")

    print(f"\n{'=' * 50}\n  Dropout 20% - mean over completed seeds\n{'-' * 50}")
    for d in args.defenses:
        v = [x["f1_binary"] for k, x in results.items() if k.startswith(d + "|")]
        if v:
            print(f"  {d:<12} n={len(v)}  F1 = {np.mean(v):.2f} ± {(np.std(v, ddof=1) if len(v) > 1 else 0):.2f}")
    print(f"  Saved: {out_json}")


if __name__ == "__main__":
    main()