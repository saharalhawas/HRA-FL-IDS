"""
Additional experiments (one file, five parts)
==============================================
  hparam   : HRA sensitivity to alpha and tau                  18 runs
  adaptive : on-off, late-start and stealthy attackers         18 runs
  ablation : HRA component ablation                            15 runs
  baseline : clean FL without attack + confusion matrix        10 runs (fast)
  overhead : aggregation time per defense (run ALONE)          ~1 h

All parts use model poisoning, 10 clients, 40% malicious, Non-IID, 50 rounds,
unless stated otherwise. Results are saved after every run; re-run the same
command to resume.

Usage:
  python3 run_remaining.py --part hparam
  python3 run_remaining.py --part adaptive
  python3 run_remaining.py --part ablation
  python3 run_remaining.py --part baseline
  python3 run_remaining.py --part overhead      # run when nothing else is running
  python3 run_remaining.py --part hparam --rounds 2 --seeds 42   # quick test
"""

import argparse
import json
import os
import time

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

RESULTS_DIR = os.path.join("results", "remaining")
SEEDS3 = [42, 123, 7]
SEEDS10 = [42, 123, 7, 99, 500, 2024, 31, 77, 1001, 256]


def hra(alpha=0.35, tau=0.05, instant=False, kmin=4, fallback=True, warmup=8):
    return DefenseConfig(alpha=alpha, use_instant_rep=instant, hard_exclude_threshold=tau,
                         min_active_clients=kmin, use_median_fallback=fallback,
                         min_checkpoint_round=warmup)


def mp_cfg(scale=5.0, noise=5.0):
    c = AttackConfig.for_attack("model_poisoning")
    c.scale_factor, c.noise_std = scale, noise
    return c


# ---------------------------------------------------------------- configurations
def part_configs(part):
    """Return list of (name, aggregation, defense_cfg, attack_kwargs)."""
    if part == "hparam":
        out = []
        for a in [0.1, 0.2, 0.5]:
            out.append((f"alpha={a}", "hra", hra(alpha=a), {}))
        for t in [None, 0.02, 0.1]:
            out.append((f"tau={t}", "hra", hra(tau=t), {}))
        return out
    if part == "adaptive":
        out = []
        for pat in ["onoff", "late", "stealth"]:
            for agg in ["hra", "multi_krum"]:
                out.append((f"{pat}|{agg}", agg, DefenseConfig.unified() if agg == "hra" else DefenseConfig(),
                            {"pattern": pat}))
        return out
    if part == "ablation":
        return [
            ("A_fedavg", "fedavg", DefenseConfig(), {}),
            ("B_median_only", "coordinate_median", DefenseConfig(), {}),
            ("C_median_instant_rep", "hra", hra(instant=True, tau=None, kmin=1, fallback=False, warmup=1), {}),
            ("D_median_ema_no_excl", "hra", hra(tau=None, kmin=1, fallback=False, warmup=1), {}),
            ("E_full_hra", "hra", DefenseConfig.unified(), {}),
        ]
    if part == "baseline":
        return [("clean_fedavg", "fedavg", DefenseConfig(), {"no_attack": True})]
    raise ValueError(part)


def attack_active(pattern, r):
    """Whether malicious clients attack in round r (0-based)."""
    if pattern == "onoff":      # 5 benign rounds, 5 attack rounds, repeated
        return (r // 5) % 2 == 1
    if pattern == "late":       # behave honestly for 20 rounds to build reputation, then attack
        return r >= 20
    return True                 # constant (including stealth)


def run_one(name, agg, dcfg, akw, seed, dataset, args):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.reshuffle(seed)
    client_data = dataset.split_data(args.num_clients, iid=False)
    val_loader, test_loader = dataset.get_val_loader(), dataset.get_test_loader()
    server = FederatedServer(MLPModel, device, args.num_clients,
                             defense_config=dcfg, input_dim=dataset.input_dim)

    no_attack = akw.get("no_attack", False)
    pattern = akw.get("pattern", "constant")
    acfg = mp_cfg(1.5, 0.0) if pattern == "stealth" else mp_cfg()
    n_mal = 0 if no_attack else int(round(args.num_clients * 0.4))
    malicious = set(range(n_mal))
    clients = []
    for i in range(args.num_clients):
        is_mal = i in malicious
        clients.append(FederatedClient(
            i, client_data[i], device, model_type="mlp", is_malicious=is_mal,
            attack_type="model_poisoning" if is_mal else None, batch_size=args.batch_size,
            loss_type="ce", global_class_weights=getattr(dataset, "global_class_weights", None),
            attack_config=acfg if is_mal else None))

    patience = 7 if no_attack else args.rounds
    excluded = []
    for r in tqdm(range(args.rounds), desc=f"{name} seed={seed}"):
        on = attack_active(pattern, r)
        for c in clients:
            if c.is_malicious:
                c.attack_config = acfg if on else None
        w, s = [], []
        for c in clients:
            w.append(c.train_local_model(server.get_global_model(), epochs=args.local_epochs, lr=args.lr))
            s.append(c.get_update_size())
        if agg == "hra":
            info = server.aggregate_hybrid_robust(w, s)
            thr = dcfg.hard_exclude_threshold if dcfg.hard_exclude_threshold is not None else 0.05
            excluded.append([int(i) for i in np.where(np.array(info["ema_reputations"]) < thr)[0]])
        elif agg == "multi_krum":
            server.aggregate_multi_krum(w, s)
        elif agg == "coordinate_median":
            server.aggregate_coordinate_median(w, s)
        else:
            server.aggregate_fedavg(w, s)
        server.checkpoint_if_best(server.evaluate(val_loader), r + 1)
        if server.should_stop_early(patience):
            break

    server.restore_best_model()
    model = server.get_global_model()
    vt, vp = get_probabilities(model, val_loader, device)
    _, th, _ = sweep_thresholds(vt, vp)
    tt, tp = get_probabilities(model, test_loader, device)
    m = evaluate_with_threshold(tt, tp, threshold=th)
    pred = (tp >= th).astype(int)
    out = {"config": name, "seed": seed, "best_round": int(server.best_round), "threshold": float(th),
           "accuracy": float(m["accuracy"]), "f1_binary": float(m["f1_binary"]),
           "precision_binary": float(m["precision_binary"]), "recall_binary": float(m["recall_binary"]),
           "roc_auc": float(m["roc_auc"]), "mcc": float(matthews_corrcoef(tt, pred) * 100),
           "pr_auc": float(average_precision_score(tt, tp) * 100),
           "confusion_matrix": m["confusion_matrix"], "rounds_run": r + 1}
    if excluded and malicious:
        honest = set(range(args.num_clients)) - malicious
        last = set(excluded[-1])
        out.update({"malicious_excluded_final_pct": len(malicious & last) / len(malicious) * 100,
                    "honest_wrongly_excluded_final_pct": len(honest & last) / len(honest) * 100,
                    "excluded_per_round": excluded})
    return out


def run_overhead(dataset, args, out_json):
    """Time each aggregation rule on identical client updates (10 rounds)."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.reshuffle(42)
    client_data = dataset.split_data(args.num_clients, iid=False)
    clients = [FederatedClient(i, client_data[i], device, model_type="mlp", is_malicious=i < 4,
                               attack_type="model_poisoning" if i < 4 else None, batch_size=args.batch_size,
                               loss_type="ce", global_class_weights=getattr(dataset, "global_class_weights", None),
                               attack_config=mp_cfg() if i < 4 else None) for i in range(args.num_clients)]
    rules = {"fedavg": "aggregate_fedavg", "coordinate_median": "aggregate_coordinate_median",
             "trimmed_mean": "aggregate_trimmed_mean", "krum": "aggregate_krum",
             "multi_krum": "aggregate_multi_krum", "foolsgold": "aggregate_foolsgold",
             "hra": "aggregate_hybrid_robust"}
    servers = {k: FederatedServer(MLPModel, device, args.num_clients,
                                  defense_config=DefenseConfig.unified() if k == "hra" else DefenseConfig(),
                                  input_dim=dataset.input_dim) for k in rules}
    ref = servers["hra"]
    times = {k: [] for k in rules}
    for _ in tqdm(range(args.overhead_rounds), desc="overhead"):
        w = [c.train_local_model(ref.get_global_model(), epochs=args.local_epochs, lr=args.lr) for c in clients]
        s = [c.get_update_size() for c in clients]
        for k, fn in rules.items():
            servers[k].global_model.load_state_dict(ref.get_global_model().state_dict())
            reps = 20
            t0 = time.perf_counter()
            for _ in range(reps):
                getattr(servers[k], fn)(w, s)
            times[k].append((time.perf_counter() - t0) / reps * 1000)
        ref.aggregate_hybrid_robust(w, s)
    model = MLPModel(input_dim=dataset.input_dim)
    n_params = sum(p.numel() for p in model.parameters())
    res = {"aggregation_ms": {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                              for k, v in times.items()},
           "parameters": int(n_params), "model_size_kb": round(n_params * 4 / 1024, 2),
           "rounds_timed": args.overhead_rounds, "repeats_per_round": 20}
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\n  {'Defense':<20}{'ms / round':>14}")
    for k, v in res["aggregation_ms"].items():
        print(f"  {k:<20}{v['mean']:>9.2f} ± {v['std']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["hparam", "adaptive", "ablation", "baseline", "overhead"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--overhead-rounds", type=int, default=10)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if args.rounds == 50 else f"_test{args.rounds}rounds"
    out_json = os.path.join(RESULTS_DIR, f"{args.part}{suffix}.json")
    dataset = FLDataset("data/cic", None, dataset_type="cicids2017",
                        model_type="mlp", val_ratio=0.1, seed=42, subsample_size=500000)

    if args.part == "overhead":
        if args.rounds != 50:
            args.overhead_rounds = 2
        run_overhead(dataset, args, out_json)
        return

    seeds = args.seeds or (SEEDS10 if args.part == "baseline" else SEEDS3)
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    configs = part_configs(args.part)
    for seed in seeds:
        for name, agg, dcfg, akw in configs:
            key = f"{name}|{seed}"
            if key in results:
                print(f"  [Skip] {key}")
                continue
            res = run_one(name, agg, dcfg, akw, seed, dataset, args)
            results[key] = res
            json.dump(results, open(out_json, "w"), indent=2)
            print(f"  [Saved] {key}: F1={res['f1_binary']:.2f}% AUC={res['roc_auc']:.2f}%")

    print(f"\n{'=' * 60}\n  {args.part} — mean over completed seeds\n{'-' * 60}")
    for name, *_ in configs:
        v = [x for k, x in results.items() if k.rsplit("|", 1)[0] == name]
        if v:
            f1 = np.array([x["f1_binary"] for x in v])
            print(f"  {name:<26} n={len(v):<3} F1={f1.mean():6.2f} ± {(f1.std(ddof=1) if len(f1) > 1 else 0):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()