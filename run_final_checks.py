"""
Additional robustness checks (one file, four parts)
====================================================
  kmin_warmup : HRA sensitivity to Kmin and warm-up                                   15 runs
  nofallback  : HRA without median fallback (ablation row F)                           3 runs
  coldstart   : attacker that attacks ONLY in rounds 1-10, while reputation is still
                forming -> HRA and Multi-Krum                                          6 runs
  scalability : server aggregation time for K = 10, 20, 50, 100 clients
                (synthetic updates, no training)                                      ~2 min
                
Same setting as run_remaining.py: model poisoning (s=5, sigma=5), 10 clients, 40%
malicious, Non-IID, 50 rounds, seeds 42 / 123 / 7. Saves after every run; re-run the
same command to continue where it stopped.

Usage (put this file in the same folder as run_remaining.py):
  caffeinate -dims python3 run_final_checks.py --part kmin_warmup
  caffeinate -dims python3 run_final_checks.py --part nofallback
  caffeinate -dims python3 run_final_checks.py --part coldstart
  python3 run_final_checks.py --part scalability        # run ALONE, when nothing else runs
  python3 run_final_checks.py --part coldstart --rounds 2 --seeds 42   # quick test
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import run_remaining as rr
from data_loader import FLDataset
from defense_config import DefenseConfig
from federated_server import FederatedServer
from main import set_seed
from model import MLPModel

RESULTS_DIR = os.path.join("results", "final_checks")

# "early" = attack only in rounds 1-10 (0-based r < 10), honest afterwards
_orig_attack_active = rr.attack_active
def _attack_active(pattern, r):
    if pattern == "early":
        return r < 10
    return _orig_attack_active(pattern, r)
rr.attack_active = _attack_active


def part_configs(part):
    if part == "kmin_warmup":
        return [("kmin=2", "hra", rr.hra(kmin=2), {}),
                ("kmin=6", "hra", rr.hra(kmin=6), {}),
                ("warmup=1", "hra", rr.hra(warmup=1), {}),
                ("warmup=4", "hra", rr.hra(warmup=4), {}),
                ("warmup=12", "hra", rr.hra(warmup=12), {})]
    if part == "nofallback":
        return [("F_hra_no_fallback", "hra", rr.hra(fallback=False), {})]
    if part == "coldstart":
        return [("early|hra", "hra", DefenseConfig.unified(), {"pattern": "early"}),
                ("early|multi_krum", "multi_krum", DefenseConfig(), {"pattern": "early"})]
    raise ValueError(part)


def run_scalability(dataset, out_json, repeats=20):
    """Time each aggregation rule for growing K using synthetic client updates."""
    set_seed(42)
    device = torch.device("cpu")
    rules = {"fedavg": "aggregate_fedavg", "coordinate_median": "aggregate_coordinate_median",
             "trimmed_mean": "aggregate_trimmed_mean", "krum": "aggregate_krum",
             "multi_krum": "aggregate_multi_krum", "foolsgold": "aggregate_foolsgold",
             "hra": "aggregate_hybrid_robust"}
    res = {}
    for K in [10, 20, 50, 100]:
        base = MLPModel(input_dim=dataset.input_dim).state_dict()
        g = torch.Generator().manual_seed(K)
        w = []
        for _ in range(K):
            sd = copy.deepcopy(base)
            for k, v in sd.items():
                if v.is_floating_point():
                    sd[k] = v + 0.01 * torch.randn(v.shape, generator=g)
            w.append(sd)
        s = [1000] * K
        res[K] = {}
        for name, fn in rules.items():
            server = FederatedServer(MLPModel, device, K,
                                     defense_config=DefenseConfig.unified() if name == "hra" else DefenseConfig(),
                                     input_dim=dataset.input_dim)
            getattr(server, fn)(w, s)                      # warm-up call
            t = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                getattr(server, fn)(w, s)
                t.append((time.perf_counter() - t0) * 1000)
            res[K][name] = {"mean": float(np.mean(t)), "std": float(np.std(t, ddof=1))}
        print(f"  K={K:<4} " + "  ".join(f"{n}={v['mean']:.1f}ms" for n, v in res[K].items()))
    json.dump({"aggregation_ms_by_K": res, "repeats": repeats}, open(out_json, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["kmin_warmup", "nofallback", "coldstart", "scalability"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--rounds", type=int, default=50)
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

    if args.part == "scalability":
        run_scalability(dataset, out_json, repeats=20 if args.rounds == 50 else 2)
        print(f"\n  Saved: {out_json}")
        return

    seeds = args.seeds or rr.SEEDS3
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    configs = part_configs(args.part)
    for seed in seeds:
        for name, agg, dcfg, akw in configs:
            key = f"{name}|{seed}"
            if key in results:
                print(f"  [Skip] {key}")
                continue
            res = rr.run_one(name, agg, dcfg, akw, seed, dataset, args)
            if os.path.exists(out_json):   # re-read so several terminals can share one file
                results.update(json.load(open(out_json)))
            results[key] = res
            tmp = out_json + f".tmp{os.getpid()}"
            json.dump(results, open(tmp, "w"), indent=2)
            os.replace(tmp, out_json)
            print(f"  [Saved] {key}: F1={res['f1_binary']:.2f}% AUC={res['roc_auc']:.2f}%")

    print(f"\n{'=' * 60}\n  {args.part} — mean over completed seeds\n{'-' * 60}")
    if os.path.exists(out_json):
        results.update(json.load(open(out_json)))
    for name, *_ in configs:
        v = [x for k, x in results.items() if k.rsplit("|", 1)[0] == name]
        if v:
            f1 = np.array([x["f1_binary"] for x in v])
            print(f"  {name:<22} n={len(v):<3} F1={f1.mean():6.2f} ± {(f1.std(ddof=1) if len(f1) > 1 else 0):.2f}")
    print("=" * 60)
    print(f"  Saved: {out_json}")


if __name__ == "__main__":
    main()