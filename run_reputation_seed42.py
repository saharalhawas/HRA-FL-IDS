"""
Figs. 8 and 9: reputation of every client per round (model poisoning, seed 42)
==============================================================================
Re-runs the main HRA model-poisoning experiment for seed 42 (unified config,
40% malicious, 50 rounds) and records, after every round, the EMA reputation
and the per-round (normalized) score of all 10 clients. Then draws:
  figures/Fig8_reputation_heatmap.png
  figures/Fig9_reputation_traces.png
and saves the numbers to results/final_checks/reputation_seed42.json.
The final paper figures are drawn from this JSON by analysis/make_fig8_fig9.py.

The run is identical to the seed-42 HRA run of the main results
(expected F1 = 91.79%); the script prints the F1 so you can check.

Put this file in the same folder as run_remaining.py and run:
  caffeinate -dims python3 run_reputation_seed42.py
Quick test (2 rounds):
  python3 run_reputation_seed42.py --rounds 2
Safe to run in parallel with another experiment (it writes its own files).
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_remaining as rr
from data_loader import FLDataset
from federated_server import FederatedServer

LOG = {"ema": [], "score": []}
_orig = FederatedServer.aggregate_hybrid_robust


def _logged(self, client_weights, client_sizes):
    info = _orig(self, client_weights, client_sizes)
    LOG["ema"].append([float(x) for x in info["ema_reputations"]])
    LOG["score"].append([float(x) for x in info["round_reputations"]])
    return info


FederatedServer.aggregate_hybrid_robust = _logged


def draw(ema, out_dir, n_mal=4):
    ema = np.array(ema)                       # rounds x clients
    R, K = ema.shape
    rounds = np.arange(1, R + 1)
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = [f"C{i} ({'M' if i < n_mal else 'H'})" for i in range(K)]

    # Fig. 8: heatmap
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=300)
    im = ax.imshow(ema.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   extent=[0.5, R + 0.5, K - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(K)); ax.set_yticklabels(labels)
    ax.axhline(n_mal - 0.5, color="black", lw=1.2)
    ax.set_xlabel("Federated round"); ax.set_title("EMA reputation per client (model poisoning, seed 42)", loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("Reputation")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "Fig8_reputation_heatmap.png"), bbox_inches="tight"); plt.close(fig)

    # Fig. 9: (a) traces, (b) honest-malicious gap
    mal, hon = ema[:, :n_mal].mean(1), ema[:, n_mal:].mean(1)
    gap = hon - mal
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=300)
    for i in range(K):
        a.plot(rounds, ema[:, i], color="#e24a4a" if i < n_mal else "#2a78d6", lw=0.7, alpha=0.45)
    a.plot(rounds, mal, color="#b01f1f", lw=2.4, label="Malicious mean")
    a.plot(rounds, hon, color="#14549e", lw=2.4, label="Honest mean")
    a.axhline(0.05, color="#555", lw=0.9, ls="--"); a.text(R, 0.07, "τ = 0.05", ha="right", fontsize=8, color="#555")
    a.set_xlabel("Federated round"); a.set_ylabel("EMA reputation"); a.set_ylim(0, 1.05)
    a.set_title("(a) Per-client reputation", loc="left", fontsize=11); a.legend(frameon=False, fontsize=9)
    b.plot(rounds, gap, color="#333", lw=2)
    b.set_xlabel("Federated round"); b.set_ylabel("Honest mean − malicious mean"); b.set_ylim(0, 1.05)
    b.set_title("(b) Honest–malicious reputation gap", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "Fig9_reputation_traces.png"), bbox_inches="tight"); plt.close(fig)
    if R >= 11:
        print(f"  Gap after round 10: mean {gap[10:].mean():.3f}, range {gap[10:].min():.3f}-{gap[10:].max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    out_dir = os.path.join("results", "final_checks")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if args.rounds == 50 else f"_test{args.rounds}rounds"
    out_json = os.path.join(out_dir, f"reputation_seed42{suffix}.json")

    dataset = FLDataset("data/cic", None, dataset_type="cicids2017",
                        model_type="mlp", val_ratio=0.1, seed=42, subsample_size=500000)
    res = rr.run_one("E_full_hra", "hra", rr.hra(), {}, 42, dataset, args)
    res["ema_reputation_per_round"] = LOG["ema"]
    res["round_score_per_round"] = LOG["score"]
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"\n  F1 = {res['f1_binary']:.2f}%  (main seed-42 run: 91.79%)")
    print(f"  Saved: {out_json}")
    draw(LOG["ema"], "figures" + suffix)
    print(f"  Figures saved in figures{suffix}/")


if __name__ == "__main__":
    main()