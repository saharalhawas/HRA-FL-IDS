"""
Paper figures for the revision (run from the project folder, after all experiments):
  python3 make_figures.py
Reads results/... JSON files and writes PNGs (300 dpi) to figures/.
"""
import json, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
S = [42, 123, 7, 99, 500, 2024, 31, 77, 1001, 256]
OUT = "figures/"
os.makedirs(OUT, exist_ok=True)
DC = "results/defense_comparison_seeds/model_poisoning.json"
ABL = "results/remaining/ablation.json"
NOFB = "results/final_checks/nofallback.json"

# ---------- Fig A: Table 7 grouped bars ----------
d = json.load(open(DC))
names = {"hra": "HRA (ours)", "multi_krum": "Multi-Krum", "trimmed_mean": "Trimmed Mean", "krum": "Krum",
         "coordinate_median": "Coord. Median", "foolsgold": "FoolsGold", "fedavg": "FedAvg"}
order = list(names)
f1 = [[d[f"{k}|{s}"]["f1_binary"] for s in S] for k in order]
mcc = [[d[f"{k}|{s}"]["mcc"] for s in S] for k in order]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=300)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
b1 = ax.bar(x - w / 2 - 0.01, [np.mean(v) for v in f1], w, yerr=[np.std(v, ddof=1) for v in f1],
            color=BLUE, label="F1-score", error_kw=dict(ecolor=INK2, lw=1, capsize=2.5))
b2 = ax.bar(x + w / 2 + 0.01, [np.mean(v) for v in mcc], w, yerr=[np.std(v, ddof=1) for v in mcc],
            color=ORANGE, label="MCC", error_kw=dict(ecolor=INK2, lw=1, capsize=2.5))
for xi, v in zip(x, f1):
    ax.text(xi - w / 2, np.mean(v) + np.std(v, ddof=1) + 2, f"{np.mean(v):.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels([names[k] for k in order])
ax.set_ylabel("Score (%)"); ax.set_ylim(0, 110); ax.set_yticks(range(0, 101, 20))
ax.legend(frameon=False, loc="upper right", ncol=2)
ax.set_title("Defense comparison under model poisoning (40% malicious clients)", loc="left", fontsize=11, color=INK)
fig.text(0.01, -0.02, "Mean ± standard deviation over 10 random seeds. Labels show mean F1.", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(OUT + "Fig_Table7_defense_comparison.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig B: Table 8 ablation ----------
abl = json.load(open(ABL))
nf = json.load(open(NOFB))
cfg = [("A_fedavg", "A. FedAvg"), ("B_median_only", "B. Median only"), ("C_median_instant_rep", "C. + Instant rep."),
       ("D_median_ema_no_excl", "D. + EMA rep."), ("E_full_hra", "E. Full HRA"), ("F_hra_no_fallback", "F. HRA w/o fallback")]
vals = []
for c, _ in cfg:
    src = nf if c.startswith("F_") else abl
    vals.append([v["f1_binary"] for v in src.values() if v["config"] == c])
fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=300)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
x = np.arange(len(cfg))
ax.bar(x, [np.mean(v) for v in vals], 0.6, yerr=[np.std(v, ddof=1) for v in vals], color=BLUE,
       error_kw=dict(ecolor=INK2, lw=1, capsize=3))
for xi, v in zip(x, vals):
    ax.scatter([xi + o for o in (-0.12, 0, 0.12)], v, s=14, color="#0d366b", zorder=3, edgecolor="white", lw=0.6)
    ax.text(xi, max(np.mean(v) + np.std(v, ddof=1), max(v)) + 2.5, f"{np.mean(v):.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in cfg]); ax.set_ylabel("F1-score (%)"); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
ax.set_title("Ablation of HRA components (model poisoning, 40% malicious clients)", loc="left", fontsize=11, color=INK)
fig.text(0.01, -0.02, "Bars: mean ± standard deviation over 3 seeds; dots: individual seeds (42, 123, 7).", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(OUT + "Fig_Table8_ablation.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig C: client-level detection per round ----------
M, H = set(range(4)), set(range(4, 10))
mal, hon = [], []
for rd in range(50):
    a, b = [], []
    for s in S:
        e = set(d[f"hra|{s}"]["excluded_per_round"][rd])
        a.append(100 * len(e & M) / 4); b.append(100 * len(e & H) / 6)
    mal.append(np.mean(a)); hon.append(np.mean(b))
r = np.arange(1, 51)
fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=300)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.plot(r, mal, color=BLUE, lw=2); ax.plot(r, hon, color=ORANGE, lw=2)
ax.text(51, mal[-1], "Malicious clients excluded", color=INK, fontsize=8, va="center")
ax.text(51, hon[-1] + 3, "Honest clients wrongly excluded", color=INK, fontsize=8, va="center")
ax.axvline(9.4, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.text(9.9, 50, "all attackers excluded\n(mean round 9.4)", fontsize=8, color=INK2)
ax.set_xlim(1, 50); ax.set_ylim(0, 105); ax.set_xlabel("Federated round"); ax.set_ylabel("Clients (%)")
ax.set_title("HRA client-level detection over rounds (model poisoning)", loc="left", fontsize=11, color=INK)
fig.text(0.01, -0.03, "Mean over 10 random seeds; 4 malicious and 6 honest clients.", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(OUT + "Fig_client_detection_per_round.png", bbox_inches="tight"); plt.close(fig)

# per-round table for Excel / paper
json.dump({"round": r.tolist(), "malicious_excluded_pct": mal, "honest_wrongly_excluded_pct": hon},
          open(OUT + "detection_per_round.json", "w"), indent=1)
print("Saved 3 figures + detection_per_round.json in", OUT)
