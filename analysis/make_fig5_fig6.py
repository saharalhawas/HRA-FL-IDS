"""
Figs. 5 and 6 of the paper (mean ± std over 10 seeds). Run from the repository root:
    python3 analysis/make_fig5_fig6.py
Reads results/remaining/baseline.json, results/defense_comparison_seeds/{label_flipping,model_poisoning}.json
and results/unified_defense/unified_defense_results.json; writes figures/Fig5_scenarios.png and
figures/Fig6_attack_defense_cycle.png.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = [42, 123, 7, 99, 500, 2024, 31, 77, 1001, 256]
base = json.load(open("results/remaining/baseline.json"))
lf = json.load(open("results/defense_comparison_seeds/label_flipping.json"))
mp = json.load(open("results/defense_comparison_seeds/model_poisoning.json"))
uni = json.load(open("results/unified_defense/unified_defense_results.json"))
os.makedirs("figures", exist_ok=True)


def m(x):
    return dict(acc=x["accuracy"], f1=x["f1_binary"], rec=x["recall_binary"], prec=x["precision_binary"], mcc=x["mcc"])


SC = [("Secure baseline", "#8a8a85", [m(base[f"clean_fedavg|{s}"]) for s in S]),
      ("Label flipping, no defense", "#f2a36b", [m(lf[f"fedavg|{s}"]) for s in S]),
      ("Label flipping + HRA", "#1aaf7a", [m(uni[f"label_flipping|unified|{s}"]) for s in S]),
      ("Model poisoning, no defense", "#e24a4a", [m(mp[f"fedavg|{s}"]) for s in S]),
      ("Model poisoning + HRA", "#2a78d6", [m(mp[f"hra|{s}"]) for s in S])]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#777"})

# ---------- Fig. 5 ----------
keys = [("acc", "Accuracy"), ("f1", "F1-score"), ("rec", "Recall"), ("prec", "Precision"), ("mcc", "MCC")]
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=300)
ax.yaxis.grid(True, color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
w, x = 0.16, np.arange(len(keys))
for i, (lab, col, v) in enumerate(SC):
    mu = np.array([np.mean([r[k] for r in v]) for k, _ in keys])
    sd = np.array([np.std([r[k] for r in v], ddof=1) for k, _ in keys])
    ax.bar(x + (i - 2) * w, mu, w * 0.92, color=col, label=lab, yerr=[np.minimum(sd, mu), np.minimum(sd, 100 - mu)],
           error_kw=dict(ecolor="#333", lw=1.1, capsize=2.5))
ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys]); ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=10)
fig.tight_layout(); fig.savefig("figures/Fig5_scenarios.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig. 6 ----------
f1 = {lab: [r["f1"] for r in v] for lab, _, v in SC}
panels = [("(a) Label flipping", "Label flipping", "#f2a36b", "#1aaf7a"),
          ("(b) Model poisoning", "Model poisoning", "#e24a4a", "#2a78d6")]
plt.rcParams.update({"font.size": 10})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), dpi=300, sharey=True)
for ax, (title, name, cn, ch) in zip(axs, panels):
    v = [f1["Secure baseline"], f1[f"{name}, no defense"], f1[f"{name} + HRA"]]
    mu = [np.mean(a) for a in v]; sd = [np.std(a, ddof=1) for a in v]
    ax.yaxis.grid(True, color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    ax.bar([0, 1, 2], mu, 0.55, color=["#8a8a85", cn, ch],
           yerr=[np.minimum(sd, mu), np.minimum(sd, 100 - np.array(mu))], error_kw=dict(ecolor="#333", lw=1.1, capsize=3))
    for i, (a, b) in enumerate(zip(mu, sd)):
        ax.text(i, a + b + 2, f"{a:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(0.85, mu[1] + 33), xytext=(0.3, mu[0] - 12), arrowprops=dict(arrowstyle="->", color="#e24a4a", lw=1.6))
    ax.text(0.55, mu[0] - 35, f"−{mu[0] - mu[1]:.2f} pp", color="#c03030", ha="center", fontsize=9.5)
    ax.annotate("", xy=(1.7, mu[2] - 12), xytext=(1.15, mu[1] + 33), arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1.6))
    ax.text(1.5, mu[0] - 35, f"+{mu[2] - mu[1]:.2f} pp", color="#1f5fb0", ha="center", fontsize=9.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Secure\nbaseline", f"{name},\nno defense", f"{name},\nHRA"], fontsize=9)
    ax.set_title(title, loc="left", fontsize=11); ax.set_ylim(0, 112)
axs[0].set_ylabel("F1-score (%)")
fig.tight_layout(); fig.savefig("figures/Fig6_attack_defense_cycle.png", bbox_inches="tight"); plt.close(fig)
print("Saved figures/Fig5_scenarios.png and figures/Fig6_attack_defense_cycle.png")
