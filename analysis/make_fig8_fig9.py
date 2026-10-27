"""
Figs. 8 and 9 (EMA reputation, model poisoning, seed 42) in the paper's figure style.
    python3 make_fig8_fig9.py [path/to/reputation_seed42.json]
Default input: results/final_checks/reputation_seed42.json. Output: figures/.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

src = sys.argv[1] if len(sys.argv) > 1 else "results/final_checks/reputation_seed42.json"
ema = np.array(json.load(open(src))["ema_reputation_per_round"])
R, K = ema.shape; NM = 4; rounds = np.arange(1, R + 1)
names = [f.name for f in font_manager.fontManager.ttflist]
serif = next((f for f in ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"] if f in names), "serif")
plt.rcParams.update({"font.family": serif, "font.size": 11, "axes.edgecolor": "#333",
                     "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "stix"})
os.makedirs("figures", exist_ok=True)

# ---------- Fig. 8: heatmap ----------
fig, ax = plt.subplots(figsize=(10, 3.0), dpi=300)
im = ax.imshow(ema.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest",
               extent=[0.5, R + 0.5, K - 0.5, -0.5])
ax.axhline(NM - 0.5, color="white", lw=3)
ax.set_yticks(range(K)); ax.set_yticklabels([f"C{i} ({'M' if i < NM else 'H'})" for i in range(K)])
ax.set_xticks(range(1, R + 1, 5)); ax.set_xlabel("Communication Round", fontsize=12)
ax.set_xticks(np.arange(0.5, R + 1, 1), minor=True); ax.grid(False)
for x in range(1, R + 1, 5):
    ax.axvline(x, color="white", lw=0.5, alpha=0.35)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03); cb.set_label("EMA Reputation", fontsize=12)
fig.tight_layout(); fig.savefig("figures/Fig8_reputation_heatmap.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig. 9: traces and gap ----------
mal, hon = ema[:, :NM].mean(1), ema[:, NM:].mean(1); gap = hon - mal
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.3), dpi=300, sharey=True)
for i in range(K):
    a.plot(rounds, ema[:, i], color="#e53935" if i < NM else "#5b8fe0", lw=0.8, alpha=0.45)
a.plot(rounds, mal, color="#e53935", lw=2.4, label=f"Malicious (mean, n={NM})")
a.plot(rounds, hon, color="#2f6fe0", lw=2.4, label=f"Honest (mean, n={K - NM})")
a.axhline(0.05, color="#777", lw=0.8, ls="--")
a.text(R, 0.075, r"$\tau$ = 0.05", ha="right", fontsize=9, color="#555")
a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=9, framealpha=0.9)
a.set_title("(a) Individual Client Traces", fontsize=12)
b.plot(rounds, gap, color="#7c4ddf", lw=2, label="Reputation Gap")
b.fill_between(rounds, 0, gap, color="#7c4ddf", alpha=0.3)
b.axhline(0, color="#777", lw=0.8, ls="--")
b.legend(loc="upper right", fontsize=9, framealpha=0.9)
b.set_title("(b) Honest–Malicious Gap", fontsize=12)
for ax in (a, b):
    ax.grid(True, color="#e6e6e6", lw=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("Communication Round", fontsize=12); ax.set_xlim(-1, R + 1)
a.set_ylabel("EMA Reputation Score", fontsize=12); a.set_ylim(-0.03, 1.05)
fig.tight_layout(); fig.savefig("figures/Fig9_reputation_traces.png", bbox_inches="tight"); plt.close(fig)
print(f"Saved figures/Fig8_reputation_heatmap.png and figures/Fig9_reputation_traces.png (gap after round 10: "
      f"{gap[10:].min():.2f}-{gap[10:].max():.2f})")
