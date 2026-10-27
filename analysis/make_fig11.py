"""Fig. 11: F1 vs attack intensity, (a) model poisoning and (b) label flipping (mean over 3 seeds).
Values are the Table 13 means (seeds 42, 123, 7) computed from results/attack_sensitivity/*.json and,
for the default rows, from the main runs (results/defense_comparison_seeds, results/unified_defense)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = 28.90  # collapsed F1
D = {"HRA (Ours)": ("#2a78d6", "o"), "Multi-Krum": ("#eb6834", "s"), "Krum": ("#1aaf7a", "^"), "Trimmed Mean": ("#6a4fd8", "D")}
MP = {"ratio": ([0.1, 0.2, 0.3, 0.4, 0.5], {"HRA (Ours)": [92.90, 93.17, 92.75, 92.81, C], "Multi-Krum": [93.17, 93.24, 93.04, 92.72, C],
                                             "Krum": [92.56, 91.11, 92.23, 89.92, 92.85], "Trimmed Mean": [93.24, 91.88, 91.35, 89.63, C]}),
      "scale": ([1.5, 2, 5, 10], {"HRA (Ours)": [92.72, 92.32, 92.81, 92.36], "Multi-Krum": [92.52, 92.63, 92.72, 92.76],
                                  "Krum": [90.08, 90.08, 89.92, 89.92], "Trimmed Mean": [92.79, 91.52, 89.63, 90.43]}),
      "noise": ([0, 2, 5, 10], {"HRA (Ours)": [91.76, 92.35, 92.81, 92.35], "Multi-Krum": [92.56, 92.80, 92.72, 92.60],
                                "Krum": [92.19, 89.92, 89.92, 89.92], "Trimmed Mean": [87.78, 89.39, 89.63, 91.48]})}
LF = {"ratio": ([0.1, 0.2, 0.3, 0.4, 0.5], {"HRA (Ours)": [93.33, 93.38, 93.51, 93.32, 91.68], "Multi-Krum": [92.94, 93.02, 93.04, 93.44, 90.06],
                                             "Krum": [92.17, 92.40, 91.83, 91.98, 92.81], "Trimmed Mean": [92.17, 92.69, 92.53, 92.10, C]}),
      "scale": ([1.5, 2, 5, 10], {"HRA (Ours)": [92.98, 92.96, 93.32, 93.58], "Multi-Krum": [93.13, 93.25, 93.44, 93.31],
                                  "Krum": [91.98, 91.98, 91.98, 91.98], "Trimmed Mean": [91.00, 92.17, 92.10, 92.63]})}
XL = {"ratio": "Malicious-client ratio m/K", "scale": "Scaling factor s", "noise": "Noise level σ"}
DEF = {"ratio": 0.4, "scale": 5, "noise": 5}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#777", "xtick.color": "#444", "ytick.color": "#444"})
fig, axs = plt.subplots(2, 3, figsize=(11, 6.6), dpi=300, sharey=True)
LO = 86
for row, (data, title) in enumerate([(MP, "(a) Model poisoning"), (LF, "(b) Label flipping (no noise parameter)")]):
    for col, part in enumerate(["ratio", "scale", "noise"]):
        ax = axs[row, col]
        if part not in data:
            ax.axis("off"); continue
        xs, vals = data[part]; pos = np.arange(len(xs))
        ax.yaxis.grid(True, color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
        ax.axvspan(xs.index(DEF[part]) - 0.2, xs.index(DEF[part]) + 0.2, color="#e6e5e1", zorder=0)
        collapsed = []
        for name, (col_, mk) in D.items():
            y = np.array(vals[name], float); ok = y > 50
            ax.plot(pos[ok], y[ok], color=col_, marker=mk, lw=3 if name.startswith("HRA") else 2,
                    ms=8 if name.startswith("HRA") else 6, mec="white", label=name, zorder=3 if name.startswith("HRA") else 2)
            for p in pos[~ok]:
                ax.plot(p, LO + 0.3, marker="v", ms=11, mfc="none", mec=col_, mew=2, clip_on=False)
                collapsed.append(name.replace(" (Ours)", ""))
        if collapsed:
            ax.text(pos[-1] - 0.15, LO + 1.0, "▼ collapse (F1 28.90):\n" + ", ".join(collapsed), ha="right", fontsize=8.5, color="#444")
        ax.set_xticks(pos); ax.set_xticklabels([str(v) for v in xs]); ax.set_xlabel(XL[part])
        ax.set_ylim(LO, 95)
        if col == 0:
            ax.set_ylabel("F1-score (%)"); ax.set_title(title, loc="left", fontsize=12)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("figures/Fig11_attack_intensity.png", bbox_inches="tight")
print("Saved figures/Fig11_attack_intensity.png")
