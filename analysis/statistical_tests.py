"""
Statistical tests (paired Wilcoxon + 95% CI)
=====================================================================
Paired Wilcoxon signed-rank test (same seeds) + 95% confidence intervals
(t-distribution, n-1 degrees of freedom) for:

  1. Table 7 - HRA vs every other defense under model poisoning (10 seeds)
     reads: results/defense_comparison_seeds/model_poisoning.json
  2. Unified HRA config vs previous per-attack config, both attacks (10 seeds)
     reads: results/unified_defense/unified_defense_results.json

No training - takes a few seconds. Writes results/statistical_tests.json and
prints the tables.

Usage:
  python3 analysis/statistical_tests.py      # run from the repository root
"""

import json
import os

import numpy as np
from scipy import stats

SEEDS = [42, 123, 7, 99, 500, 2024, 31, 77, 1001, 256]
METRICS = {"f1_binary": "F1", "roc_auc": "ROC-AUC", "mcc": "MCC"}
DC_JSON = os.path.join("results", "defense_comparison_seeds", "model_poisoning.json")
UNI_JSON = os.path.join("results", "unified_defense", "unified_defense_results.json")
OUT_JSON = os.path.join("results", "statistical_tests.json")


def ci95(x):
    x = np.asarray(x, dtype=float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def wilcoxon_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)


def compare(name_a, a, name_b, b, metric):
    ma, la, ha = ci95(a)
    mb, lb, hb = ci95(b)
    return {"A": name_a, "B": name_b, "metric": METRICS[metric],
            "A_mean": ma, "A_ci95": [la, ha], "B_mean": mb, "B_ci95": [lb, hb],
            "mean_diff": ma - mb, "A_wins": int(np.sum(np.asarray(a) > np.asarray(b))),
            "n": len(a), "wilcoxon_p": wilcoxon_p(a, b)}


def show(title, rows):
    print(f"\n{'=' * 104}\n  {title}\n{'-' * 104}")
    print(f"  {'A':<12}{'B':<20}{'Metric':<9}{'A mean [95% CI]':<26}{'B mean [95% CI]':<26}{'wins':>6}{'p':>9}")
    for r in rows:
        a = f"{r['A_mean']:.2f} [{r['A_ci95'][0]:.2f}, {r['A_ci95'][1]:.2f}]"
        b = f"{r['B_mean']:.2f} [{r['B_ci95'][0]:.2f}, {r['B_ci95'][1]:.2f}]"
        star = " *" if r["wilcoxon_p"] < 0.05 else ""
        print(f"  {r['A']:<12}{r['B']:<20}{r['metric']:<9}{a:<26}{b:<26}{r['A_wins']:>3}/{r['n']:<2}{r['wilcoxon_p']:>8.4f}{star}")
    print("  * = significant at p < 0.05 (paired Wilcoxon signed-rank, two-sided)")


def main():
    out = {}

    # 1) Table 7: HRA vs each defense (model poisoning)
    if os.path.exists(DC_JSON):
        d = json.load(open(DC_JSON))
        others = ["multi_krum", "trimmed_mean", "krum", "coordinate_median", "foolsgold", "fedavg"]
        rows = []
        for o in others:
            for m in METRICS:
                rows.append(compare("HRA", [d[f"hra|{s}"][m] for s in SEEDS],
                                    o, [d[f"{o}|{s}"][m] for s in SEEDS], m))
        out["table7_hra_vs_defenses"] = rows
        show("Table 7 - HRA vs each defense, model poisoning, 10 seeds", rows)
    else:
        print(f"  [Missing] {DC_JSON}")

    # 2) Unified vs previous configuration, both attacks
    if os.path.exists(UNI_JSON):
        u = json.load(open(UNI_JSON))
        rows = []
        for atk in ["label_flipping", "model_poisoning"]:
            for m in METRICS:
                a = [u[f"{atk}|unified|{s}"][m] for s in SEEDS]
                b = [u[f"{atk}|lf_tuned|{s}"][m] for s in SEEDS]
                r = compare("Unified", a, "Previous (a=0.1)", b, m)
                r["attack"] = atk
                rows.append(r)
        out["unified_vs_previous"] = rows
        show("Unified HRA vs previous config (label flipping rows first, then model poisoning), 10 seeds", rows)
    else:
        print(f"  [Missing] {UNI_JSON}")

    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\n  Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()