"""Output paths for main.py (original pipeline). The paper's results are in the folders written by the run_*.py scripts."""

import os
import json
import shutil

RESULTS_ROOT = "results"
BASELINE_DIR = os.path.join(RESULTS_ROOT, "baseline")
SUMMARY_DIR = os.path.join(RESULTS_ROOT, "summary")

BASELINE_JSON = os.path.join(BASELINE_DIR, "secure_baseline_no_fs.json")
BASELINE_SEED42_JSON = os.path.join(BASELINE_DIR, "secure_baseline_seed42.json")
SUMMARY_JSON = os.path.join(SUMMARY_DIR, "all_attacks_comparison.json")
SUMMARY_PNG = os.path.join(SUMMARY_DIR, "all_attacks_comparison.png")

LEGACY_MIGRATIONS = [
    ("Results_without_Feature_Select.json", BASELINE_JSON),
    ("cross_validation_results.json", None),  # keep legacy, do not overwrite canonical
    ("attack_results.json", os.path.join(RESULTS_ROOT, "label_flipping", "attack_no_defense.json")),
    ("defense_results.json", os.path.join(RESULTS_ROOT, "label_flipping", "attack_with_defense.json")),
]


def ensure_dirs():
    """Create standard results directory tree."""
    for d in [BASELINE_DIR, SUMMARY_DIR,
              os.path.join(RESULTS_ROOT, "label_flipping"),
              os.path.join(RESULTS_ROOT, "model_poisoning")]:
        os.makedirs(d, exist_ok=True)


def get_attack_dir(attack_type: str) -> str:
    return os.path.join(RESULTS_ROOT, attack_type)


def get_results_paths(attack_type: str) -> dict:
    """Return all output paths for one attack type."""
    base = get_attack_dir(attack_type)
    return {
        "dir": base,
        "attack_no_defense_json": os.path.join(base, "attack_no_defense.json"),
        "attack_with_defense_json": os.path.join(base, "attack_with_defense.json"),
        "attack_no_defense_dashboard": os.path.join(base, "attack_no_defense_dashboard.png"),
        "attack_with_defense_dashboard": os.path.join(base, "attack_with_defense_dashboard.png"),
        "comparison_baseline_vs_attack": os.path.join(base, "comparison_baseline_vs_attack.png"),
        "comparison_three_way": os.path.join(base, "comparison_three_way.png"),
        "reputation_trace": os.path.join(base, "reputation_trace.json"),
    }


def migrate_legacy_results():
    """Copy legacy flat JSON files into organized folders if targets missing."""
    ensure_dirs()
    for src_name, dst in LEGACY_MIGRATIONS:
        if dst is None:
            continue
        src = os.path.join(RESULTS_ROOT, src_name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  [Migrated] {src_name} -> {dst}")


def load_baseline_means(baseline_json: str = BASELINE_JSON) -> dict:
    """Load per-seed baseline JSON and return mean metrics."""
    if not os.path.exists(baseline_json):
        return {}
    with open(baseline_json) as f:
        data = json.load(f)
    keys = ["accuracy", "f1_binary", "recall_binary", "precision_binary", "roc_auc"]
    out = {"source": baseline_json, "seeds": list(data.keys())}
    for k in keys:
        vals = [v[k] for v in data.values() if k in v]
        if vals:
            out[f"{k}_mean"] = float(sum(vals) / len(vals))
    return out


def metrics_to_json(metrics: dict, extra: dict = None) -> dict:
    """Serialize test metrics for JSON output."""
    out = {
        "accuracy": float(metrics.get("accuracy", 0)),
        "f1_binary": float(metrics.get("f1_binary", 0)),
        "recall_binary": float(metrics.get("recall_binary", 0)),
        "precision_binary": float(metrics.get("precision_binary", 0)),
        "roc_auc": float(metrics.get("roc_auc", 0)),
    }
    if extra:
        out.update(extra)
    return out
