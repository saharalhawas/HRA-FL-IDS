"""
FL-IDS core pipeline
====================
Provides run_experiment() and set_seed(), which are used by the run_*.py
scripts that produce all results reported in the paper.

Running this file directly executes the original single-seed pipeline:
  Phase 1: Clean FL baseline (5 seeds, no attack)
  Phase 2: Per-attack runs   (FedAvg under attack + HRA defense)
  Phase 3: Cross-attack summary

Usage:
  python main.py                          # baseline + both attacks
  python main.py --attack label_flipping  # single attack only
  python main.py --skip-baseline          # reuse existing baseline JSON
"""

import argparse
import json
import os
import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from attack_config import ALL_ATTACK_TYPES, ATTACK_LABELS, AttackConfig
from data_loader import FLDataset
from evaluation import (
    evaluate_with_threshold,
    get_probabilities,
    sweep_thresholds,
)
from defense_config import DefenseConfig
from federated_client import FederatedClient
from federated_server import FederatedServer
from model import CNNLSTMModel, MLPModel
from results_paths import (
    BASELINE_DIR,
    BASELINE_JSON,
    BASELINE_SEED42_JSON,
    SUMMARY_DIR,
    SUMMARY_JSON,
    SUMMARY_PNG,
    ensure_dirs,
    get_results_paths,
    load_baseline_means,
    metrics_to_json,
    migrate_legacy_results,
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.titlesize': 15,
})

BASELINE_SEEDS = [42, 123, 7, 99, 500]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_experiment(
    num_clients=10,
    num_rounds=20,
    malicious_ratio=0.0,
    attack_type=None,
    defense=False,
    iid=False,
    model_type='mlp',
    batch_size=16,
    local_epochs=5,
    lr=0.01,
    loss_type='ce',
    patience=20,
    seed=42,
    dataset=None,
    attack_config=None,
    model_save_path=None,
    defense_config=None,
):
    """Run one federated learning experiment."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = attack_type or "clean"
    print(
        f"\nRunning: model={model_type} | attack={tag} | clients={num_clients} | "
        f"rounds={num_rounds} | mal={malicious_ratio} | defense={defense} | seed={seed}"
    )

    if dataset is None:
        dataset = FLDataset(
            "data/KDDTrain+.txt", "data/KDDTest+.txt",
            model_type=model_type, seed=seed,
        )

    client_data = dataset.split_data(num_clients, iid=iid)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    if model_type == 'mlp':
        model_class = MLPModel
        model_kwargs = {'input_dim': dataset.input_dim}
    else:
        model_class = CNNLSTMModel
        model_kwargs = {'input_dim': dataset.input_dim}

    if defense_config is not None:
        defense_cfg = defense_config
    else:
        defense_cfg = DefenseConfig.for_attack(attack_type) if defense else DefenseConfig()
    
    server = FederatedServer(
        model_class, device, num_clients,
        defense_config=defense_cfg,
        **model_kwargs,
    )
    if defense and attack_type == 'model_poisoning':
        print(
            f"  [Defense] model_poisoning mode: instant_rep={defense_cfg.use_instant_rep}, "
            f"exclude<{defense_cfg.hard_exclude_threshold}, "
            f"checkpoint_from_round={defense_cfg.min_checkpoint_round}"
        )

    clients = []
    num_malicious = int(num_clients * malicious_ratio)
    for i in range(num_clients):
        is_malicious = i < num_malicious
        client_attack = attack_type if is_malicious else None
        cfg = AttackConfig.for_attack(attack_type) if is_malicious and attack_type else None
        if attack_config is not None and is_malicious:
            cfg = attack_config
        clients.append(FederatedClient(
            i, client_data[i], device,
            model_type=model_type,
            is_malicious=is_malicious,
            attack_type=client_attack,
            batch_size=batch_size,
            loss_type=loss_type,
            global_class_weights=getattr(dataset, 'global_class_weights', None),
            attack_config=cfg,
        ))

    history = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'loss': [],
        'f1_binary': [], 'recall_binary': [], 'precision_binary': [],
        'f1_macro': [], 'roc_auc': [],
    }
    reputation_history = []

    pbar = tqdm(range(num_rounds), desc=f"FL [{model_type.upper()}] {tag} seed={seed}")
    for r in pbar:
        client_weights = []
        client_sizes = []

        for client in clients:
            local_weights = client.train_local_model(
                server.get_global_model(), epochs=local_epochs, lr=lr
            )
            client_weights.append(local_weights)
            client_sizes.append(client.get_update_size())

        if defense:
            rep_info = server.aggregate_hybrid_robust(client_weights, client_sizes)
            reputation_history.append({
                'round': r + 1,
                'round_reputations': rep_info['round_reputations'],
                'ema_reputations': rep_info['ema_reputations'],
            })
        else:
            server.aggregate_fedavg(client_weights, client_sizes)

        val_metrics = server.evaluate(val_loader)

        for key in history:
            if key in val_metrics:
                history[key].append(val_metrics[key])

        is_best = server.checkpoint_if_best(val_metrics, r + 1)
        best_marker = " *" if is_best else ""
        pbar.set_postfix({
            'Acc': f"{val_metrics['accuracy']:.1f}%",
            'F1b': f"{val_metrics['f1_binary']:.1f}%{best_marker}",
            'AUC': f"{val_metrics['roc_auc']:.1f}%",
        })

        if server.should_stop_early(patience):
            print(f"\n  >> Early stopping at round {r + 1} (patience={patience})")
            break

    best_val_metrics, best_round = server.restore_best_model()

    val_targets, val_probs = get_probabilities(server.get_global_model(), val_loader, device)
    _, best_threshold, _ = sweep_thresholds(val_targets, val_probs)

    test_targets, test_probs = get_probabilities(server.get_global_model(), test_loader, device)
    test_metrics = evaluate_with_threshold(test_targets, test_probs, threshold=best_threshold)

    test_metrics['val_targets'] = val_targets
    test_metrics['val_probs'] = val_probs
    test_metrics['test_targets'] = test_targets
    test_metrics['test_probs'] = test_probs
    test_metrics['best_threshold'] = best_threshold
    test_metrics['loss'] = val_metrics['loss']
    test_metrics['reputation_history'] = reputation_history

    if best_val_metrics:
        print(
            f"\n  >> Best model Round {best_round} "
            f"(val AUC={best_val_metrics['roc_auc']:.2f}% | thresh={best_threshold:.2f})"
        )
        print(
            f"  >> TEST: Acc={test_metrics['accuracy']:.2f}% | "
            f"F1b={test_metrics['f1_binary']:.2f}% | "
            f"AUC={test_metrics['roc_auc']:.2f}% | "
            f"Recall={test_metrics['recall_binary']:.2f}%"
        )
        cm = test_metrics['confusion_matrix']
        print(
            f"  >> Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} "
            f"FN={cm[1][0]} TP={cm[1][1]}"
        )
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
            server.save_best_model(model_save_path)

    return history, server, test_metrics


def run_baseline_phase(dataset, seeds, num_clients, num_rounds, local_epochs, lr,
                       batch_size, patience, out_dir):
    """5-seed CV baseline without feature selection."""
    print(f"\n{'=' * 80}\n  PHASE 1: SECURE BASELINE (no feature selection)\n{'=' * 80}")
    os.makedirs(out_dir, exist_ok=True)
    baseline_results = {}

    for seed in seeds:
        print(f"\n--- Baseline Seed {seed} ---")
        dataset.reshuffle(seed)
        history, _, test_metrics = run_experiment(
            num_clients=num_clients,
            num_rounds=num_rounds,
            malicious_ratio=0.0,
            model_type='mlp',
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            seed=seed,
            dataset=dataset,
            loss_type='ce',
            patience=patience,
        )
        baseline_results[seed] = (history, test_metrics)
        print_summary_table(history, test_metrics, num_clients, num_rounds)

    baseline_json = os.path.join(out_dir, "secure_baseline_no_fs.json")
    with open(baseline_json, "w") as f:
        json.dump({
            str(k): metrics_to_json(v[1]) for k, v in baseline_results.items()
        }, f, indent=4)
    print(f"  [Saved] {baseline_json}")

    best_seed = max(baseline_results, key=lambda s: baseline_results[s][1]['f1_binary'])
    plot_full_dashboard(
        baseline_results[best_seed][0],
        f'Secure Baseline no-FS (seed={best_seed})',
        os.path.join(out_dir, f"dashboard_seed{best_seed}.png"),
    )
    return baseline_results, baseline_json


def run_matched_baseline_seed42(dataset, num_clients, num_rounds, local_epochs, lr,
                                batch_size, patience, seed, out_path):
    """Single-seed baseline with same patience as attack runs."""
    print(f"\n--- Matched baseline seed={seed} (patience={patience}) ---")
    dataset.reshuffle(seed)
    history, _, test_metrics = run_experiment(
        num_clients=num_clients,
        num_rounds=num_rounds,
        malicious_ratio=0.0,
        model_type='mlp',
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
        seed=seed,
        dataset=dataset,
        loss_type='ce',
        patience=patience,
    )
    with open(out_path, "w") as f:
        json.dump(metrics_to_json(test_metrics, {"seed": seed}), f, indent=4)
    print(f"  [Saved] {out_path}")
    return test_metrics


def verify_defense_recovery(baseline_f1, attack_f1, defense_f1, attack_type):
    """Print defense recovery checks."""
    ok_near_baseline = defense_f1 >= baseline_f1 - 2.0
    ok_beats_attack = defense_f1 >= attack_f1 + 30.0
    status_b = "PASS" if ok_near_baseline else "WARN"
    status_a = "PASS" if ok_beats_attack else "WARN"
    print(
        f"\n  [Defense check {attack_type}] "
        f"vs baseline: {status_b} ({defense_f1:.1f}% vs {baseline_f1:.1f}%) | "
        f"vs attack: {status_a} (+{defense_f1 - attack_f1:.1f} pp)"
    )
    return ok_near_baseline and ok_beats_attack


def run_attack_phase(
    attack_type,
    dataset,
    num_clients,
    num_rounds,
    local_epochs,
    lr,
    batch_size,
    malicious_ratio,
    seed,
    patience_attack,
    baseline_json,
):
    """Run attack (no defense) + attack (with defense) for one attack type."""
    label = ATTACK_LABELS.get(attack_type, attack_type)
    paths = get_results_paths(attack_type)
    os.makedirs(paths["dir"], exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"  ATTACK PHASE: {label} ({int(malicious_ratio * 100)}% malicious)")
    print(f"{'=' * 80}")

    dataset.reshuffle(seed)

    print(f"\n  >> Scenario A: {label} — FedAvg (no defense)")
    attack_history, _, attack_metrics = run_experiment(
        num_clients=num_clients,
        num_rounds=num_rounds,
        malicious_ratio=malicious_ratio,
        attack_type=attack_type,
        defense=False,
        model_type='mlp',
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
        seed=seed,
        dataset=dataset,
        loss_type='ce',
        patience=patience_attack,
    )
    print_summary_table(attack_history, attack_metrics, num_clients, num_rounds)
    plot_full_dashboard(
        attack_history,
        f'{label} — No Defense ({int(malicious_ratio * 100)}% malicious)',
        paths["attack_no_defense_dashboard"],
    )
    attack_payload = metrics_to_json(attack_metrics, {
        'malicious_ratio': malicious_ratio,
        'attack_type': attack_type,
        'defense': False,
    })
    with open(paths["attack_no_defense_json"], "w") as f:
        json.dump(attack_payload, f, indent=4)
    print(f"  [Saved] {paths['attack_no_defense_json']}")

    dataset.reshuffle(seed)

    print(f"\n  >> Scenario B: {label} — Hybrid Robust Defense")
    defense_history, _, defense_metrics = run_experiment(
        num_clients=num_clients,
        num_rounds=num_rounds,
        malicious_ratio=malicious_ratio,
        attack_type=attack_type,
        defense=True,
        model_type='mlp',
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
        seed=seed,
        dataset=dataset,
        loss_type='ce',
        patience=patience_attack,
    )
    print_summary_table(defense_history, defense_metrics, num_clients, num_rounds)
    plot_full_dashboard(
        defense_history,
        f'{label} — With Defense ({int(malicious_ratio * 100)}% malicious)',
        paths["attack_with_defense_dashboard"],
    )
    defense_payload = metrics_to_json(defense_metrics, {
        'malicious_ratio': malicious_ratio,
        'attack_type': attack_type,
        'defense': 'hybrid_robust_aggregation',
    })
    with open(paths["attack_with_defense_json"], "w") as f:
        json.dump(defense_payload, f, indent=4)
    print(f"  [Saved] {paths['attack_with_defense_json']}")

    rep_hist = defense_metrics.get('reputation_history', [])
    if rep_hist:
        with open(paths["reputation_trace"], "w") as f:
            json.dump({
                'attack_type': attack_type,
                'malicious_client_ids': list(range(int(num_clients * malicious_ratio))),
                'rounds': rep_hist,
            }, f, indent=2)
        print(f"  [Saved] {paths['reputation_trace']}")

    plot_attack_comparison(
        attack_metrics, baseline_json, paths["comparison_baseline_vs_attack"],
        attack_label=f"{label} ({int(malicious_ratio * 100)}%)",
    )
    plot_three_way_comparison(
        baseline_json, attack_metrics, defense_metrics,
        paths["comparison_three_way"],
        attack_label=label,
    )

    baseline_means = load_baseline_means(baseline_json)
    baseline_f1 = baseline_means.get('f1_binary_mean', 0)
    verify_defense_recovery(
        baseline_f1,
        attack_metrics['f1_binary'],
        defense_metrics['f1_binary'],
        attack_type,
    )

    return {
        'attack_type': attack_type,
        'attack': attack_payload,
        'defense': defense_payload,
        'recovery_delta_f1': float(defense_metrics['f1_binary'] - attack_metrics['f1_binary']),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_full_dashboard(history, title_suffix, filename):
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f'Federated Learning Dashboard — {title_suffix}',
                 fontsize=16, fontweight='bold', y=0.98)

    metric_names = ['accuracy', 'precision_binary', 'recall_binary', 'f1_binary']
    metric_labels = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
    colors = ['#1976D2', '#388E3C', '#F57C00', '#C2185B']
    markers = ['o', 's', '^', 'D']

    for idx, (metric, label, color) in enumerate(zip(metric_names, metric_labels, colors)):
        ax = fig.add_subplot(gs[0, idx])
        values = history[metric]
        rounds = range(1, len(values) + 1)
        ax.plot(rounds, values, color=color, linewidth=2, marker='o', markersize=2.5)
        ax.fill_between(rounds, values, alpha=0.1, color=color)
        best_val = max(values)
        best_round = values.index(best_val) + 1
        ax.scatter([best_round], [best_val], color=color, s=60, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.set_title(f'{label}\nFinal: {values[-1]:.2f}% | Best: {best_val:.2f}%',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Round', fontsize=9)
        ax.grid(True, alpha=0.2)

    ax_overlay = fig.add_subplot(gs[1, :2])
    for metric, label, color, marker in zip(metric_names, metric_labels, colors, markers):
        values = history[metric]
        rounds = range(1, len(values) + 1)
        ax_overlay.plot(rounds, values, color=color, linewidth=2, marker=marker,
                        markersize=3, alpha=0.85,
                        label=f'{label.split()[0]} ({values[-1]:.2f}%)')
    if history.get('roc_auc'):
        ax_overlay.plot(range(1, len(history['roc_auc']) + 1), history['roc_auc'],
                        color='#9C27B0', linewidth=2, marker='x', markersize=3,
                        alpha=0.85, linestyle='--',
                        label=f'ROC-AUC ({history["roc_auc"][-1]:.2f}%)')
    ax_overlay.set_xlabel('Communication Round', fontweight='bold')
    ax_overlay.set_ylabel('Score (%)', fontweight='bold')
    ax_overlay.set_title('All Metrics Convergence (Val Set)', fontweight='bold')
    ax_overlay.legend(loc='lower right', fontsize=9)
    ax_overlay.grid(True, alpha=0.2)

    ax_loss = fig.add_subplot(gs[1, 2:])
    loss_vals = history['loss']
    rounds = range(1, len(loss_vals) + 1)
    ax_loss.plot(rounds, loss_vals, color='#7B1FA2', linewidth=2, marker='o', markersize=2.5)
    ax_loss.fill_between(rounds, loss_vals, alpha=0.1, color='#7B1FA2')
    min_loss = min(loss_vals)
    ax_loss.set_title(f'Validation Loss (Min: {min_loss:.4f})', fontweight='bold')
    ax_loss.grid(True, alpha=0.2)

    ax_bar = fig.add_subplot(gs[2, :2])
    all_metrics = ['accuracy', 'f1_binary', 'recall_binary', 'precision_binary', 'roc_auc']
    bar_labels = ['Accuracy', 'F1-Score', 'Recall', 'Precision', 'ROC-AUC']
    bar_colors = ['#1976D2', '#C2185B', '#F57C00', '#388E3C', '#9C27B0']
    final_values = [history[m][-1] for m in all_metrics]
    ax_bar.bar(bar_labels, final_values, color=bar_colors, edgecolor='white', linewidth=2)
    ax_bar.set_ylabel('Score (%)', fontweight='bold')
    ax_bar.set_title('Final Metric Values (Val Set)', fontweight='bold')
    ax_bar.set_ylim(0, 105)
    ax_bar.grid(True, alpha=0.15, axis='y')

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filename}")


def print_summary_table(history, test_metrics, num_clients, num_rounds):
    print("\n" + "=" * 80)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Clients: {num_clients} | Rounds: {num_rounds}")
    print("-" * 80)
    print(f"  {'Metric':<16} {'Val Final':>10} {'Val Best':>10} {'Best Rnd':>10} {'TEST':>10}")
    print("-" * 80)
    for label, key in [
        ('ACCURACY', 'accuracy'), ('F1-BINARY', 'f1_binary'),
        ('RECALL-ATK', 'recall_binary'), ('PREC-ATK', 'precision_binary'),
        ('ROC-AUC', 'roc_auc'),
    ]:
        if key in history and history[key]:
            values = history[key]
            print(
                f"  {label:<16} {values[-1]:>9.2f}% {max(values):>9.2f}% "
                f"{values.index(max(values)) + 1:>10} {test_metrics.get(key, 0):>9.2f}%"
            )
    cm = test_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    print("-" * 80)
    print(f"  TEST CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print("=" * 80)


def _baseline_means_from_json(baseline_json):
    with open(baseline_json) as f:
        data = json.load(f)
    keys = ['accuracy', 'f1_binary', 'recall_binary', 'precision_binary']
    return {k: float(np.mean([v[k] for v in data.values() if k in v])) for k in keys}


def plot_attack_comparison(attack_test_metrics, baseline_json_path, output_filename,
                           attack_label="Under Attack"):
    if not os.path.exists(baseline_json_path):
        print(f"  [Warning] Baseline not found: {baseline_json_path}")
        return
    b = _baseline_means_from_json(baseline_json_path)
    labels = ['Accuracy', 'F1-Binary', 'Recall', 'Precision']
    keys = ['accuracy', 'f1_binary', 'recall_binary', 'precision_binary']
    baseline_vals = [b[k] for k in keys]
    attack_vals = [attack_test_metrics.get(k, 0) for k in keys]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals, width, label='Baseline (Secure, no FS)', color='#1976D2')
    ax.bar(x + width / 2, attack_vals, width, label=attack_label, color='#D32F2F')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Baseline vs Attack', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend()
    for rects in ax.containers:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {output_filename}")


def plot_three_way_comparison(baseline_json, attack_metrics, defense_metrics,
                              output_filename, attack_label="Attack"):
    if not os.path.exists(baseline_json):
        return
    b = _baseline_means_from_json(baseline_json)
    labels = ['Accuracy', 'F1-Binary', 'Recall', 'Precision']
    keys = ['accuracy', 'f1_binary', 'recall_binary', 'precision_binary']
    baseline_vals = [b[k] for k in keys]
    attack_vals = [attack_metrics.get(k, 0) for k in keys]
    defense_vals = [defense_metrics.get(k, 0) for k in keys]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, baseline_vals, width, label='Secure Baseline (no FS)', color='#1976D2')
    ax.bar(x, attack_vals, width, label=f'{attack_label} (No Defense)', color='#D32F2F')
    ax.bar(x + width, defense_vals, width, label=f'{attack_label} (Defense)', color='#388E3C')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title(f'Baseline vs {attack_label} vs Defense', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    for rects in ax.containers:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {output_filename}")


def plot_all_attacks_summary(baseline_json, attack_summaries, output_path):
    """Grouped bar: F1-binary for baseline + each attack/defense pair."""
    if not os.path.exists(baseline_json):
        return
    b = _baseline_means_from_json(baseline_json)
    scenarios = ['Baseline']
    f1_vals = [b['f1_binary']]
    colors = ['#1976D2']
    for entry in attack_summaries:
        atk = ATTACK_LABELS.get(entry['attack_type'], entry['attack_type'])
        scenarios.append(f"{atk}\n(no def)")
        f1_vals.append(entry['attack']['f1_binary'])
        colors.append('#D32F2F')
        scenarios.append(f"{atk}\n(defense)")
        f1_vals.append(entry['defense']['f1_binary'])
        colors.append('#388E3C')

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(scenarios))
    bars = ax.bar(x, f1_vals, color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('F1-Binary (Attack Class) %', fontweight='bold')
    ax.set_title('Cross-Attack Comparison (Test F1-Binary)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylim(0, 105)
    ax.axhline(y=b['f1_binary'], color='gray', linestyle='--', alpha=0.5,
               label=f"Baseline mean {b['f1_binary']:.1f}%")
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.1f}%',
                ha='center', fontweight='bold', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.15, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="FL-IDS multi-attack experiment pipeline")
    p.add_argument('--attack', default='all',
                   choices=['all'] + ALL_ATTACK_TYPES,
                   help='Which attack to run (default: all)')
    p.add_argument('--malicious-ratio', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=42, help='Seed for attack/defense splits')
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--patience-baseline', type=int, default=7)
    p.add_argument('--patience-attack', type=int, default=50)
    p.add_argument('--skip-baseline', action='store_true',
                   help='Reuse existing baseline JSON')
    p.add_argument('--baseline-json', default=BASELINE_JSON)
    p.add_argument('--num-clients', type=int, default=10)
    p.add_argument('--local-epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--run-matched-baseline', action='store_true',
                   help='Also run single-seed baseline with attack patience')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    migrate_legacy_results()

    print("=" * 80)
    print("  FL-IDS MULTI-ATTACK PIPELINE")
    print("  Baseline (no FS) | Label Flipping | Model Poisoning | Defense")
    print("=" * 80)

    dataset = FLDataset(
        "data/cic", None, dataset_type='cicids2017',
        model_type='mlp', val_ratio=0.1, seed=42, subsample_size=500000,
    )

    baseline_json = args.baseline_json
    if not args.skip_baseline or not os.path.exists(baseline_json):
        _, baseline_json = run_baseline_phase(
            dataset=dataset,
            seeds=BASELINE_SEEDS,
            num_clients=args.num_clients,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience_baseline,
            out_dir=BASELINE_DIR,
        )
    else:
        print(f"\n  [Skipped baseline] Using {baseline_json}")

    if args.run_matched_baseline:
        run_matched_baseline_seed42(
            dataset, args.num_clients, args.rounds, args.local_epochs,
            args.lr, args.batch_size, args.patience_attack, args.seed,
            BASELINE_SEED42_JSON,
        )

    attacks_to_run = ALL_ATTACK_TYPES if args.attack == 'all' else [args.attack]
    attack_summaries = []

    for attack_type in attacks_to_run:
        summary = run_attack_phase(
            attack_type=attack_type,
            dataset=dataset,
            num_clients=args.num_clients,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            malicious_ratio=args.malicious_ratio,
            seed=args.seed,
            patience_attack=args.patience_attack,
            baseline_json=baseline_json,
        )
        attack_summaries.append(summary)

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    baseline_block = load_baseline_means(baseline_json)
    full_summary = {
        'baseline': baseline_block,
        'attacks': {
            s['attack_type']: {
                'attack': s['attack'],
                'defense': s['defense'],
                'recovery_delta_f1': s['recovery_delta_f1'],
            } for s in attack_summaries
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(full_summary, f, indent=4)
    print(f"\n  [Saved] {SUMMARY_JSON}")

    plot_all_attacks_summary(baseline_json, attack_summaries, SUMMARY_PNG)

    print(f"\n{'=' * 80}\n  FINAL SUMMARY\n{'=' * 80}")
    print(f"  {'Scenario':<40} {'F1-Binary':>10} {'ROC-AUC':>10}")
    print(f"  {'-' * 62}")
    if baseline_block:
        print(f"  {'Baseline (no FS, mean)':<40} "
              f"{baseline_block.get('f1_binary_mean', 0):>9.2f}%")
    for s in attack_summaries:
        label = ATTACK_LABELS.get(s['attack_type'], s['attack_type'])
        print(f"  {label + ' (attack)':<40} {s['attack']['f1_binary']:>9.2f}% "
              f"{s['attack'].get('roc_auc', 0):>9.2f}%")
        print(f"  {label + ' (defense)':<40} {s['defense']['f1_binary']:>9.2f}% "
              f"{s['defense'].get('roc_auc', 0):>9.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
