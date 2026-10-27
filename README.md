# HRA: Hybrid Reputation-weighted Aggregation for Federated Intrusion Detection

Code, experiment scripts and per-seed results for the paper:

> S. S. Alhawas and M. A. Rassam, "Robust federated learning for intrusion detection in edge computing under adversarial attacks: a hybrid reputation-weighted aggregation approach" (under review).

HRA is a server-side aggregation rule for federated learning (FL). In every round it
1. scores each client update by its distance to the coordinate-wise median, r_k = 1 / (1 + d_k), min-max normalized;
2. accumulates the score into an exponential moving average (EMA) reputation, R_k(t) = (1 − α) R_k(t−1) + α r̂_k, starting at R_k(0) = 1;
3. excludes clients whose reputation falls below τ and weights the others by reputation;
4. falls back to the coordinate-wise median if fewer than K_min clients remain active.

The FL simulation is implemented directly in PyTorch, without an FL framework.

## Repository contents

| File(s) | Purpose |
|---|---|
| `federated_server.py` | HRA (`aggregate_hybrid_robust`) and the baselines: FedAvg, Krum, Multi-Krum, Trimmed Mean, coordinate-wise median, FoolsGold; validation-based checkpointing |
| `federated_client.py` | Local training; label-flipping and model-poisoning attacks |
| `defense_config.py`, `attack_config.py` | HRA configuration (`DefenseConfig.unified()`) and attack settings |
| `data_loader.py`, `feature_selection.py` | CIC-IDS-2017 preprocessing, deduplication, feature selection, Dirichlet Non-IID partitioning |
| `model.py`, `losses.py`, `evaluation.py` | MLP model, losses, threshold selection on validation data and metrics |
| `main.py`, `results_paths.py` | Core experiment loop |
| `centralized_baseline.py` | Centralized (non-federated) MLP upper bound |
| `validate_attacks.py` | Sanity check that both attacks degrade FedAvg |
| `run_*.py` | Experiment runners for the paper (see below) |
| `analysis/` | Statistical tests and figure scripts |
| `examples/demo_hra_aggregation.py` | HRA on synthetic updates; runs in seconds without the dataset |
| `results/` | Per-seed result files (JSON) of every experiment |

## Experiments in the paper

| Script | Paper |
|---|---|
| `run_defense_comparison_seeds.py --attack model_poisoning` | Tables 7, 9, 10; Fig. 10 (10 seeds) |
| `run_defense_comparison_seeds.py --attack label_flipping` | Table 7 (label flipping, no defense); Table 13 default row |
| `run_unified_defense.py` | Table 12 (fixed vs attack-specific configuration, both attacks, 10 seeds) |
| `run_remaining.py --part baseline / ablation / hparam / adaptive / overhead` | Tables 6, 11, 14, 15, 16 |
| `run_attack_sensitivity.py --part ratio / scale / noise [--attack label_flipping]` | Table 13, Fig. 11 |
| `run_final_checks.py --part kmin_warmup / nofallback / coldstart / scalability` | Tables 11, 14, 15; Section 6.7 |
| `run_dropout.py` | Section 5.7.10 (client dropout) |
| `run_reputation_seed42.py` | Figs. 8–9 |
| `analysis/statistical_tests.py` | Paired Wilcoxon tests and 95% confidence intervals |
| `analysis/make_*.py` | Figures 5, 6, 8, 9, 11 and the defense-comparison and ablation figures |

Every runner saves after each run and skips finished runs, so an interrupted command can be re-run to continue. Add `--rounds 2 --seeds 42` for a quick test.

## Configuration used in the paper

A single configuration is used for all attacks (`DefenseConfig.unified()`):

| Parameter | Value |
|---|---|
| EMA coefficient α | 0.35 |
| Exclusion threshold τ | 0.05 |
| Minimum active clients K_min | 4 |
| Median fallback | on |
| Checkpoint warm-up | 8 rounds |

Setting: CIC-IDS-2017 (500,000 flows, 47 features), K = 10 clients, Dirichlet γ = 0.5 Non-IID partition, 40% malicious clients (clients 0–3), 50 rounds, 5 local epochs, learning rate 0.01, batch size 64. Main experiments use seeds {42, 123, 7, 99, 500, 2024, 31, 77, 1001, 256}; sensitivity, ablation and adaptive-attack experiments use {42, 123, 7}.

## Quick start

```bash
pip install -r requirements.txt
python3 examples/demo_hra_aggregation.py      # HRA on synthetic updates (seconds)
python3 analysis/statistical_tests.py         # p-values and confidence intervals from results/
```

To re-run the experiments, download CIC-IDS-2017 (MachineLearningCVE CSV files) from the Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/ids-2017.html, place the CSV files in `data/cic/`, and run the scripts above from the repository root. The dataset is not redistributed here.

## Environment

Apple M4 (CPU only, 16 GB), macOS 15.6.1, Python 3.14.0, PyTorch 2.12.0, scikit-learn 1.9.0, NumPy 2.4.6.

## Citation

If you use this code or results, please cite the paper above (full reference to be added on publication).

## License

Copyright (c) 2026 Sahar Saleh Alhawas and Murad A. Rassam. All rights reserved. The code is provided for review and reproducibility of the paper's results; contact the authors for any other use.
