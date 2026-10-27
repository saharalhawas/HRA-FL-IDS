# Result files

One JSON file per experiment; each entry is one run (`<configuration>|<seed>`) with
F1, precision, recall, accuracy, ROC-AUC, PR-AUC, MCC, the confusion matrix, the
decision threshold and the best round. HRA runs also list the excluded clients per round.

| Folder / file | Paper |
|---|---|
| `remaining/baseline.json` | Table 6, Table 7 (secure baseline) |
| `defense_comparison_seeds/model_poisoning.json` | Tables 7, 9, 10; Figs. 5, 6, 10 |
| `defense_comparison_seeds/label_flipping.json` | Table 7, Table 13 (label flipping); Figs. 5, 6 |
| `unified_defense/unified_defense_results.json` | Table 12; label-flipping rows of Tables 7–8 |
| `statistical_tests.json` | Wilcoxon p-values and 95% CIs (Tables 10, 12) |
| `remaining/ablation.json`, `final_checks/nofallback.json` | Table 11 |
| `attack_sensitivity/*.json` | Table 13, Fig. 11 |
| `remaining/hparam.json`, `final_checks/kmin_warmup.json` | Table 14 |
| `remaining/adaptive.json`, `final_checks/coldstart.json` | Table 15 |
| `remaining/overhead.json`, `final_checks/scalability.json` | Table 16, Section 6.7 |
| `final_checks/reputation_seed42.json` | Figs. 8–9 |
| `final_checks/dropout.json` | Section 5.7.10 |
| `attack_sensitivity/lf_*.json` | Table 13 and Fig. 11 (label flipping) |

Note: for HRA configurations without hard exclusion (ablation configuration D and `tau=None` in
`remaining/hparam.json`), `excluded_per_round` lists the clients whose reputation fell below 0.05,
i.e. clients that would have been excluded; they are not removed from aggregation in these runs.
