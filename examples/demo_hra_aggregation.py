"""
Minimal HRA demo on synthetic client updates (no dataset needed, runs in seconds).

Ten clients send updates of the paper's MLP to the server for 15 rounds.
Clients 0-3 are malicious: they send a scaled, noisy update (model poisoning,
s = 5, sigma = 5, as in the paper); clients 4-9 send small honest updates.
The script prints the EMA reputation of every client per round and which
clients HRA excludes.

Run from the repository root:
    python3 examples/demo_hra_aggregation.py
"""

import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from defense_config import DefenseConfig      # noqa: E402
from federated_server import FederatedServer  # noqa: E402
from model import MLPModel                    # noqa: E402

K, MALICIOUS, ROUNDS, INPUT_DIM = 10, {0, 1, 2, 3}, 15, 47
SCALE, NOISE = 5.0, 5.0

torch.manual_seed(42)
np.random.seed(42)
server = FederatedServer(MLPModel, torch.device("cpu"), K,
                         defense_config=DefenseConfig.unified(), input_dim=INPUT_DIM)

print("round | " + " ".join(f"  C{i}{'*' if i in MALICIOUS else ' '}" for i in range(K)) + " | excluded")
for r in range(1, ROUNDS + 1):
    g = server.get_global_model().state_dict()
    updates = []
    for i in range(K):
        sd = copy.deepcopy(g)
        for k, v in sd.items():
            if not v.is_floating_point():
                continue
            delta = 0.01 * torch.randn_like(v)            # honest local change
            if i in MALICIOUS:                             # model poisoning
                delta = -SCALE * delta + NOISE * 0.01 * torch.randn_like(v)
            sd[k] = v + delta
        updates.append(sd)
    info = server.aggregate_hybrid_robust(updates, [1000] * K)
    rep = np.array(info["ema_reputations"])
    excl = [i for i in range(K) if rep[i] < server.defense_config.hard_exclude_threshold]
    print(f"{r:5d} | " + " ".join(f"{x:5.2f}" for x in rep) + f" | {excl}")

print("\n* = malicious client. Reputation starts at 1.0; clients below tau = 0.05 are excluded.")
