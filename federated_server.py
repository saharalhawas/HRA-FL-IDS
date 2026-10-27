import torch
import torch.nn as nn
import numpy as np
import copy
from collections import OrderedDict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from defense_config import DefenseConfig

class FederatedServer:
    def __init__(self, model_class, device, num_clients, defense_config=None, **model_kwargs):
        self.device = device
        self.global_model = model_class(**model_kwargs).to(device)
        self.num_clients = num_clients
        self.reputations = np.ones(num_clients)
        self.defense_config = defense_config or DefenseConfig()
        self.alpha = self.defense_config.alpha
        
        # Best model checkpointing (uses ROC-AUC for threshold-independent checkpointing)
        self.best_auc = 0.0
        self.best_metrics = None
        self.best_round = 0
        self.best_model_state = None
        self.rounds_without_improvement = 0

    def aggregate_fedavg(self, client_weights, client_sizes):
        """Standard FedAvg aggregation."""
        total_size = sum(client_sizes)
        agg_weights = OrderedDict()
        
        for i, weight in enumerate(client_weights):
            proportion = client_sizes[i] / total_size
            for key in weight.keys():
                if i == 0:
                    agg_weights[key] = weight[key] * proportion
                else:
                    agg_weights[key] += weight[key] * proportion
        
        self.global_model.load_state_dict(agg_weights)

    def aggregate_krum(self, client_weights, client_sizes):
        """Krum aggregation (Blanchard et al., 2017).

        Selects the single client update whose sum of distances to its
        closest (K - f - 2) neighbours is minimal, where f is the
        estimated number of Byzantine clients.
        """
        n = len(client_weights)
        f = max(1, int(n * 0.4))           # assume up to 40 % malicious
        n_closest = max(1, n - f - 2)
        keys = [k for k in client_weights[0] if client_weights[0][k].is_floating_point()]

        # Flatten each client's update into a single vector for distance calc
        flat = []
        for w in client_weights:
            flat.append(torch.cat([w[k].flatten().float() for k in keys]))

        # Pairwise L2 distance matrix
        dists = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(flat[i] - flat[j]).item()
                dists[i, j] = d
                dists[j, i] = d

        # Krum score = sum of distances to n_closest nearest neighbours
        scores = []
        for i in range(n):
            sorted_dists = torch.sort(dists[i])[0]
            # skip self (0) → take next n_closest
            scores.append(sorted_dists[1:1 + n_closest].sum().item())

        best = int(np.argmin(scores))
        self.global_model.load_state_dict(client_weights[best])

    def aggregate_trimmed_mean(self, client_weights, client_sizes, trim_ratio=0.4):
        """Coordinate-wise Trimmed Mean (Yin et al., 2018).

        For each parameter coordinate, sorts the values from all clients,
        trims the top and bottom *trim_ratio* fraction, and averages the
        remaining middle values.
        """
        keys = list(client_weights[0].keys())
        float_keys = [k for k in keys if client_weights[0][k].is_floating_point()]
        int_keys = [k for k in keys if k not in float_keys]
        n = len(client_weights)
        trim_count = max(1, int(n * trim_ratio))

        agg = OrderedDict()
        for key in float_keys:
            stacked = torch.stack([w[key].float() for w in client_weights], dim=0)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_count:n - trim_count]
            if trimmed.shape[0] == 0:
                # Fallback: if trim removes everything, use median
                agg[key] = torch.median(stacked, dim=0)[0]
            else:
                agg[key] = trimmed.mean(dim=0)
            agg[key] = agg[key].to(client_weights[0][key].dtype)

        # Integer keys (e.g. BatchNorm num_batches_tracked) from first client
        for key in int_keys:
            agg[key] = client_weights[0][key]

        self.global_model.load_state_dict(agg)

    def aggregate_coordinate_median(self, client_weights, client_sizes):
        """Pure coordinate-wise median aggregation.

        For each parameter coordinate, takes the median value across all
        clients.  Used as ablation baseline (Stage 1 of HRA without
        reputation weighting).
        """
        keys = list(client_weights[0].keys())
        float_keys = [k for k in keys if client_weights[0][k].is_floating_point()]
        int_keys = [k for k in keys if k not in float_keys]

        agg = OrderedDict()
        for key in float_keys:
            stacked = torch.stack([w[key] for w in client_weights], dim=0)
            agg[key] = torch.median(stacked, dim=0)[0]

        for key in int_keys:
            agg[key] = client_weights[0][key]

        self.global_model.load_state_dict(agg)

    def aggregate_hybrid_robust(self, client_weights, client_sizes):
        """Hybrid Robust Aggregation: Statistical Filter + Reputation.

        Returns:
            dict with round_reputations (normalized) and ema_reputations arrays.
        """
        keys = list(client_weights[0].keys())
        float_keys = [k for k in keys if client_weights[0][k].is_floating_point()]
        int_keys = [k for k in keys if k not in float_keys]

        median_weights = OrderedDict()
        for key in float_keys:
            stacked = torch.stack([w[key] for w in client_weights], dim=0)
            median_weights[key] = torch.median(stacked, dim=0)[0]

        round_reputations = []
        for i, weight in enumerate(client_weights):
            dist = 0.0
            for key in float_keys:
                dist += torch.norm(weight[key] - median_weights[key]).item()
            score = 1.0 / (1.0 + dist)
            round_reputations.append(score)

        round_reputations = np.array(round_reputations)
        round_reputations = (
            (round_reputations - round_reputations.min())
            / (round_reputations.max() - round_reputations.min() + 1e-9)
        )
        self.reputations = (
            (1 - self.alpha) * self.reputations + self.alpha * round_reputations
        )

        cfg = self.defense_config
        agg_reps = round_reputations.copy() if cfg.use_instant_rep else self.reputations.copy()

        if cfg.hard_exclude_threshold is not None:
            agg_reps[agg_reps < cfg.hard_exclude_threshold] = 0.0

        active = int(np.sum(agg_reps > 0))
        use_median = (
            cfg.use_median_fallback
            and active < cfg.min_active_clients
        )

        final_weights = OrderedDict()
        if use_median:
            for key in float_keys:
                final_weights[key] = median_weights[key]
        else:
            total_rep = agg_reps.sum()
            if total_rep < 1e-9:
                for key in float_keys:
                    final_weights[key] = median_weights[key]
            else:
                agg_reps = agg_reps / total_rep
                for i, weight in enumerate(client_weights):
                    rep_prop = agg_reps[i]
                    if rep_prop <= 0:
                        continue
                    for key in float_keys:
                        if key not in final_weights:
                            final_weights[key] = weight[key] * rep_prop
                        else:
                            final_weights[key] += weight[key] * rep_prop

        best_client = int(np.argmax(self.reputations))
        for key in int_keys:
            final_weights[key] = client_weights[best_client][key]

        self.global_model.load_state_dict(final_weights)
        return {
            'round_reputations': round_reputations.tolist(),
            'ema_reputations': self.reputations.copy().tolist(),
            'active_clients': active,
            'used_median_fallback': use_median,
        }

    def evaluate(self, test_loader):
        """Evaluate the global model with comprehensive metrics.
        
        Returns:
            dict with:
              - accuracy, loss (core)
              - precision/recall/f1 (weighted, backward compat for plots)
              - f1_binary, precision_binary, recall_binary (IDS-relevant)
              - f1_macro (class-balanced view)
              - roc_auc (threshold-independent)
              - confusion_matrix ([TN, FP, FN, TP] flattened)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                loss_val = criterion(outputs, targets).item()
                if not np.isfinite(loss_val):
                    loss_val = 10.0  # cap exploded loss
                total_loss += loss_val
                num_batches += 1
                
                # Clamp outputs to prevent NaN from exploding weights
                # (adversarial model-scaling can blow up activations)
                outputs = torch.clamp(outputs, min=-50.0, max=50.0)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # P(attack)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Replace any residual NaN/Inf probabilities (safety net)
        nan_mask = ~np.isfinite(all_probs)
        if nan_mask.any():
            all_probs[nan_mask] = 0.5  # neutral probability
        
        metrics = {
            # Core
            'accuracy': accuracy_score(all_targets, all_preds) * 100,
            'loss': total_loss / max(num_batches, 1),
            
            # Weighted (backward compat — used by existing plots)
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
            'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
            
             # Binary metrics (attack = positive class)
            'precision_binary': precision_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
            'recall_binary': recall_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
            'f1_binary': f1_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
            
            # Macro (treats both classes equally)
            'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100,
            
            # ROC-AUC (threshold-independent) — guarded against degenerate cases
            'roc_auc': self._safe_roc_auc(all_targets, all_probs),
            
            # Confusion matrix [TN, FP, FN, TP]
            'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
        }
        
        return metrics

    def _safe_roc_auc(self, targets, probs):
        """Compute ROC-AUC with fallback for degenerate cases."""
        try:
            return roc_auc_score(targets, probs) * 100
        except (ValueError, IndexError):
            # NaN probs, single-class batch, etc. → return chance-level
            return 50.0

    def get_global_model(self):
        return self.global_model

    def checkpoint_if_best(self, metrics, round_num):
        """Save model state if current ROC-AUC is the best seen so far."""
        if round_num < self.defense_config.min_checkpoint_round:
            self.rounds_without_improvement += 1
            return False
        current_auc = metrics.get('roc_auc', metrics.get('accuracy', 0))
        if current_auc > self.best_auc:
            self.best_auc = current_auc
            self.best_metrics = metrics.copy()
            self.best_round = round_num
            self.best_model_state = copy.deepcopy(self.global_model.state_dict())
            self.rounds_without_improvement = 0
            return True
        self.rounds_without_improvement += 1
        return False

    def should_stop_early(self, patience=5):
        """Check if early stopping criterion is met."""
        return self.rounds_without_improvement >= patience

    def restore_best_model(self):
        """Restore the global model to the best checkpoint."""
        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)
            return self.best_metrics, self.best_round
        return None, None

    def save_best_model(self, filepath):
        """Save the best model checkpoint to disk."""
        if self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'metrics': self.best_metrics,
                'round': self.best_round
            }, filepath)
    def aggregate_multi_krum(self, client_weights, client_sizes):
        """Multi-Krum aggregation (Blanchard et al., 2017).

        Scores every client exactly like Krum, then averages the
        (n - f) clients with the lowest scores instead of keeping only one.
        """
        n = len(client_weights)
        f = max(1, int(n * 0.4))           # same assumption as Krum: up to 40 % malicious
        n_closest = max(1, n - f - 2)
        m = max(1, n - f)                  # number of clients kept and averaged
        keys = list(client_weights[0].keys())
        float_keys = [k for k in keys if client_weights[0][k].is_floating_point()]
        int_keys = [k for k in keys if k not in float_keys]

        flat = [torch.cat([w[k].flatten().float() for k in float_keys]) for w in client_weights]
        dists = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(flat[i] - flat[j]).item()
                dists[i, j] = d
                dists[j, i] = d

        scores = []
        for i in range(n):
            sorted_dists = torch.sort(dists[i])[0]
            scores.append(sorted_dists[1:1 + n_closest].sum().item())
        selected = [int(i) for i in np.argsort(scores)[:m]]

        agg = OrderedDict()
        for key in float_keys:
            agg[key] = torch.stack([client_weights[i][key] for i in selected], dim=0).mean(dim=0)
        for key in int_keys:
            agg[key] = client_weights[selected[0]][key]
        self.global_model.load_state_dict(agg)
        return {'selected_clients': selected}

    def aggregate_foolsgold(self, client_weights, client_sizes):
        """FoolsGold aggregation (Fung et al., 2020).

        Down-weights clients whose accumulated update history is very similar
        to another client's (the signature of coordinated poisoning).
        """
        keys = list(client_weights[0].keys())
        float_keys = [k for k in keys if client_weights[0][k].is_floating_point()]
        int_keys = [k for k in keys if k not in float_keys]
        global_state = self.global_model.state_dict()
        n = len(client_weights)

        deltas = torch.stack([
            torch.cat([(w[k] - global_state[k]).flatten().float() for k in float_keys])
            for w in client_weights
        ])
        if getattr(self, 'fg_history', None) is None:
            self.fg_history = torch.zeros_like(deltas)
        self.fg_history += deltas

        hist = self.fg_history / self.fg_history.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cs = (hist @ hist.T).cpu().numpy() - np.eye(n)
        maxcs = cs.max(axis=1)
        for i in range(n):                 # pardoning
            for j in range(n):
                if i != j and maxcs[i] < maxcs[j] and maxcs[j] > 0:
                    cs[i, j] = cs[i, j] * maxcs[i] / maxcs[j]
        wv = np.clip(1 - cs.max(axis=1), 0, 1)
        if wv.max() > 0:
            wv = wv / wv.max()
        wv[wv == 1] = 0.99
        with np.errstate(divide='ignore'):
            wv = np.log(wv / (1 - wv)) + 0.5
        wv[np.isinf(wv) & (wv > 0)] = 1.0
        wv = np.clip(np.nan_to_num(wv, nan=0.0, neginf=0.0), 0, 1)

        agg = OrderedDict()
        total = wv.sum()
        for key in float_keys:
            if total < 1e-9:
                agg[key] = global_state[key].clone()
            else:
                stacked = torch.stack([w[key] - global_state[key] for w in client_weights], dim=0)
                weights = torch.tensor(wv, dtype=stacked.dtype, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
                agg[key] = global_state[key] + (weights * stacked).sum(dim=0) / total
        best = int(np.argmax(wv))
        for key in int_keys:
            agg[key] = client_weights[best][key]
        self.global_model.load_state_dict(agg)
        return {'foolsgold_weights': wv.tolist()}