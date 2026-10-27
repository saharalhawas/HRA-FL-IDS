import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from losses import FocalLoss
from attack_config import AttackConfig


class FederatedClient:
    def __init__(self, client_id, data, device, model_type='mlp',
                 is_malicious=False, attack_type=None, batch_size=16,
                 loss_type='ce', global_class_weights=None,
                 attack_config: AttackConfig = None):
        self.client_id = client_id
        self.X, self.y = data
        self.device = device
        self.model_type = model_type
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.loss_type = loss_type
        self.attack_config = attack_config

        if self.is_malicious and self.attack_type == 'label_flipping':
            attack_mask = (self.y == 1)
            benign_mask = (self.y == 0)

            self.y[attack_mask] = 0

            benign_flip_ratio = (
                attack_config.benign_flip_ratio if attack_config else 0.5
            )
            benign_indices = np.where(benign_mask)[0]
            flip_count = int(len(benign_indices) * benign_flip_ratio)
            if flip_count > 0:
                flip_indices = np.random.choice(
                    benign_indices, size=flip_count, replace=False
                )
                self.y[flip_indices] = 1

        if global_class_weights is not None:
            self.class_weights = global_class_weights.to(device)
        else:
            self.class_weights = torch.ones(2, dtype=torch.float32).to(device)

        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(self.X, dtype=torch.float32),
                torch.tensor(self.y, dtype=torch.long)
            ),
            batch_size=batch_size, shuffle=True, drop_last=True
        )

    def train_local_model(self, global_model, epochs=5, lr=0.01):
        """Train local model; malicious clients apply post-training poisoning."""
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        cfg = self.attack_config
        actual_epochs = epochs
        actual_lr = lr
        if self.is_malicious and cfg is not None:
            actual_lr = lr * cfg.lr_multiplier
            actual_epochs = int(epochs * cfg.epoch_multiplier)

        if self.model_type == 'mlp':
            optimizer = optim.AdamW(
                model.parameters(), lr=actual_lr, weight_decay=1e-4
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        if self.loss_type == 'focal':
            criterion = FocalLoss(gamma=2.0, alpha=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        for _ in range(actual_epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        state_dict = model.state_dict()

        if self.is_malicious and cfg is not None:
            global_state = global_model.state_dict()

            if self.attack_type in ('label_flipping', 'model_poisoning'):
                scale = cfg.scale_factor
                flip_sign = cfg.flip_sign
                for key in state_dict:
                    if not state_dict[key].is_floating_point():
                        continue
                    delta = state_dict[key] - global_state[key]
                    poisoned = global_state[key] + flip_sign * scale * delta
                    if cfg.noise_std > 0:
                        ref_std = torch.std(global_state[key].float()).clamp(min=1e-6)
                        poisoned = poisoned + (
                            cfg.noise_std * ref_std
                            * torch.randn_like(state_dict[key])
                        )
                    state_dict[key] = poisoned

        return state_dict

    def get_update_size(self):
        return len(self.X)
