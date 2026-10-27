"""Attack configuration presets for federated poisoning experiments."""

from dataclasses import dataclass


@dataclass
class AttackConfig:
    """Tunable parameters for a federated poisoning attack."""

    name: str
    lr_multiplier: float = 1.0
    epoch_multiplier: float = 1.0
    scale_factor: float = 5.0
    noise_std: float = 0.0
    benign_flip_ratio: float = 0.5
    flip_sign: float = 1.0  # +1 amplify delta, -1 reverse direction (model poisoning)

    @classmethod
    def for_attack(cls, attack_type: str) -> "AttackConfig":
        """Return preset config for a known attack type."""
        presets = {
            "label_flipping": cls(
                name="label_flipping",
                lr_multiplier=3.0,
                epoch_multiplier=3.0,
                scale_factor=5.0,
                noise_std=0.0,
                benign_flip_ratio=0.5,
                flip_sign=1.0,
            ),
            "model_poisoning": cls(
                name="model_poisoning",
                lr_multiplier=1.0,
                epoch_multiplier=1.0,
                scale_factor=5.0,
                noise_std=5.0,
                benign_flip_ratio=0.0,
                flip_sign=-1.0,
            ),
        }
        if attack_type not in presets:
            raise ValueError(
                f"Unknown attack_type '{attack_type}'. "
                f"Use one of: {list(presets.keys())}"
            )
        return presets[attack_type]


ATTACK_LABELS = {
    "label_flipping": "Label Flipping + Scaling",
    "model_poisoning": "Model Poisoning (Reversed Delta + Noise)",
}

ALL_ATTACK_TYPES = ["label_flipping", "model_poisoning"]
