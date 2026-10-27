"""HRA defense configuration. All HRA results reported in the paper use DefenseConfig.unified()."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DefenseConfig:
    """HRA settings. The defaults are the previous label-flipping configuration
    (alpha = 0.1, no exclusion; 'lf_tuned' in run_unified_defense.py). Baseline
    defenses (Krum, Trimmed Mean, etc.) receive DefenseConfig() but ignore the HRA fields."""

    alpha: float = 0.1
    use_instant_rep: bool = False
    hard_exclude_threshold: Optional[float] = None
    min_active_clients: int = 1
    use_median_fallback: bool = False
    min_checkpoint_round: int = 1

    @classmethod
    def for_attack(cls, attack_type: Optional[str]) -> "DefenseConfig":
        """Legacy per-attack presets, used only by main.py. Not used for the paper's results."""
        if attack_type == "model_poisoning":
            return cls(
                alpha=0.35,
                use_instant_rep=False,
                hard_exclude_threshold=0.05,
                min_active_clients=4,
                use_median_fallback=True,
                min_checkpoint_round=8,
            )
        return cls()
    

    @classmethod
    def unified(cls) -> "DefenseConfig":
        """Single fixed configuration used for ALL attacks (attack-agnostic)."""
        return cls(
            alpha=0.35,
            use_instant_rep=False,
            hard_exclude_threshold=0.05,
            min_active_clients=4,
            use_median_fallback=True,
            min_checkpoint_round=8,
        )
    