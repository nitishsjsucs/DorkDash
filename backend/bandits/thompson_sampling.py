"""
Vanilla Thompson Sampling with Beta-Bernoulli model.
Replicates DoorDash's current system from Arjun Sharma's 2022 blog post:
- Beta distribution prior per arm
- Treatment-effect modeling (not raw metrics) per Weinstein's 2025 blog
- Weight decay for non-stationarity handling
- Batch updates (daily cadence)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ArmState:
    arm_id: str
    alpha: float = 1.0       # successes + prior
    beta_param: float = 1.0  # failures + prior
    n_pulls: int = 0
    total_reward: float = 0.0
    history: List[float] = field(default_factory=list)


class VanillaThompsonSampling:
    """
    Standard Thompson Sampling for discrete incentive arms.
    Models treatment effect (lift over control) using Beta-Bernoulli.
    """

    def __init__(
        self,
        arm_ids: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay_factor: float = 0.95,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.decay_factor = decay_factor
        self.arms: Dict[str, ArmState] = {}
        for arm_id in arm_ids:
            self.arms[arm_id] = ArmState(
                arm_id=arm_id,
                alpha=prior_alpha,
                beta_param=prior_beta,
            )
        self.cumulative_regret = 0.0
        self.regret_history: List[float] = []
        self.reward_history: List[Dict] = []

    def select_arm(self) -> str:
        """Sample from each arm's posterior and select the one with highest sample."""
        best_arm = None
        best_sample = -np.inf
        samples = {}

        for arm_id, arm in self.arms.items():
            sample = self.rng.beta(arm.alpha, arm.beta_param)
            samples[arm_id] = sample
            if sample > best_sample:
                best_sample = sample
                best_arm = arm_id

        return best_arm

    def update(self, arm_id: str, reward: float):
        """
        Update arm posterior with observed reward.
        reward should be 0 or 1 (Bernoulli).
        """
        arm = self.arms[arm_id]
        arm.n_pulls += 1
        arm.total_reward += reward

        if reward > 0.5:
            arm.alpha += 1.0
        else:
            arm.beta_param += 1.0

        arm.history.append(reward)

    def batch_update(self, observations: List[Tuple[str, float]]):
        """
        Batch update from a list of (arm_id, reward) tuples.
        Applies weight decay before updating to handle non-stationarity
        (per Arjun's blog: "apply a weight-decay parameter on previous observations").
        """
        # Apply decay to existing counts
        for arm in self.arms.values():
            arm.alpha = 1.0 + (arm.alpha - 1.0) * self.decay_factor
            arm.beta_param = 1.0 + (arm.beta_param - 1.0) * self.decay_factor

        # Apply new observations
        for arm_id, reward in observations:
            self.update(arm_id, reward)

    def update_regret(self, chosen_reward: float, optimal_reward: float):
        """Track cumulative regret."""
        instant_regret = optimal_reward - chosen_reward
        self.cumulative_regret += max(0, instant_regret)
        self.regret_history.append(self.cumulative_regret)

    def get_arm_stats(self) -> List[Dict]:
        """Get statistics for all arms."""
        stats = []
        for arm_id, arm in self.arms.items():
            mean = arm.alpha / (arm.alpha + arm.beta_param)
            var = (arm.alpha * arm.beta_param) / (
                (arm.alpha + arm.beta_param) ** 2 * (arm.alpha + arm.beta_param + 1)
            )
            stats.append({
                "arm_id": arm_id,
                "alpha": arm.alpha,
                "beta": arm.beta_param,
                "mean": mean,
                "variance": var,
                "n_pulls": arm.n_pulls,
                "total_reward": arm.total_reward,
                "avg_reward": arm.total_reward / max(1, arm.n_pulls),
            })
        return stats

    def get_state(self) -> Dict:
        """Serialize full bandit state."""
        return {
            "algorithm": "VanillaThompsonSampling",
            "arms": self.get_arm_stats(),
            "cumulative_regret": self.cumulative_regret,
            "regret_history": self.regret_history[-100:],  # last 100 for API
            "total_pulls": sum(a.n_pulls for a in self.arms.values()),
        }
