"""
Linear Thompson Sampling (LinTS) contextual bandit.
Extends DoorDash's vanilla TS by incorporating context:
  - zone supply deficit
  - time of day (sin/cos encoding)
  - peak hour indicator
  - weekend indicator
  - zone elasticity
  - dasher tenure
  - dasher base responsiveness

References:
  - Agrawal & Goyal (2013), "Thompson Sampling for Contextual Bandits with Linear Payoffs"
  - arXiv 2508.13411, "Decentralized Contextual Bandits with Network Adaptivity"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


CONTEXT_FEATURES = [
    "supply_deficit",
    "hour_sin",
    "hour_cos",
    "is_peak",
    "is_weekend",
    "base_elasticity",
    "tenure_normalized",
    "base_responsiveness",
]


def build_context_vector(zone_ctx: Dict, dasher_ctx: Dict) -> np.ndarray:
    """Build a feature vector from zone and dasher context dicts."""
    features = []
    for feat in CONTEXT_FEATURES:
        val = zone_ctx.get(feat, dasher_ctx.get(feat, 0.0))
        features.append(float(val))
    # Add intercept term
    features.append(1.0)
    return np.array(features)


@dataclass
class LinArmState:
    arm_id: str
    d: int                                      # feature dimension
    B: np.ndarray = field(default=None)         # d x d precision matrix
    mu_hat: np.ndarray = field(default=None)    # d-dim mean estimate
    f: np.ndarray = field(default=None)         # cumulative reward-weighted features
    n_pulls: int = 0
    total_reward: float = 0.0

    def __post_init__(self):
        if self.B is None:
            self.B = np.eye(self.d)
        if self.mu_hat is None:
            self.mu_hat = np.zeros(self.d)
        if self.f is None:
            self.f = np.zeros(self.d)


class LinThompsonSampling:
    """
    Contextual bandit using Linear Thompson Sampling.
    Each arm has a linear reward model: E[reward | context x, arm a] = x^T theta_a
    Posterior over theta_a is multivariate normal.
    """

    def __init__(
        self,
        arm_ids: List[str],
        context_dim: int = len(CONTEXT_FEATURES) + 1,  # +1 for intercept
        v_squared: float = 0.1,  # exploration parameter (tuned down for faster convergence)
        lambda_reg: float = 0.5,  # regularization
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.d = context_dim
        self.v_squared = v_squared
        self.lambda_reg = lambda_reg
        self.arms: Dict[str, LinArmState] = {}

        for arm_id in arm_ids:
            self.arms[arm_id] = LinArmState(
                arm_id=arm_id,
                d=self.d,
                B=lambda_reg * np.eye(self.d),
            )

        self.cumulative_regret = 0.0
        self.regret_history: List[float] = []
        self.reward_history: List[Dict] = []

    def select_arm(self, context: np.ndarray) -> str:
        """
        Select arm using Thompson Sampling:
        1. For each arm, sample theta ~ N(mu_hat, v^2 * B^-1)
        2. Compute predicted reward: x^T theta
        3. Select arm with highest predicted reward
        """
        best_arm = None
        best_value = -np.inf
        samples = {}

        for arm_id, arm in self.arms.items():
            # Compute posterior covariance
            B_inv = np.linalg.inv(arm.B)
            # Sample from posterior
            theta_sample = self.rng.multivariate_normal(
                arm.mu_hat, self.v_squared * B_inv
            )
            # Predicted reward
            predicted = context @ theta_sample
            samples[arm_id] = predicted

            if predicted > best_value:
                best_value = predicted
                best_arm = arm_id

        return best_arm

    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """
        Update the arm's posterior with the new observation.
        B <- B + x x^T
        f <- f + reward * x
        mu_hat <- B^-1 f
        """
        arm = self.arms[arm_id]
        arm.n_pulls += 1
        arm.total_reward += reward

        # Rank-1 update to precision matrix
        arm.B += np.outer(context, context)
        arm.f += reward * context

        # Update mean estimate
        arm.mu_hat = np.linalg.solve(arm.B, arm.f)

    def batch_update(self, observations: List[Tuple[str, np.ndarray, float]]):
        """
        Batch update from list of (arm_id, context, reward) tuples.
        """
        for arm_id, context, reward in observations:
            self.update(arm_id, context, reward)

    def update_regret(self, chosen_reward: float, optimal_reward: float):
        """Track cumulative regret."""
        instant_regret = optimal_reward - chosen_reward
        self.cumulative_regret += max(0, instant_regret)
        self.regret_history.append(self.cumulative_regret)

    def get_arm_stats(self) -> List[Dict]:
        """Get statistics for all arms."""
        stats = []
        for arm_id, arm in self.arms.items():
            B_inv = np.linalg.inv(arm.B)
            uncertainty = np.sqrt(np.diag(B_inv)).mean()
            stats.append({
                "arm_id": arm_id,
                "n_pulls": arm.n_pulls,
                "total_reward": arm.total_reward,
                "avg_reward": arm.total_reward / max(1, arm.n_pulls),
                "mean_estimate": arm.mu_hat.tolist(),
                "uncertainty": uncertainty,
            })
        return stats

    def get_state(self) -> Dict:
        """Serialize full bandit state."""
        return {
            "algorithm": "LinThompsonSampling",
            "context_features": CONTEXT_FEATURES,
            "context_dim": self.d,
            "arms": self.get_arm_stats(),
            "cumulative_regret": self.cumulative_regret,
            "regret_history": self.regret_history[-100:],
            "total_pulls": sum(a.n_pulls for a in self.arms.values()),
        }
