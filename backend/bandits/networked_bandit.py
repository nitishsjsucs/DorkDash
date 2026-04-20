"""
Networked Contextual Linear Thompson Sampling (NetLinTS).
Implements global+local decomposition across geographic zones with
adaptive information sharing via network weights.

Instead of treating 10 zones as independent contextual bandits or fully
pooling into one model, NetLinTS decomposes the reward model into:
  theta_i = theta_global + delta_i
where theta_global captures shared structure (time-of-day effects,
broad incentive elasticity) and delta_i captures zone-specific deviations.

Network weights control how much information neighbouring zones share.
Zones in the same cluster share more; distant zones share less.

References:
  - "Decentralised Contextual Linear Bandits over Networks" (arXiv 2508.13411, 2025)
  - NetLinUCB / Net-SGD-UCB: reduced learning complexity on shared structure
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
    "spillover_pressure",
    "neighbour_avg_deficit",
    "tenure_normalized",
    "base_responsiveness",
]


def build_context_vector(zone_ctx: Dict, dasher_ctx: Dict) -> np.ndarray:
    """Build feature vector from zone and dasher context dicts."""
    features = []
    for feat in CONTEXT_FEATURES:
        val = zone_ctx.get(feat, dasher_ctx.get(feat, 0.0))
        features.append(float(val))
    features.append(1.0)  # intercept
    return np.array(features)


@dataclass
class NodeArmState:
    """Per-node (zone), per-arm posterior state."""
    arm_id: str
    d: int
    B_local: np.ndarray = field(default=None)
    f_local: np.ndarray = field(default=None)
    n_pulls: int = 0
    total_reward: float = 0.0

    def __post_init__(self):
        if self.B_local is None:
            self.B_local = np.eye(self.d) * 0.5
        if self.f_local is None:
            self.f_local = np.zeros(self.d)


class NetLinTS:
    """
    Networked Linear Thompson Sampling.

    Architecture:
    - One global model (shared across all zones): captures common structure
    - Per-zone local residual models: capture zone-specific deviations
    - Network weights control information sharing between nodes
    - Posterior for zone i: theta_i = theta_global + delta_i

    The key insight: time-of-day effects and broad incentive elasticity
    are shared; zone-specific supply dynamics and dasher populations differ.
    """

    def __init__(
        self,
        arm_ids: List[str],
        n_zones: int = 10,
        adjacency: Optional[np.ndarray] = None,
        context_dim: int = len(CONTEXT_FEATURES) + 1,
        v_squared: float = 0.08,
        lambda_global: float = 0.5,
        lambda_local: float = 1.0,
        sharing_strength: float = 0.3,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.d = context_dim
        self.v_squared = v_squared
        self.n_zones = n_zones
        self.sharing_strength = sharing_strength
        self.arm_ids = arm_ids

        # Network structure
        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self.adjacency = np.eye(n_zones)

        # Normalise adjacency for weighted averaging
        row_sums = self.adjacency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.norm_adjacency = self.adjacency / row_sums

        # Global model: shared across all zones
        self.global_arms: Dict[str, Dict] = {}
        for arm_id in arm_ids:
            self.global_arms[arm_id] = {
                "B": lambda_global * np.eye(self.d),
                "f": np.zeros(self.d),
                "mu": np.zeros(self.d),
            }

        # Per-zone local models
        self.zone_arms: Dict[int, Dict[str, NodeArmState]] = {}
        for z in range(n_zones):
            self.zone_arms[z] = {}
            for arm_id in arm_ids:
                self.zone_arms[z][arm_id] = NodeArmState(
                    arm_id=arm_id,
                    d=self.d,
                    B_local=lambda_local * np.eye(self.d),
                )

        # Tracking
        self.cumulative_regret = 0.0
        self.regret_history: List[float] = []
        self.zone_regret: Dict[int, float] = {z: 0.0 for z in range(n_zones)}

    def _get_zone_posterior(self, zone_id: int, arm_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute combined posterior mean and covariance for a zone-arm pair.
        theta_zone = theta_global + delta_zone
        Posterior precision = B_global + B_local (precision addition)
        """
        g = self.global_arms[arm_id]
        l = self.zone_arms[zone_id][arm_id]

        # Combined precision and feature accumulator
        B_combined = g["B"] + l.B_local
        f_combined = g["f"] + l.f_local

        # Incorporate neighbour information (network sharing)
        for j in range(self.n_zones):
            if j == zone_id:
                continue
            w = self.norm_adjacency[zone_id, j] * self.sharing_strength
            if w > 1e-6:
                nj = self.zone_arms[j][arm_id]
                B_combined = B_combined + w * nj.B_local
                f_combined = f_combined + w * nj.f_local

        # Posterior mean
        try:
            B_inv = np.linalg.inv(B_combined)
            mu = B_inv @ f_combined
        except np.linalg.LinAlgError:
            B_inv = np.eye(self.d) * 0.1
            mu = np.zeros(self.d)

        return mu, B_inv

    def select_arm(self, context: np.ndarray, zone_id: int = 0) -> Tuple[str, float]:
        """
        Select arm using Thompson Sampling with networked posterior.
        Returns (arm_id, selection_probability_estimate).
        """
        best_arm = None
        best_value = -np.inf
        arm_values = {}

        for arm_id in self.arm_ids:
            mu, cov = self._get_zone_posterior(zone_id, arm_id)
            try:
                theta_sample = self.rng.multivariate_normal(mu, self.v_squared * cov)
            except (np.linalg.LinAlgError, ValueError):
                theta_sample = mu + self.rng.normal(0, 0.1, size=self.d)
            predicted = context @ theta_sample
            arm_values[arm_id] = predicted

            if predicted > best_value:
                best_value = predicted
                best_arm = arm_id

        # Estimate selection probability (for propensity logging)
        # Approximate: softmax of predicted values
        vals = np.array(list(arm_values.values()))
        vals_shifted = vals - vals.max()
        exp_vals = np.exp(vals_shifted / max(self.v_squared, 0.01))
        probs = exp_vals / exp_vals.sum()
        arm_idx = self.arm_ids.index(best_arm)
        selection_prob = float(probs[arm_idx])

        return best_arm, selection_prob

    def update(self, arm_id: str, context: np.ndarray, reward: float, zone_id: int = 0):
        """
        Update both global and local models with new observation.
        Global gets all data; local only gets zone-specific data.
        """
        outer = np.outer(context, context)

        # Update global model
        g = self.global_arms[arm_id]
        g["B"] = g["B"] + outer
        g["f"] = g["f"] + reward * context
        try:
            g["mu"] = np.linalg.solve(g["B"], g["f"])
        except np.linalg.LinAlgError:
            pass

        # Update local model for this zone
        l = self.zone_arms[zone_id][arm_id]
        l.B_local = l.B_local + outer
        l.f_local = l.f_local + reward * context
        l.n_pulls += 1
        l.total_reward += reward

    def update_regret(self, chosen_reward: float, optimal_reward: float, zone_id: int = 0):
        """Track cumulative regret globally and per-zone."""
        instant_regret = max(0, optimal_reward - chosen_reward)
        self.cumulative_regret += instant_regret
        self.regret_history.append(self.cumulative_regret)
        self.zone_regret[zone_id] = self.zone_regret.get(zone_id, 0.0) + instant_regret

    def get_arm_stats(self, zone_id: int = 0) -> List[Dict]:
        """Get arm statistics for a specific zone."""
        stats = []
        for arm_id in self.arm_ids:
            mu, cov = self._get_zone_posterior(zone_id, arm_id)
            l = self.zone_arms[zone_id][arm_id]
            uncertainty = np.sqrt(np.diag(cov)).mean()
            stats.append({
                "arm_id": arm_id,
                "n_pulls": l.n_pulls,
                "total_reward": l.total_reward,
                "avg_reward": l.total_reward / max(1, l.n_pulls),
                "posterior_mean_norm": float(np.linalg.norm(mu)),
                "uncertainty": float(uncertainty),
            })
        return stats

    def get_global_stats(self) -> List[Dict]:
        """Get global model statistics."""
        stats = []
        for arm_id in self.arm_ids:
            g = self.global_arms[arm_id]
            total_pulls = sum(self.zone_arms[z][arm_id].n_pulls for z in range(self.n_zones))
            stats.append({
                "arm_id": arm_id,
                "total_pulls_all_zones": total_pulls,
                "global_mean_norm": float(np.linalg.norm(g["mu"])),
            })
        return stats

    def get_state(self) -> Dict:
        """Serialize full bandit state."""
        zone_stats = {}
        for z in range(self.n_zones):
            zone_stats[str(z)] = self.get_arm_stats(z)

        return {
            "algorithm": "NetLinTS",
            "context_features": CONTEXT_FEATURES,
            "context_dim": self.d,
            "n_zones": self.n_zones,
            "sharing_strength": self.sharing_strength,
            "global_stats": self.get_global_stats(),
            "zone_stats": zone_stats,
            "cumulative_regret": self.cumulative_regret,
            "zone_regret": {str(k): v for k, v in self.zone_regret.items()},
            "regret_history": self.regret_history[-500:],
            "total_pulls": sum(
                self.zone_arms[z][a].n_pulls
                for z in range(self.n_zones)
                for a in self.arm_ids
            ),
        }
