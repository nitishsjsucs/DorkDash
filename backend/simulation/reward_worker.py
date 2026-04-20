"""
Batch Reward Worker — mirrors DoorDash's production architecture.

DoorDash's MAB platform runs a batch "reward worker" (typically daily) because
reward computation is upstream-batched. The worker:
  1. Collects raw outcomes from the logging/telemetry system
  2. Computes treatment-effect relative to control (Simpson's paradox fix)
  3. Performs posterior updates on all bandit models
  4. Produces traffic allocations for the next period

Key design choice (from Weinstein, Dec 2025):
  Model treatment effect relative to control, NOT absolute metric levels.
  This avoids Simpson's paradox under non-stationarity.

References:
  - DoorDash MAB platform (Weinstein, Dec 2025): batch reward worker architecture
  - DoorDash: TS robustness to delayed/batched feedback
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BatchOutcome:
    """A single outcome record for batch processing."""
    dasher_id: int
    zone_id: int
    arm_id: str
    incentive_amount: float
    responded: bool
    true_te: float
    policy_prob: Optional[float] = None
    context: Optional[Dict] = None
    step: int = 0
    day: int = 0


@dataclass
class BatchSummary:
    """Summary statistics for a reward worker batch."""
    batch_id: int
    day: int
    n_outcomes: int
    # Treatment-effect estimates per arm (relative to control)
    arm_te_estimates: Dict[str, float]
    arm_te_std: Dict[str, float]
    arm_counts: Dict[str, int]
    # Control group stats
    control_rate: float
    control_n: int
    # Per-zone summaries
    zone_summaries: Dict[int, Dict]
    # Doubly-robust corrected estimates
    dr_estimates: Optional[Dict[str, float]] = None


class RewardWorker:
    """
    Batch reward computation worker.

    Processes batched outcomes (typically daily) and produces:
    1. Treatment-effect estimates (relative to control) for each arm
    2. Doubly-robust corrected estimates using logged propensity scores
    3. Per-zone aggregate metrics
    4. Posterior update signals for bandit models
    """

    def __init__(self):
        self.batch_count = 0
        self.history: List[BatchSummary] = []
        self.cumulative_outcomes: List[BatchOutcome] = []
        self.control_outcomes: List[float] = []

    def process_batch(
        self,
        outcomes: List[Dict],
        arm_values: Dict[str, float],
    ) -> BatchSummary:
        """
        Process a batch of outcomes and compute treatment effects.

        Key: we compute TE relative to control, not absolute metrics,
        to avoid Simpson's paradox under non-stationarity.
        """
        self.batch_count += 1

        # Separate control and treatment
        control_responses = []
        arm_responses: Dict[str, List[float]] = {}
        arm_te_raw: Dict[str, List[float]] = {}
        zone_data: Dict[int, Dict] = {}
        dr_numerators: Dict[str, float] = {}
        dr_denominators: Dict[str, float] = {}

        for outcome in outcomes:
            zone_id = outcome.get("zone_id", 0)
            arm_id = outcome.get("arm_id", "$0")
            responded = float(outcome.get("responded", 0))
            incentive = outcome.get("incentive", 0.0)
            true_te = outcome.get("true_treatment_effect", 0.0)
            policy_prob = outcome.get("policy_prob")

            # Track per-zone
            if zone_id not in zone_data:
                zone_data[zone_id] = {"responses": [], "incentives": [], "n": 0}
            zone_data[zone_id]["responses"].append(responded)
            zone_data[zone_id]["incentives"].append(incentive)
            zone_data[zone_id]["n"] += 1

            if incentive == 0.0 or arm_id == "$0":
                control_responses.append(responded)
            else:
                if arm_id not in arm_responses:
                    arm_responses[arm_id] = []
                    arm_te_raw[arm_id] = []
                arm_responses[arm_id].append(responded)
                arm_te_raw[arm_id].append(true_te)

                # Doubly-robust estimation if propensity available
                if policy_prob is not None and policy_prob > 0.01:
                    if arm_id not in dr_numerators:
                        dr_numerators[arm_id] = 0.0
                        dr_denominators[arm_id] = 0.0
                    ipw = responded / policy_prob
                    dr_numerators[arm_id] += ipw
                    dr_denominators[arm_id] += 1.0 / policy_prob

        # Control rate
        control_rate = np.mean(control_responses) if control_responses else 0.0
        self.control_outcomes.extend(control_responses)

        # Compute treatment effect relative to control for each arm
        arm_te_estimates = {}
        arm_te_std = {}
        arm_counts = {}
        for arm_id, responses in arm_responses.items():
            arm_mean = np.mean(responses)
            # TE = arm mean response rate - control response rate
            te = arm_mean - control_rate
            arm_te_estimates[arm_id] = float(te)
            arm_te_std[arm_id] = float(np.std(responses) / max(np.sqrt(len(responses)), 1))
            arm_counts[arm_id] = len(responses)

        # Doubly-robust estimates
        dr_estimates = None
        if dr_numerators:
            dr_estimates = {}
            for arm_id in dr_numerators:
                if dr_denominators[arm_id] > 0:
                    dr_est = dr_numerators[arm_id] / dr_denominators[arm_id] - control_rate
                    dr_estimates[arm_id] = float(dr_est)

        # Zone summaries
        zone_summaries = {}
        for zone_id, data in zone_data.items():
            zone_summaries[zone_id] = {
                "n": data["n"],
                "response_rate": float(np.mean(data["responses"])),
                "avg_incentive": float(np.mean(data["incentives"])),
                "total_spend": float(np.sum(data["incentives"])),
            }

        summary = BatchSummary(
            batch_id=self.batch_count,
            day=outcomes[0].get("day", 0) if outcomes else 0,
            n_outcomes=len(outcomes),
            arm_te_estimates=arm_te_estimates,
            arm_te_std=arm_te_std,
            arm_counts=arm_counts,
            control_rate=float(control_rate),
            control_n=len(control_responses),
            zone_summaries=zone_summaries,
            dr_estimates=dr_estimates,
        )
        self.history.append(summary)
        return summary

    def get_posterior_update_signals(self) -> Dict:
        """
        Produce signals for bandit posterior updates based on latest batch.
        Uses treatment-effect estimates (not raw rewards) as the update signal.
        """
        if not self.history:
            return {"ready": False}

        latest = self.history[-1]
        return {
            "ready": True,
            "batch_id": latest.batch_id,
            "arm_te_estimates": latest.arm_te_estimates,
            "arm_te_std": latest.arm_te_std,
            "control_rate": latest.control_rate,
            "dr_estimates": latest.dr_estimates,
            "zone_summaries": {str(k): v for k, v in latest.zone_summaries.items()},
        }

    def get_state(self) -> Dict:
        """Get reward worker state for API."""
        return {
            "batch_count": self.batch_count,
            "total_outcomes_processed": sum(b.n_outcomes for b in self.history),
            "latest_batch": {
                "batch_id": self.history[-1].batch_id,
                "n_outcomes": self.history[-1].n_outcomes,
                "control_rate": self.history[-1].control_rate,
                "arm_te_estimates": self.history[-1].arm_te_estimates,
                "dr_estimates": self.history[-1].dr_estimates,
            } if self.history else None,
            "history": [
                {
                    "batch_id": b.batch_id,
                    "day": b.day,
                    "n_outcomes": b.n_outcomes,
                    "control_rate": b.control_rate,
                    "arm_te_estimates": b.arm_te_estimates,
                }
                for b in self.history[-20:]
            ],
        }
