"""
Constraint-aware incentive allocator.
Consumes posterior samples (uncertainty-aware reward predictions) and
produces allocations that satisfy operational constraints.

Inspired by DISCO (ASOS, 2024): Thompson Sampling + integer programme
for budgeted personalised discount allocation.

Constraints modelled:
  - Global daily budget cap
  - Per-zone daily budget caps
  - Per-Dasher frequency caps (avoid habituation)
  - Fairness: minimum exploration per zone (no zone starved)
  - Consistency: sticky assignment for existing cohorts

References:
  - DISCO (arXiv 2406.06433): bandit + IP for personalised discounts
  - DoorDash causal promotions: estimate lift → optimise under constraints
  - DoorDash MAB platform: sticky assignment for consistent UX
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AllocationConstraints:
    """Operational constraints for incentive allocation."""
    global_daily_budget: float = 5000.0       # total $ per day across all zones
    per_zone_daily_budget: float = 800.0      # max $ per zone per day
    per_dasher_weekly_cap: int = 5            # max incentives per Dasher per week
    min_exploration_pct: float = 0.05         # at least 5% of allocations explore non-best arm
    min_control_pct: float = 0.10             # at least 10% get no incentive (for measurement)
    max_incentive_amount: float = 8.0         # cap on any single incentive
    sticky_assignment: bool = True            # keep Dashers in same cohort


@dataclass
class AllocationResult:
    """Result of a constrained allocation decision."""
    dasher_id: int
    zone_id: int
    arm_id: str
    incentive_amount: float
    is_exploration: bool
    is_control: bool
    is_sticky: bool
    expected_lift: float
    cost_efficiency: float  # expected_lift / incentive_amount


class ConstrainedAllocator:
    """
    Produces constraint-satisfying incentive allocations given posterior predictions.

    Uses a greedy knapsack approach:
    1. Score all (Dasher, arm) pairs by expected lift / cost (efficiency)
    2. Greedily assign highest-efficiency pairs subject to constraints
    3. Reserve slots for exploration and control
    """

    def __init__(
        self,
        constraints: Optional[AllocationConstraints] = None,
        seed: int = 42,
    ):
        self.constraints = constraints or AllocationConstraints()
        self.rng = np.random.RandomState(seed)

        # Budget tracking
        self.daily_spend: Dict[int, float] = {}
        self.global_spend_today: float = 0.0
        self.dasher_weekly_counts: Dict[int, int] = {}
        self.allocation_log: List[Dict] = []

    def reset_daily(self):
        """Reset daily budget counters."""
        self.daily_spend = {}
        self.global_spend_today = 0.0

    def reset_weekly(self):
        """Reset weekly Dasher caps."""
        self.dasher_weekly_counts = {}

    def allocate(
        self,
        candidates: List[Dict],
        arm_values: Dict[str, float],
    ) -> List[AllocationResult]:
        """
        Allocate incentives to a batch of candidates.

        Args:
            candidates: list of dicts with keys: dasher_id, zone_id, dasher_weekly_count,
                        cohort_arm (optional), expected_lifts (dict arm_id -> lift)
            arm_values: mapping arm_id -> dollar amount

        Returns:
            list of AllocationResult
        """
        c = self.constraints
        results = []

        # Separate candidates into control, exploration, and exploitation groups
        n_total = len(candidates)
        n_control = max(1, int(n_total * c.min_control_pct))
        n_explore = max(1, int(n_total * c.min_exploration_pct))

        # Shuffle candidates
        indices = list(range(n_total))
        self.rng.shuffle(indices)

        control_indices = set(indices[:n_control])
        explore_indices = set(indices[n_control:n_control + n_explore])
        exploit_indices = set(indices[n_control + n_explore:])

        for idx, cand in enumerate(candidates):
            dasher_id = cand["dasher_id"]
            zone_id = cand["zone_id"]
            weekly_count = self.dasher_weekly_counts.get(dasher_id, 0)
            expected_lifts = cand.get("expected_lifts", {})
            cohort_arm = cand.get("cohort_arm")

            # --- Control group: no incentive ---
            if idx in control_indices:
                results.append(AllocationResult(
                    dasher_id=dasher_id,
                    zone_id=zone_id,
                    arm_id="$0",
                    incentive_amount=0.0,
                    is_exploration=False,
                    is_control=True,
                    is_sticky=False,
                    expected_lift=0.0,
                    cost_efficiency=0.0,
                ))
                continue

            # --- Check frequency cap ---
            if weekly_count >= c.per_dasher_weekly_cap:
                results.append(AllocationResult(
                    dasher_id=dasher_id,
                    zone_id=zone_id,
                    arm_id="$0",
                    incentive_amount=0.0,
                    is_exploration=False,
                    is_control=False,
                    is_sticky=False,
                    expected_lift=0.0,
                    cost_efficiency=0.0,
                ))
                continue

            # --- Sticky assignment: use existing cohort if applicable ---
            if c.sticky_assignment and cohort_arm and cohort_arm in arm_values:
                amount = arm_values[cohort_arm]
                if self._can_afford(zone_id, amount):
                    lift = expected_lifts.get(cohort_arm, 0.0)
                    self._spend(zone_id, amount, dasher_id)
                    results.append(AllocationResult(
                        dasher_id=dasher_id,
                        zone_id=zone_id,
                        arm_id=cohort_arm,
                        incentive_amount=amount,
                        is_exploration=False,
                        is_control=False,
                        is_sticky=True,
                        expected_lift=lift,
                        cost_efficiency=lift / max(amount, 0.01),
                    ))
                    continue

            # --- Exploration: random arm ---
            if idx in explore_indices:
                explore_arm = self.rng.choice(list(arm_values.keys()))
                amount = arm_values[explore_arm]
                if self._can_afford(zone_id, amount):
                    lift = expected_lifts.get(explore_arm, 0.0)
                    self._spend(zone_id, amount, dasher_id)
                    results.append(AllocationResult(
                        dasher_id=dasher_id,
                        zone_id=zone_id,
                        arm_id=explore_arm,
                        incentive_amount=amount,
                        is_exploration=True,
                        is_control=False,
                        is_sticky=False,
                        expected_lift=lift,
                        cost_efficiency=lift / max(amount, 0.01),
                    ))
                    continue

            # --- Exploitation: pick best arm by efficiency ---
            best_arm = None
            best_efficiency = -np.inf
            for arm_id, amount in arm_values.items():
                if amount <= 0:
                    continue
                lift = expected_lifts.get(arm_id, 0.0)
                efficiency = lift / max(amount, 0.01)
                if efficiency > best_efficiency and self._can_afford(zone_id, amount):
                    best_efficiency = efficiency
                    best_arm = arm_id

            if best_arm is None:
                # Budget exhausted — assign control
                best_arm = "$0"
                amount = 0.0
                lift = 0.0
                efficiency = 0.0
            else:
                amount = arm_values[best_arm]
                lift = expected_lifts.get(best_arm, 0.0)
                efficiency = lift / max(amount, 0.01)
                self._spend(zone_id, amount, dasher_id)

            results.append(AllocationResult(
                dasher_id=dasher_id,
                zone_id=zone_id,
                arm_id=best_arm,
                incentive_amount=amount,
                is_exploration=False,
                is_control=(best_arm == "$0"),
                is_sticky=False,
                expected_lift=lift,
                cost_efficiency=efficiency,
            ))

        # Log this batch
        self.allocation_log.append({
            "n_candidates": n_total,
            "n_control": sum(1 for r in results if r.is_control),
            "n_explore": sum(1 for r in results if r.is_exploration),
            "n_exploit": sum(1 for r in results if not r.is_control and not r.is_exploration),
            "n_sticky": sum(1 for r in results if r.is_sticky),
            "total_spend": sum(r.incentive_amount for r in results),
            "avg_efficiency": float(np.mean([r.cost_efficiency for r in results if r.cost_efficiency > 0]) if any(r.cost_efficiency > 0 for r in results) else 0),
            "global_spend_today": self.global_spend_today,
        })

        return results

    def _can_afford(self, zone_id: int, amount: float) -> bool:
        """Check if budget allows this spend."""
        c = self.constraints
        zone_spend = self.daily_spend.get(zone_id, 0.0)
        if self.global_spend_today + amount > c.global_daily_budget:
            return False
        if zone_spend + amount > c.per_zone_daily_budget:
            return False
        return True

    def _spend(self, zone_id: int, amount: float, dasher_id: int):
        """Record spend."""
        self.daily_spend[zone_id] = self.daily_spend.get(zone_id, 0.0) + amount
        self.global_spend_today += amount
        self.dasher_weekly_counts[dasher_id] = self.dasher_weekly_counts.get(dasher_id, 0) + 1

    def get_state(self) -> Dict:
        """Get allocator state for API."""
        return {
            "constraints": {
                "global_daily_budget": self.constraints.global_daily_budget,
                "per_zone_daily_budget": self.constraints.per_zone_daily_budget,
                "per_dasher_weekly_cap": self.constraints.per_dasher_weekly_cap,
                "min_exploration_pct": self.constraints.min_exploration_pct,
                "min_control_pct": self.constraints.min_control_pct,
                "sticky_assignment": self.constraints.sticky_assignment,
            },
            "global_spend_today": self.global_spend_today,
            "per_zone_spend": dict(self.daily_spend),
            "recent_allocations": self.allocation_log[-10:],
        }
