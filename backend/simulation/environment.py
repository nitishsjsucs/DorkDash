"""
DoorDash-style marketplace simulation with 10 geographic zones.
Models Dasher incentive-response as a function of:
  - incentive amount
  - zone supply deficit
  - time of day
  - Dasher tenure
  - non-stationary shifts (both gradual drift and abrupt regime changes)
  - zone interference / spillover (incentivising one zone pulls supply from neighbours)
  - delayed / batched feedback (outcomes arrive with lag)
  - zone clusters for networked bandit learning

References:
  - DoorDash MAB platform (Weinstein, Dec 2025): batch reward worker, treatment-effect modelling
  - Cluster-based bandits under interference (arXiv 2025)
  - DISCO (ASOS 2024): operational constraints in bandit allocation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque


@dataclass
class ZoneConfig:
    zone_id: int
    name: str
    base_demand: float          # avg orders per hour
    base_supply: float          # avg available dashers per hour
    elasticity: float           # how responsive dashers are to incentives
    peak_hours: List[int]       # hours with elevated demand
    peak_multiplier: float      # demand multiplier during peak
    cluster_id: int = 0         # geographic cluster for networked bandits


# Clusters: 0=SF Core, 1=SF Outer, 2=East Bay, 3=South Bay
DEFAULT_ZONES = [
    ZoneConfig(0, "Downtown SF",       120, 80,  0.7, [11,12,13,17,18,19,20], 1.8, cluster_id=0),
    ZoneConfig(1, "Mission District",   90, 70,  0.8, [11,12,17,18,19],       1.5, cluster_id=0),
    ZoneConfig(2, "SoMa",              100, 65,  0.6, [11,12,13,18,19,20],    1.7, cluster_id=0),
    ZoneConfig(3, "Marina",             70, 55,  0.9, [12,13,18,19],          1.4, cluster_id=1),
    ZoneConfig(4, "Sunset",             60, 50,  1.0, [17,18,19],             1.3, cluster_id=1),
    ZoneConfig(5, "Richmond",           55, 45,  1.1, [17,18,19],             1.3, cluster_id=1),
    ZoneConfig(6, "Oakland Downtown",   85, 60,  0.75,[11,12,17,18,19],       1.6, cluster_id=2),
    ZoneConfig(7, "Berkeley",           65, 50,  0.85,[11,12,13,18,19],       1.5, cluster_id=2),
    ZoneConfig(8, "San Jose Central",  110, 75,  0.65,[11,12,13,17,18,19,20], 1.7, cluster_id=3),
    ZoneConfig(9, "Palo Alto",          50, 40,  0.95,[12,13,18,19],          1.4, cluster_id=3),
]

# Adjacency matrix: geographic proximity for interference and information sharing
# W[i][j] = spillover weight from zone j's incentive on zone i's supply
# Higher weight = zones are closer / more coupled
DEFAULT_ADJACENCY = np.array([
    # DT   Mis  SoMa Mar  Sun  Rich Oak  Berk SJ   PA
    [0.0,  0.6, 0.7, 0.4, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],  # Downtown SF
    [0.6,  0.0, 0.5, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0],  # Mission
    [0.7,  0.5, 0.0, 0.3, 0.1, 0.1, 0.2, 0.1, 0.0, 0.0],  # SoMa
    [0.4,  0.2, 0.3, 0.0, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],  # Marina
    [0.1,  0.2, 0.1, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # Sunset
    [0.1,  0.1, 0.1, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Richmond
    [0.1,  0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0],  # Oakland DT
    [0.0,  0.0, 0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],  # Berkeley
    [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # San Jose
    [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],  # Palo Alto
])


@dataclass
class Dasher:
    dasher_id: int
    home_zone: int
    tenure_days: int            # how long they've been a dasher
    base_responsiveness: float  # intrinsic probability of responding [0,1]
    active: bool = True
    incentive_count_7d: int = 0 # incentives received in last 7 days (for habituation caps)
    cohort_arm: Optional[str] = None  # sticky assignment cohort (for consistency)


@dataclass
class SimulationState:
    current_day: int = 0
    current_hour: int = 8
    total_steps: int = 0
    elasticity_drift: Dict[int, float] = field(default_factory=dict)
    # Spillover tracking: how much supply was pulled from each zone this step
    spillover_effects: Dict[int, float] = field(default_factory=dict)
    # Pending delayed outcomes: list of (resolve_step, dasher_id, zone_id, incentive, true_prob)
    pending_outcomes: deque = field(default_factory=deque)
    # Change-point log for non-stationarity auditing
    change_point_log: List[Dict] = field(default_factory=list)
    # Per-zone incentive spend tracking (for budget constraints)
    daily_spend: Dict[int, float] = field(default_factory=dict)
    total_spend: Dict[int, float] = field(default_factory=dict)


class MarketplaceSimulator:
    """Simulates a DoorDash-like marketplace across multiple zones."""

    def __init__(
        self,
        zones: Optional[List[ZoneConfig]] = None,
        n_dashers_per_zone: int = 100,
        seed: int = 42,
        drift_period_days: int = 7,
        drift_magnitude: float = 0.15,
        spillover_strength: float = 0.15,
        feedback_delay_hours: int = 2,
        abrupt_change_prob: float = 0.03,
        adjacency: Optional[np.ndarray] = None,
    ):
        self.rng = np.random.RandomState(seed)
        self.zones = zones or DEFAULT_ZONES
        self.n_zones = len(self.zones)
        self.drift_period = drift_period_days
        self.drift_magnitude = drift_magnitude
        self.spillover_strength = spillover_strength
        self.feedback_delay_hours = feedback_delay_hours
        self.abrupt_change_prob = abrupt_change_prob
        self.adjacency = adjacency if adjacency is not None else DEFAULT_ADJACENCY[:self.n_zones, :self.n_zones]
        self.state = SimulationState()

        # Cluster structure
        self.clusters: Dict[int, List[int]] = {}
        for zone in self.zones:
            cid = zone.cluster_id
            if cid not in self.clusters:
                self.clusters[cid] = []
            self.clusters[cid].append(zone.zone_id)

        # Initialize dashers
        self.dashers: List[Dasher] = []
        dasher_id = 0
        for zone in self.zones:
            for _ in range(n_dashers_per_zone):
                tenure = self.rng.randint(1, 730)  # 1 day to 2 years
                responsiveness = np.clip(
                    self.rng.beta(2, 5) + 0.1 * (tenure / 365), 0.05, 0.95
                )
                self.dashers.append(Dasher(
                    dasher_id=dasher_id,
                    home_zone=zone.zone_id,
                    tenure_days=tenure,
                    base_responsiveness=responsiveness,
                ))
                dasher_id += 1

        self.n_dashers = len(self.dashers)
        # Initialize elasticity drift at 0
        for z in self.zones:
            self.state.elasticity_drift[z.zone_id] = 0.0
            self.state.spillover_effects[z.zone_id] = 0.0
            self.state.daily_spend[z.zone_id] = 0.0
            self.state.total_spend[z.zone_id] = 0.0

        # Recent incentive actions per zone (for spillover computation)
        self._recent_incentives: Dict[int, List[float]] = {z.zone_id: [] for z in self.zones}

    def reset(self):
        """Reset simulation to initial state."""
        self.state = SimulationState()
        for z in self.zones:
            self.state.elasticity_drift[z.zone_id] = 0.0
            self.state.spillover_effects[z.zone_id] = 0.0
            self.state.daily_spend[z.zone_id] = 0.0
            self.state.total_spend[z.zone_id] = 0.0
        self._recent_incentives = {z.zone_id: [] for z in self.zones}
        for d in self.dashers:
            d.incentive_count_7d = 0
            d.cohort_arm = None

    def _get_demand(self, zone: ZoneConfig, hour: int, day: int) -> float:
        """Get current demand for a zone, including day-of-week effects."""
        demand = zone.base_demand
        if hour in zone.peak_hours:
            demand *= zone.peak_multiplier
        # Weekend boost
        day_of_week = day % 7
        if day_of_week >= 5:  # Sat, Sun
            demand *= 1.25
        # Add noise
        demand *= (1 + self.rng.normal(0, 0.05))
        return max(0, demand)

    def _get_supply(self, zone: ZoneConfig, hour: int, day: int) -> float:
        """Get organic (unincentivized) supply for a zone."""
        supply = zone.base_supply
        # Less organic supply during off-peak
        if hour < 10 or hour > 21:
            supply *= 0.5
        elif hour in zone.peak_hours:
            supply *= 0.9  # some dashers come organically during peak
        # Weekend effect
        day_of_week = day % 7
        if day_of_week >= 5:
            supply *= 1.1
        supply *= (1 + self.rng.normal(0, 0.05))
        return max(0, supply)

    def get_supply_deficit(self, zone_id: int) -> float:
        """Current supply deficit ratio: (demand - supply) / demand. Positive = undersupplied."""
        zone = self.zones[zone_id]
        demand = self._get_demand(zone, self.state.current_hour, self.state.current_day)
        supply = self._get_supply(zone, self.state.current_hour, self.state.current_day)
        if demand <= 0:
            return 0.0
        return (demand - supply) / demand

    def get_zone_context(self, zone_id: int) -> Dict[str, float]:
        """Get the context vector for a zone at the current time step."""
        zone = self.zones[zone_id]
        deficit = self.get_supply_deficit(zone_id)
        hour = self.state.current_hour
        day = self.state.current_day
        spillover = self.state.spillover_effects.get(zone_id, 0.0)

        return {
            "zone_id": float(zone_id),
            "cluster_id": float(zone.cluster_id),
            "supply_deficit": deficit,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_peak": float(hour in zone.peak_hours),
            "is_weekend": float((day % 7) >= 5),
            "base_elasticity": zone.elasticity + self.state.elasticity_drift[zone_id],
            "spillover_pressure": spillover,
            "neighbour_avg_deficit": self._neighbour_avg_deficit(zone_id),
        }

    def get_dasher_context(self, dasher: Dasher) -> Dict[str, float]:
        """Get context features for a specific dasher."""
        return {
            "tenure_normalized": dasher.tenure_days / 365.0,
            "base_responsiveness": dasher.base_responsiveness,
            "home_zone": float(dasher.home_zone),
        }

    def _neighbour_avg_deficit(self, zone_id: int) -> float:
        """Average supply deficit of neighbouring zones, weighted by adjacency."""
        weights = self.adjacency[zone_id]
        total_w = weights.sum()
        if total_w < 1e-6:
            return 0.0
        deficits = [self.get_supply_deficit(j) * weights[j] for j in range(self.n_zones) if j != zone_id]
        return sum(deficits) / total_w

    def _compute_spillover(self):
        """
        Compute spillover effects: incentivising zone j pulls Dashers from zone i.
        Spillover = sum_j adjacency[i,j] * avg_incentive_in_j * spillover_strength
        This reduces effective supply in zone i when nearby zones are heavily incentivised.
        """
        for i in range(self.n_zones):
            spill = 0.0
            for j in range(self.n_zones):
                if i == j:
                    continue
                avg_inc_j = np.mean(self._recent_incentives[j]) if self._recent_incentives[j] else 0.0
                spill += self.adjacency[i, j] * avg_inc_j * self.spillover_strength
            self.state.spillover_effects[i] = spill

    def compute_response_probability(
        self,
        dasher: Dasher,
        incentive_amount: float,
        zone_id: int,
    ) -> float:
        """
        Compute the TRUE probability a dasher responds to an incentive.
        This is the ground truth — models try to learn this.

        Uses a logistic function:
          P(respond) = sigmoid(
              base_responsiveness_logit
              + elasticity * incentive_amount
              + deficit_bonus
              + tenure_effect
              + time_effect
              + spillover_effect
              + habituation_penalty
          )
        """
        zone = self.zones[zone_id]
        effective_elasticity = zone.elasticity + self.state.elasticity_drift[zone_id]
        deficit = self.get_supply_deficit(zone_id)
        hour = self.state.current_hour
        spillover = self.state.spillover_effects.get(zone_id, 0.0)

        # Convert base responsiveness to logit space
        p = np.clip(dasher.base_responsiveness, 0.01, 0.99)
        base_logit = np.log(p / (1 - p))

        # Normalised tenure in [0, ~2+]
        tenure_norm = dasher.tenure_days / 365.0

        # Incentive effect (core): higher incentive → higher response, but with
        # a strong *multiplicative* interaction with tenure.  Veterans know the
        # per-order economics and are far less swayed by bonus cash (they stop
        # responding to marginal $ once their hourly wage clears a threshold);
        # new dashers remain highly elastic.  This creates a *crossing* — the
        # optimal arm literally depends on tenure.
        tenure_elasticity_mult = max(0.1, 1.0 - 0.7 * tenure_norm)
        incentive_effect = (
            effective_elasticity
            * np.sqrt(max(0, incentive_amount))
            * tenure_elasticity_mult
        )

        # Deficit bonus: dashers more responsive when zone is tight (more orders = more earnings)
        deficit_bonus = 0.5 * max(0, deficit)

        # Deficit × incentive interaction.  In a tight zone organic earnings
        # already clear, so paying a *big* incentive adds little marginal
        # acceptance and actually signals desperation (negative coefficient).
        # In a slack zone a big incentive is the only thing that converts.
        deficit_incentive_interaction = -0.35 * max(0, deficit) * incentive_amount

        # Tenure baseline shift (intercept only): veterans are slightly less
        # responsive overall because their outside option is higher.
        tenure_factor = -0.2 * tenure_norm

        # Time effect: dashers more responsive during evening
        time_effect = 0.2 * np.sin(2 * np.pi * (hour - 6) / 24)

        # Spillover: nearby zones pulling dashers away reduces response here
        spillover_penalty = -0.3 * spillover

        # Habituation: dashers who received many recent incentives become less responsive
        habituation_penalty = -0.05 * min(dasher.incentive_count_7d, 10)

        logit = (base_logit + incentive_effect + deficit_bonus
                 + deficit_incentive_interaction + tenure_factor
                 + time_effect + spillover_penalty + habituation_penalty)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
        return probability

    def true_treatment_effect(
        self,
        dasher: Dasher,
        incentive_amount: float,
        zone_id: int,
    ) -> float:
        """
        The TRUE causal treatment effect: P(respond|incentive) - P(respond|no incentive).
        This is what causal models try to estimate.
        """
        p_treated = self.compute_response_probability(dasher, incentive_amount, zone_id)
        p_control = self.compute_response_probability(dasher, 0.0, zone_id)
        return p_treated - p_control

    def send_incentive(
        self,
        dasher_id: int,
        incentive_amount: float,
        zone_id: Optional[int] = None,
        policy_prob: Optional[float] = None,
    ) -> Tuple[bool, float, Dict]:
        """
        Send an incentive to a dasher and observe whether they respond.
        Supports both immediate and delayed feedback modes.

        Args:
            policy_prob: probability with which the policy chose this arm (for propensity logging)

        Returns:
            (responded: bool, true_te: float, info: dict)
        """
        dasher = self.dashers[dasher_id]
        if zone_id is None:
            zone_id = dasher.home_zone

        prob = self.compute_response_probability(dasher, incentive_amount, zone_id)
        true_te = self.true_treatment_effect(dasher, incentive_amount, zone_id)
        responded = self.rng.random() < prob

        # Track spend
        self.state.daily_spend[zone_id] = self.state.daily_spend.get(zone_id, 0.0) + incentive_amount
        self.state.total_spend[zone_id] = self.state.total_spend.get(zone_id, 0.0) + incentive_amount

        # Track for spillover computation
        self._recent_incentives[zone_id].append(incentive_amount)

        # Track dasher habituation
        if incentive_amount > 0:
            dasher.incentive_count_7d += 1

        info = {
            "dasher_id": dasher_id,
            "zone_id": zone_id,
            "cluster_id": self.zones[zone_id].cluster_id,
            "incentive": incentive_amount,
            "response_prob": prob,
            "true_treatment_effect": true_te,
            "responded": responded,
            "hour": self.state.current_hour,
            "day": self.state.current_day,
            "step": self.state.total_steps,
            "policy_prob": policy_prob,  # logged for doubly-robust correction
            "spillover_pressure": self.state.spillover_effects.get(zone_id, 0.0),
        }
        return responded, true_te, info

    def send_incentive_delayed(
        self,
        dasher_id: int,
        incentive_amount: float,
        zone_id: Optional[int] = None,
        policy_prob: Optional[float] = None,
    ) -> Dict:
        """
        Send incentive with delayed feedback: outcome resolves after feedback_delay_hours.
        Returns info dict immediately; call resolve_pending() to get outcomes later.
        """
        dasher = self.dashers[dasher_id]
        if zone_id is None:
            zone_id = dasher.home_zone

        prob = self.compute_response_probability(dasher, incentive_amount, zone_id)
        true_te = self.true_treatment_effect(dasher, incentive_amount, zone_id)
        resolve_step = self.state.total_steps + self.feedback_delay_hours

        self.state.daily_spend[zone_id] = self.state.daily_spend.get(zone_id, 0.0) + incentive_amount
        self.state.total_spend[zone_id] = self.state.total_spend.get(zone_id, 0.0) + incentive_amount
        self._recent_incentives[zone_id].append(incentive_amount)
        if incentive_amount > 0:
            dasher.incentive_count_7d += 1

        pending = {
            "resolve_step": resolve_step,
            "dasher_id": dasher_id,
            "zone_id": zone_id,
            "cluster_id": self.zones[zone_id].cluster_id,
            "incentive": incentive_amount,
            "response_prob": prob,
            "true_treatment_effect": true_te,
            "hour": self.state.current_hour,
            "day": self.state.current_day,
            "step": self.state.total_steps,
            "policy_prob": policy_prob,
        }
        self.state.pending_outcomes.append(pending)

        return {**pending, "responded": None}  # outcome unknown yet

    def resolve_pending(self) -> List[Dict]:
        """
        Resolve any pending delayed outcomes whose delay has elapsed.
        Returns list of resolved outcome dicts with 'responded' filled in.
        """
        resolved = []
        while (self.state.pending_outcomes and
               self.state.pending_outcomes[0]["resolve_step"] <= self.state.total_steps):
            pending = self.state.pending_outcomes.popleft()
            responded = self.rng.random() < pending["response_prob"]
            pending["responded"] = responded
            resolved.append(pending)
        return resolved

    def step(self):
        """Advance simulation by one hour."""
        self.state.current_hour += 1
        self.state.total_steps += 1

        # Recompute spillover effects each step
        self._compute_spillover()

        # Clear recent incentives buffer (only look at current step's incentives)
        self._recent_incentives = {z.zone_id: [] for z in self.zones}

        if self.state.current_hour >= 24:
            self.state.current_hour = 0
            self.state.current_day += 1

            # Reset daily spend
            for z in self.zones:
                self.state.daily_spend[z.zone_id] = 0.0

            # Decay dasher incentive counts weekly
            if self.state.current_day % 7 == 0:
                for d in self.dashers:
                    d.incentive_count_7d = max(0, d.incentive_count_7d - 3)

            # Apply non-stationary drift every drift_period days
            if self.state.current_day % self.drift_period == 0:
                self._apply_drift()

            # Random abrupt regime change (competitor entry, holiday, etc.)
            if self.rng.random() < self.abrupt_change_prob:
                self._apply_abrupt_change()

    def _apply_drift(self):
        """Gradual shift of elasticity parameters (slowly-varying non-stationarity)."""
        for zone in self.zones:
            drift = self.rng.normal(0, self.drift_magnitude)
            self.state.elasticity_drift[zone.zone_id] += drift
            self.state.elasticity_drift[zone.zone_id] = np.clip(
                self.state.elasticity_drift[zone.zone_id], -0.5, 0.5
            )

    def _apply_abrupt_change(self):
        """
        Abrupt regime change: sudden shift in a random zone's elasticity.
        Models competitor entry, holiday demand spike, major event, etc.
        """
        zone_id = self.rng.randint(0, self.n_zones)
        old_drift = self.state.elasticity_drift[zone_id]
        shift = self.rng.choice([-0.4, -0.3, 0.3, 0.4])
        self.state.elasticity_drift[zone_id] = np.clip(old_drift + shift, -0.6, 0.6)

        event = {
            "day": self.state.current_day,
            "zone_id": zone_id,
            "zone_name": self.zones[zone_id].name,
            "old_drift": old_drift,
            "new_drift": self.state.elasticity_drift[zone_id],
            "shift": shift,
            "type": "abrupt_regime_change",
        }
        self.state.change_point_log.append(event)

    def generate_batch_data(
        self,
        n_steps: int = 168,  # 1 week of hours
        incentive_amounts: Optional[List[float]] = None,
        n_dashers_per_step: int = 20,
    ) -> List[Dict]:
        """
        Generate a batch of observational data for causal model training.
        Simulates random incentive assignment (like historical data).
        """
        if incentive_amounts is None:
            incentive_amounts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        records = []
        for _ in range(n_steps):
            for zone in self.zones:
                zone_ctx = self.get_zone_context(zone.zone_id)
                zone_dashers = [d for d in self.dashers if d.home_zone == zone.zone_id]

                selected = self.rng.choice(
                    len(zone_dashers),
                    size=min(n_dashers_per_step, len(zone_dashers)),
                    replace=False,
                )

                for idx in selected:
                    dasher = zone_dashers[idx]
                    incentive = self.rng.choice(incentive_amounts)
                    responded, true_te, info = self.send_incentive(
                        dasher.dasher_id, incentive, zone.zone_id
                    )
                    dasher_ctx = self.get_dasher_context(dasher)
                    record = {**info, **zone_ctx, **dasher_ctx}
                    records.append(record)

            self.step()

        return records

    def get_adjacency_matrix(self) -> List[List[float]]:
        """Get the zone adjacency matrix for the frontend."""
        return self.adjacency.tolist()

    def get_cluster_map(self) -> Dict[int, List[int]]:
        """Get cluster → zone_ids mapping."""
        return {k: list(v) for k, v in self.clusters.items()}

    def get_snapshot(self) -> Dict:
        """Get current state snapshot for API consumption."""
        zone_snapshots = []
        for zone in self.zones:
            deficit = self.get_supply_deficit(zone.zone_id)
            ctx = self.get_zone_context(zone.zone_id)
            zone_snapshots.append({
                "zone_id": zone.zone_id,
                "name": zone.name,
                "cluster_id": zone.cluster_id,
                "supply_deficit": deficit,
                "effective_elasticity": ctx["base_elasticity"],
                "is_peak": bool(ctx["is_peak"]),
                "demand": self._get_demand(zone, self.state.current_hour, self.state.current_day),
                "supply": self._get_supply(zone, self.state.current_hour, self.state.current_day),
                "spillover_pressure": self.state.spillover_effects.get(zone.zone_id, 0.0),
                "daily_spend": self.state.daily_spend.get(zone.zone_id, 0.0),
                "total_spend": self.state.total_spend.get(zone.zone_id, 0.0),
            })

        return {
            "day": self.state.current_day,
            "hour": self.state.current_hour,
            "total_steps": self.state.total_steps,
            "zones": zone_snapshots,
            "n_dashers": self.n_dashers,
            "clusters": self.get_cluster_map(),
            "change_points": self.state.change_point_log[-10:],
            "pending_outcomes": len(self.state.pending_outcomes),
        }
