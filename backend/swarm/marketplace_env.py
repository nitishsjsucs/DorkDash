"""
MarketplaceEnvironment: The world that Dasher agents inhabit.

Manages zone states, incentive offers, demand fluctuations, and
collects agent responses each simulation step. Designed to interface
with the v2 bandit allocator (bandits propose incentives, agents respond).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.swarm.dasher_agent import DasherAgent, DasherPersona


# Bay Area zone names matching the v2 simulator
ZONE_NAMES = [
    "SF_Downtown", "SF_Mission", "SF_Sunset", "Oakland_DT",
    "Oakland_Hills", "Berkeley", "San_Jose_DT", "San_Jose_South",
    "Palo_Alto", "Mountain_View",
]

DASHER_FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
    "Avery", "Skyler", "Drew", "Jamie", "Dakota", "Reese", "Rowan",
    "Sage", "Phoenix", "Blair", "Harper", "Ellis", "Lane", "Emery",
    "Finley", "Hayden", "Kendall", "Parker", "Peyton", "Reagan",
    "Sawyer", "Cameron", "Devon",
]

SCHEDULE_PREFS = ["morning", "afternoon", "evening", "flexible"]


@dataclass
class ZoneState:
    """Current state of a single marketplace zone."""
    zone_id: int
    name: str
    base_demand: float              # 0–1 tightness
    current_demand: float
    current_incentive: float        # $ set by bandit allocator
    dasher_count: int = 0           # dashers currently in this zone
    total_offers: int = 0
    total_accepts: int = 0
    total_rejects: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_offers == 0:
            return 0.0
        return self.total_accepts / self.total_offers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "current_demand": round(self.current_demand, 3),
            "current_incentive": round(self.current_incentive, 2),
            "dasher_count": self.dasher_count,
            "total_offers": self.total_offers,
            "total_accepts": self.total_accepts,
            "acceptance_rate": round(self.acceptance_rate, 3),
        }


class MarketplaceEnvironment:
    """
    Step-based marketplace simulation environment.

    Usage:
        env = MarketplaceEnvironment(n_zones=10, n_dashers=100)
        env.reset()
        for day in range(14):
            env.set_incentives(bandit_proposed_incentives)
            results = await env.step()
            # results contains per-agent decisions
    """

    def __init__(
        self,
        n_zones: int = 10,
        n_dashers: int = 100,
        seed: int = 42,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.n_zones = n_zones
        self.n_dashers = n_dashers
        self.seed = seed
        self.api_key = api_key
        self.model = model
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.zones: List[ZoneState] = []
        self.agents: List[DasherAgent] = []
        self.step_count: int = 0
        self.day_count: int = 0
        self.step_history: List[Dict[str, Any]] = []

        # Adjacency for spillover (same as v2 simulator)
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> np.ndarray:
        """Build zone adjacency matrix (geographic proximity)."""
        adj = np.zeros((self.n_zones, self.n_zones))
        # Cluster structure: zones 0-2 (SF), 3-5 (East Bay), 6-9 (South Bay)
        clusters = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        for cluster in clusters:
            for i in cluster:
                for j in cluster:
                    if i != j and i < self.n_zones and j < self.n_zones:
                        adj[i, j] = 0.3
        # Cross-cluster weak links
        cross_links = [(2, 3), (5, 6), (2, 8)]
        for i, j in cross_links:
            if i < self.n_zones and j < self.n_zones:
                adj[i, j] = adj[j, i] = 0.1
        return adj

    def reset(self):
        """Initialise / reinitialise the environment."""
        self.step_count = 0
        self.day_count = 0
        self.step_history = []

        # Create zones
        self.zones = []
        for z in range(self.n_zones):
            base_demand = 0.3 + self.rng.random() * 0.5  # 0.3–0.8
            self.zones.append(ZoneState(
                zone_id=z,
                name=ZONE_NAMES[z] if z < len(ZONE_NAMES) else f"Zone_{z}",
                base_demand=base_demand,
                current_demand=base_demand,
                current_incentive=2.0,  # default $2
            ))

        # Create Dasher agents with diverse personas
        self.agents = []
        for i in range(self.n_dashers):
            home_zone = self.rng.randint(0, self.n_zones - 1)
            persona = DasherPersona(
                agent_id=i,
                name=f"{DASHER_FIRST_NAMES[i % len(DASHER_FIRST_NAMES)]}_{i}",
                home_zone=home_zone,
                tenure_days=self.rng.randint(1, 500),
                base_elasticity=0.2 + self.rng.random() * 0.8,  # 0.2–1.0
                schedule_preference=self.rng.choice(SCHEDULE_PREFS),
                income_target_daily=40 + self.rng.random() * 80,  # $40–$120
                risk_tolerance=self.rng.random(),
                fatigue_rate=0.1 + self.rng.random() * 0.4,
            )
            self.agents.append(DasherAgent(
                persona=persona,
                api_key=self.api_key,
                model=self.model,
            ))

        # Assign dashers to zones
        self._update_zone_counts()

    def _update_zone_counts(self):
        """Count dashers per zone."""
        for z in self.zones:
            z.dasher_count = 0
        for agent in self.agents:
            zone_id = agent.memory.current_zone
            if zone_id is not None and 0 <= zone_id < self.n_zones:
                self.zones[zone_id].dasher_count += 1

    def set_incentives(self, incentives: Dict[int, float]):
        """
        Set incentive amounts per zone (proposed by the bandit allocator).
        incentives = {zone_id: dollar_amount}
        """
        for zone_id, amount in incentives.items():
            if 0 <= zone_id < self.n_zones:
                self.zones[zone_id].current_incentive = amount

    def _get_time_of_day(self) -> str:
        """Simulate time-of-day based on step count."""
        hour = (self.step_count * 4) % 24  # each step = 4 hours
        if hour < 12:
            return "morning"
        elif hour < 17:
            return "afternoon"
        else:
            return "evening"

    def _fluctuate_demand(self):
        """Add noise to zone demand each step."""
        for z in self.zones:
            noise = self.np_rng.normal(0, 0.05)
            z.current_demand = max(0.1, min(0.95, z.base_demand + noise))

    async def step(self) -> Dict[str, Any]:
        """
        Run one simulation step:
        1. Fluctuate demand
        2. Present offers to each Dasher in their current zone
        3. Collect decisions
        4. Process zone switches
        5. Return aggregated results
        """
        self.step_count += 1
        time_of_day = self._get_time_of_day()
        self._fluctuate_demand()

        # Check for new day (every 6 steps = 24 hours)
        if self.step_count % 6 == 0:
            self.day_count += 1
            for agent in self.agents:
                agent.reset_day()

        decisions = []
        for agent in self.agents:
            zone_id = agent.memory.current_zone
            if zone_id is None or zone_id >= self.n_zones:
                zone_id = agent.persona.home_zone
                agent.memory.current_zone = zone_id

            zone = self.zones[zone_id]

            # Build neighbour incentive map
            neighbour_incentives = {}
            for j in range(self.n_zones):
                if self.adjacency[zone_id, j] > 0:
                    neighbour_incentives[j] = self.zones[j].current_incentive

            offer = {
                "zone_id": zone_id,
                "incentive_amount": zone.current_incentive,
                "time_of_day": time_of_day,
                "zone_demand": zone.current_demand,
                "neighbour_incentives": neighbour_incentives,
            }

            decision = await agent.decide(offer)
            decisions.append(decision)

            # Update zone counters
            zone.total_offers += 1
            if decision.get("accepted"):
                zone.total_accepts += 1
            else:
                zone.total_rejects += 1

            # Handle zone switching
            switch_to = decision.get("would_switch_zone")
            if switch_to is not None and 0 <= switch_to < self.n_zones:
                agent.memory.current_zone = switch_to

        self._update_zone_counts()

        # Aggregate results
        n_accepted = sum(1 for d in decisions if d.get("accepted"))
        n_rejected = len(decisions) - n_accepted
        zone_summaries = [z.to_dict() for z in self.zones]

        step_result = {
            "step": self.step_count,
            "day": self.day_count,
            "time_of_day": time_of_day,
            "total_agents": len(self.agents),
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "acceptance_rate": n_accepted / max(1, len(decisions)),
            "zones": zone_summaries,
            "decisions": decisions,
            "agent_switches": sum(
                1 for d in decisions if d.get("would_switch_zone") is not None
            ),
        }
        self.step_history.append({
            k: v for k, v in step_result.items() if k != "decisions"
        })

        return step_result

    def get_state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "n_zones": self.n_zones,
            "n_dashers": self.n_dashers,
            "step_count": self.step_count,
            "day_count": self.day_count,
            "zones": [z.to_dict() for z in self.zones],
            "agent_summary": {
                "total": len(self.agents),
                "by_zone": {
                    z.zone_id: z.dasher_count for z in self.zones
                },
                "avg_earnings": (
                    sum(a.memory.total_earnings for a in self.agents)
                    / max(1, len(self.agents))
                ),
                "avg_acceptance_rate": (
                    sum(a.memory.recent_acceptance_rate() for a in self.agents)
                    / max(1, len(self.agents))
                ),
            },
            "step_history": self.step_history[-20:],  # last 20 steps
        }

    def get_agent_states(self) -> List[Dict[str, Any]]:
        """Return state for every agent."""
        return [a.get_state() for a in self.agents]
