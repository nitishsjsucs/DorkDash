"""
DasherSwarmRunner: Orchestrates a swarm simulation connected to the v2 bandit platform.

Flow:
  1. Bandits propose per-zone incentives
  2. Swarm agents autonomously accept/reject
  3. Aggregate responses fed back to bandits as reward signal
  4. Repeat for N days

Also provides an OASIS bridge for Docker sidecar mode.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import numpy as np

from backend.swarm.marketplace_env import MarketplaceEnvironment


class DasherSwarmRunner:
    """
    Runs a multi-agent swarm experiment, optionally connected to v2 bandits.

    Usage (standalone):
        runner = DasherSwarmRunner(n_dashers=100, n_zones=10)
        results = await runner.run(n_days=7)

    Usage (connected to v2):
        runner = DasherSwarmRunner(n_dashers=100)
        runner.connect_bandit(bandit_instance, allocator_instance)
        results = await runner.run(n_days=7)
    """

    def __init__(
        self,
        n_zones: int = 10,
        n_dashers: int = 100,
        seed: int = 42,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        steps_per_day: int = 6,
    ):
        self.env = MarketplaceEnvironment(
            n_zones=n_zones,
            n_dashers=n_dashers,
            seed=seed,
            api_key=api_key,
            model=model,
        )
        self.steps_per_day = steps_per_day
        self.bandit = None
        self.allocator = None
        self.results: Optional[Dict[str, Any]] = None

    def connect_bandit(self, bandit, allocator=None):
        """Connect v2 bandit and optional allocator for incentive proposals."""
        self.bandit = bandit
        self.allocator = allocator

    def _bandit_propose_incentives(self, step: int) -> Dict[int, float]:
        """
        Get per-zone incentive proposals from the connected bandit.
        Falls back to uniform $2 if no bandit connected.
        """
        n_zones = self.env.n_zones

        if self.bandit is None:
            # Default: sweep different incentive levels for exploration
            base = 1.0 + (step % 6) * 0.5  # $1.0 to $3.5
            return {z: base + (z % 3) * 0.5 for z in range(n_zones)}

        # If bandit is connected, use its selections
        incentives = {}
        for z in range(n_zones):
            try:
                # Build a simple context for the bandit
                zone_state = self.env.zones[z]
                context = np.array([
                    zone_state.current_demand,
                    float(z) / n_zones,
                    float(step % self.steps_per_day) / self.steps_per_day,
                    0.5,  # peak indicator
                    0.5,  # tenure placeholder
                    0.0,  # spillover placeholder
                ])
                arm_id, _ = self.bandit.select_arm(context)
                # Map arm to dollar value
                arm_values = [0, 1, 2, 3, 4, 5]
                incentives[z] = float(arm_values[arm_id % len(arm_values)])
            except Exception:
                incentives[z] = 2.0

        return incentives

    def _feed_rewards_to_bandit(
        self, incentives: Dict[int, float], decisions: List[Dict[str, Any]]
    ):
        """Feed swarm agent responses back to the bandit as reward signals."""
        if self.bandit is None:
            return

        for d in decisions:
            zone = d.get("zone", 0)
            incentive = d.get("incentive", 0.0)
            accepted = d.get("accepted", False)
            try:
                self.bandit.update(
                    arm_id=int(incentive),
                    reward=1.0 if accepted else 0.0,
                    context=np.array([0.5, zone / 10, 0.5, 0.5, 0.5, 0.0]),
                )
            except Exception:
                pass

    async def run(self, n_days: int = 7) -> Dict[str, Any]:
        """Run a full swarm simulation for n_days."""
        self.env.reset()
        total_steps = n_days * self.steps_per_day

        all_step_results = []
        daily_summaries = []

        for step in range(total_steps):
            # Get incentive proposals
            incentives = self._bandit_propose_incentives(step)
            self.env.set_incentives(incentives)

            # Run one step
            step_result = await self.env.step()
            all_step_results.append({
                k: v for k, v in step_result.items() if k != "decisions"
            })

            # Feed rewards back to bandit
            self._feed_rewards_to_bandit(incentives, step_result["decisions"])

            # Daily summary
            if (step + 1) % self.steps_per_day == 0:
                day = (step + 1) // self.steps_per_day
                day_steps = all_step_results[
                    -self.steps_per_day:
                ]
                daily_summaries.append(self._summarize_day(day, day_steps))

        # Final aggregation
        self.results = {
            "n_days": n_days,
            "n_dashers": self.env.n_dashers,
            "n_zones": self.env.n_zones,
            "total_steps": total_steps,
            "final_zones": [z.to_dict() for z in self.env.zones],
            "daily_summaries": daily_summaries,
            "step_history": all_step_results,
            "agent_states": self.env.get_agent_states(),
            "aggregate": self._compute_aggregate(all_step_results),
        }
        return self.results

    def _summarize_day(
        self, day: int, day_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Produce a summary for one simulation day."""
        total_accepted = sum(s["n_accepted"] for s in day_steps)
        total_agents = sum(s["total_agents"] for s in day_steps)
        switches = sum(s.get("agent_switches", 0) for s in day_steps)

        return {
            "day": day,
            "total_offers": total_agents,
            "total_accepted": total_accepted,
            "acceptance_rate": total_accepted / max(1, total_agents),
            "zone_switches": switches,
            "zone_acceptance_rates": {
                z["zone_id"]: z["acceptance_rate"]
                for s in day_steps
                for z in s.get("zones", [])
            },
        }

    def _compute_aggregate(
        self, steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute experiment-wide aggregate metrics."""
        acceptance_rates = [s["acceptance_rate"] for s in steps]
        return {
            "mean_acceptance_rate": float(np.mean(acceptance_rates)),
            "std_acceptance_rate": float(np.std(acceptance_rates)),
            "total_offers": sum(s["total_agents"] for s in steps),
            "total_accepted": sum(s["n_accepted"] for s in steps),
            "total_zone_switches": sum(
                s.get("agent_switches", 0) for s in steps
            ),
            "acceptance_trend": [
                round(r, 3) for r in acceptance_rates
            ],
        }

    def get_state(self) -> Dict[str, Any]:
        """Return runner state for API consumption."""
        return {
            "is_complete": self.results is not None,
            "environment": self.env.get_state(),
            "results_summary": (
                self.results.get("aggregate") if self.results else None
            ),
        }


class OASISBridge:
    """
    Bridge to connect to an OASIS Docker sidecar for large-scale simulations.

    The OASIS sidecar runs in a Python 3.11 Docker container and communicates
    via HTTP API. This bridge translates between our DasherSwarm protocol
    and the OASIS agent graph format.

    Requires: docker-compose up oasis-sidecar
    """

    def __init__(self, oasis_url: str = "http://localhost:5050"):
        self.oasis_url = oasis_url
        self.session_id: Optional[str] = None

    async def is_available(self) -> bool:
        """Check if the OASIS sidecar is running."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.oasis_url}/health", timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def create_session(
        self,
        n_agents: int,
        profiles: List[Dict[str, Any]],
    ) -> str:
        """Create an OASIS simulation session with Dasher profiles."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.oasis_url}/session/create",
                json={"n_agents": n_agents, "profiles": profiles},
            ) as resp:
                data = await resp.json()
                self.session_id = data["session_id"]
                return self.session_id

    async def step(
        self, incentives: Dict[int, float]
    ) -> Dict[str, Any]:
        """Send incentive offers to OASIS and get agent decisions."""
        if not self.session_id:
            raise RuntimeError("No active OASIS session")

        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.oasis_url}/session/{self.session_id}/step",
                json={"incentives": incentives},
            ) as resp:
                return await resp.json()

    async def close(self):
        """Tear down the OASIS session."""
        if self.session_id:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{self.oasis_url}/session/{self.session_id}/close"
                    )
            except Exception:
                pass
            self.session_id = None
