"""
MiroFish Bridge: Connects to a MiroFish instance running via Docker.

MiroFish is a universal swarm intelligence engine (https://github.com/666ghj/MiroFish)
that uses LLM-powered agents with personas, memory, and social interactions.
This bridge translates between our DasherSwarm protocol and MiroFish's API.

Default endpoint: http://localhost:5001
Start via: docker-compose up mirofish
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class MiroFishBridge:
    """
    Bridge to a MiroFish Docker container for swarm intelligence simulation.

    MiroFish provides:
    - LLM agents with rich personas and episodic memory
    - Social interaction dynamics (agents observe and influence each other)
    - Swarm prediction aggregation
    - Built-in visualization (Vue frontend on port 3000 in MiroFish)
    """

    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if MiroFish is running."""
        if not HAS_HTTPX:
            # Try with urllib as fallback
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{self.base_url}/health",
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    self._available = resp.status == 200
            except Exception:
                self._available = False
            return self._available

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def create_scenario(
        self,
        name: str,
        description: str,
        agent_profiles: List[Dict[str, Any]],
        n_rounds: int = 10,
    ) -> Dict[str, Any]:
        """
        Create a MiroFish scenario with Dasher agent profiles.

        Each profile should include:
        - name: Agent name
        - bio: Rich persona description (MiroFish uses this for LLM personality)
        - attributes: Dict of custom attributes (zone, elasticity, etc.)
        """
        payload = {
            "name": name,
            "description": description,
            "agents": agent_profiles,
            "rounds": n_rounds,
            "config": {
                "interaction_mode": "marketplace",
                "allow_agent_communication": True,
                "aggregation_method": "weighted_vote",
            },
        }

        if HAS_HTTPX:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/scenario/create",
                    json=payload,
                )
                return resp.json()
        else:
            import urllib.request
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/scenario/create",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())

    async def run_prediction(
        self,
        scenario_id: str,
        question: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a swarm prediction within a scenario.

        The question is presented to all agents, who independently reason
        and then their answers are aggregated using swarm intelligence.

        Example:
            question = "Should zone 3 incentive be increased from $2 to $4?"
            context = {"zone_3_demand": 0.8, "current_acceptance_rate": 0.3}
        """
        payload = {
            "scenario_id": scenario_id,
            "question": question,
            "context": context,
        }

        if HAS_HTTPX:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/predict",
                    json=payload,
                )
                return resp.json()
        else:
            import urllib.request
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/predict",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())

    async def get_agent_states(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Get individual agent states from a MiroFish scenario."""
        if HAS_HTTPX:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/api/scenario/{scenario_id}/agents",
                )
                return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/api/scenario/{scenario_id}/agents",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())

    def generate_dasher_profiles(self, n_agents: int = 10) -> List[Dict[str, Any]]:
        """
        Generate MiroFish-compatible Dasher agent profiles.
        These are rich persona descriptions that MiroFish's LLM agents
        will use to make autonomous decisions.
        """
        import random

        archetypes = [
            {
                "type": "veteran_morning",
                "bio_template": (
                    "Experienced DoorDash Dasher with {tenure} months of experience. "
                    "Prefers morning shifts in {zone}. Moderate price sensitivity — "
                    "accepts $2+ incentives in home zone but needs $3+ to travel. "
                    "Daily income goal: ${income}. Knows the best routes."
                ),
                "elasticity": 0.5,
                "risk_tolerance": 0.3,
            },
            {
                "type": "new_flexible",
                "bio_template": (
                    "New DoorDash Dasher, just started {tenure} weeks ago. "
                    "Very price sensitive — accepts almost any incentive to build experience. "
                    "Flexible schedule, willing to work any zone. Daily income goal: ${income}."
                ),
                "elasticity": 0.9,
                "risk_tolerance": 0.8,
            },
            {
                "type": "selective_evening",
                "bio_template": (
                    "Part-time DoorDash Dasher in {zone}. Works evenings after day job. "
                    "Only takes high-value incentives ($4+). Sticks to home zone. "
                    "Daily income goal: ${income}. Quality over quantity."
                ),
                "elasticity": 0.2,
                "risk_tolerance": 0.1,
            },
            {
                "type": "social_fulltime",
                "bio_template": (
                    "Full-time Dasher in {zone} with {tenure} months experience. "
                    "Watches what other dashers do and tends to follow the crowd. "
                    "Will cross zones for good incentives ($3+). Daily income goal: ${income}."
                ),
                "elasticity": 0.6,
                "risk_tolerance": 0.5,
            },
            {
                "type": "strategic_optimizer",
                "bio_template": (
                    "Strategic DoorDash Dasher in {zone}. Compares zone incentives "
                    "before accepting. High risk tolerance, travels far for right price. "
                    "{tenure} months experience. Daily income goal: ${income}."
                ),
                "elasticity": 0.7,
                "risk_tolerance": 0.9,
            },
        ]

        zones = [
            "SF Downtown", "SF Mission", "SF Sunset", "Oakland Downtown",
            "Oakland Hills", "Berkeley", "San Jose Downtown", "San Jose South",
            "Palo Alto", "Mountain View",
        ]
        names = [
            "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
            "Avery", "Skyler", "Drew", "Jamie", "Dakota", "Reese", "Rowan",
        ]

        profiles = []
        for i in range(n_agents):
            arch = archetypes[i % len(archetypes)]
            zone = zones[i % len(zones)]
            name = f"{names[i % len(names)]}_{i}"
            tenure = random.randint(1, 36)
            income = random.randint(40, 130)

            profiles.append({
                "name": name,
                "bio": arch["bio_template"].format(
                    tenure=tenure, zone=zone, income=income,
                ),
                "attributes": {
                    "home_zone": zone,
                    "elasticity": arch["elasticity"],
                    "risk_tolerance": arch["risk_tolerance"],
                    "archetype": arch["type"],
                    "tenure_months": tenure,
                    "daily_income_goal": income,
                },
            })

        return profiles

    def get_status(self) -> Dict[str, Any]:
        """Return bridge status for API."""
        return {
            "base_url": self.base_url,
            "available": self._available,
            "has_httpx": HAS_HTTPX,
        }
