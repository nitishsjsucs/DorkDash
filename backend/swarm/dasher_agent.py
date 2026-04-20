"""
DasherAgent: An LLM-powered autonomous agent representing a single Dasher.

Each agent has:
  - A persona (tenure, home zone, elasticity, schedule preference)
  - Episodic memory of past incentive offers and decisions
  - LLM-driven decision-making for accept/reject/zone-switch
  - Optional rule-based fallback when OpenAI is unavailable
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class DasherPersona:
    """Immutable identity of a simulated Dasher."""
    agent_id: int
    name: str
    home_zone: int
    tenure_days: int                    # 1 = brand-new, 365+ = veteran
    base_elasticity: float              # 0.0–1.0: how sensitive to incentives
    schedule_preference: str            # "morning", "afternoon", "evening", "flexible"
    income_target_daily: float          # $ goal per day
    risk_tolerance: float               # 0.0–1.0: willingness to try new zones
    fatigue_rate: float                 # how quickly motivation declines with repeated offers


@dataclass
class DasherMemory:
    """Episodic memory for one Dasher agent."""
    offers_received: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    earnings_history: List[float] = field(default_factory=list)
    current_zone: Optional[int] = None
    total_earnings: float = 0.0
    consecutive_rejections: int = 0
    offers_today: int = 0

    def record_offer(self, offer: Dict[str, Any]):
        self.offers_received.append(offer)
        self.offers_today += 1

    def record_decision(self, decision: Dict[str, Any]):
        self.decisions.append(decision)
        if decision.get("accepted"):
            self.consecutive_rejections = 0
            earnings = decision.get("earnings", 0.0)
            self.total_earnings += earnings
            self.earnings_history.append(earnings)
        else:
            self.consecutive_rejections += 1

    def reset_daily(self):
        self.offers_today = 0

    def recent_acceptance_rate(self, window: int = 10) -> float:
        recent = self.decisions[-window:]
        if not recent:
            return 0.5
        return sum(1 for d in recent if d.get("accepted")) / len(recent)

    def summary(self, max_recent: int = 5) -> str:
        """Produce a natural-language memory summary for the LLM prompt."""
        lines = [
            f"Total earnings so far: ${self.total_earnings:.2f}",
            f"Offers received today: {self.offers_today}",
            f"Recent acceptance rate: {self.recent_acceptance_rate():.0%}",
            f"Consecutive rejections: {self.consecutive_rejections}",
            f"Current zone: {self.current_zone}",
        ]
        if self.decisions:
            lines.append("Recent decisions:")
            for d in self.decisions[-max_recent:]:
                action = "ACCEPTED" if d.get("accepted") else "REJECTED"
                lines.append(
                    f"  - {action} ${d.get('incentive', '?')} in zone {d.get('zone', '?')} "
                    f"(reason: {d.get('reason', 'n/a')})"
                )
        return "\n".join(lines)


DASHER_SYSTEM_PROMPT = """You are a DoorDash Dasher (delivery driver). You make autonomous decisions
about whether to accept incentive offers based on your personal situation.

Your persona:
{persona_block}

Your memory:
{memory_block}

When presented with an incentive offer, respond with a JSON object:
{{
  "accepted": true/false,
  "reason": "brief explanation",
  "would_switch_zone": null or zone_id (integer),
  "satisfaction": 1-10
}}

Decision factors to consider:
- Is the incentive worth your time given your earnings goal?
- Are you fatigued from too many offers today?
- Does this zone match your preference/home zone?
- Have you been rejecting a lot lately (maybe lower your bar)?
- Are neighbouring zones offering better deals?

Respond ONLY with the JSON object, no other text."""


class DasherAgent:
    """A single LLM-powered Dasher agent."""

    def __init__(
        self,
        persona: DasherPersona,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.persona = persona
        self.memory = DasherMemory(current_zone=persona.home_zone)
        self.model = model

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.has_openai = (
            HAS_OPENAI
            and self.api_key
            and self.api_key != "sk-your-key-here"
        )
        if self.has_openai:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def _persona_block(self) -> str:
        p = self.persona
        return (
            f"Name: {p.name}\n"
            f"Home zone: {p.home_zone}\n"
            f"Tenure: {p.tenure_days} days\n"
            f"Elasticity (price sensitivity): {p.base_elasticity:.2f}\n"
            f"Schedule preference: {p.schedule_preference}\n"
            f"Daily income target: ${p.income_target_daily:.2f}\n"
            f"Risk tolerance: {p.risk_tolerance:.2f}\n"
            f"Fatigue rate: {p.fatigue_rate:.2f}"
        )

    def _build_system_prompt(self) -> str:
        return DASHER_SYSTEM_PROMPT.format(
            persona_block=self._persona_block(),
            memory_block=self.memory.summary(),
        )

    async def decide(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given an incentive offer, decide whether to accept or reject.

        offer = {
            "zone_id": int,
            "incentive_amount": float,
            "time_of_day": str,       # "morning"/"afternoon"/"evening"
            "zone_demand": float,     # 0–1 demand tightness
            "neighbour_incentives": dict[int, float],  # zone_id -> $ offered
        }
        """
        self.memory.record_offer(offer)

        if self.has_openai and self.client:
            return await self._llm_decide(offer)
        else:
            return self._rule_based_decide(offer)

    async def _llm_decide(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI to make the decision."""
        user_msg = (
            f"New incentive offer:\n"
            f"  Zone: {offer['zone_id']}\n"
            f"  Amount: ${offer['incentive_amount']:.2f}\n"
            f"  Time: {offer.get('time_of_day', 'unknown')}\n"
            f"  Zone demand: {offer.get('zone_demand', 0):.2f}\n"
            f"  Neighbour offers: {json.dumps(offer.get('neighbour_incentives', {}))}\n"
            f"\nDecide: accept or reject?"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            decision = json.loads(content)
        except Exception:
            decision = self._rule_based_decide(offer)

        decision["agent_id"] = self.persona.agent_id
        decision["zone"] = offer["zone_id"]
        decision["incentive"] = offer["incentive_amount"]
        decision["timestamp"] = datetime.now().isoformat()
        self.memory.record_decision(decision)
        return decision

    def _rule_based_decide(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic fallback: persona-weighted logistic decision.
        Mirrors the parametric simulator but with agent-specific noise.
        """
        p = self.persona
        amount = offer["incentive_amount"]
        zone = offer["zone_id"]
        demand = offer.get("zone_demand", 0.5)

        # Base acceptance probability
        base_prob = 1 / (1 + 2.71828 ** (-(amount - 2.5) * p.base_elasticity * 2))

        # Zone preference bonus
        zone_bonus = 0.1 if zone == p.home_zone else -0.05
        if p.risk_tolerance > 0.5:
            zone_bonus = max(zone_bonus, 0.0)  # risk-tolerant dashers don't penalise

        # Demand adjustment
        demand_bonus = (demand - 0.5) * 0.2

        # Fatigue penalty
        fatigue_penalty = min(self.memory.offers_today * p.fatigue_rate * 0.05, 0.3)

        # Earnings pressure: if below daily target, lower the bar
        remaining = max(0, p.income_target_daily - self.memory.total_earnings)
        urgency_bonus = min(remaining / p.income_target_daily * 0.15, 0.15) if p.income_target_daily > 0 else 0

        prob = base_prob + zone_bonus + demand_bonus - fatigue_penalty + urgency_bonus
        prob = max(0.05, min(0.95, prob))

        accepted = random.random() < prob

        # Zone switching logic
        would_switch = None
        neighbour_incentives = offer.get("neighbour_incentives", {})
        if not accepted and neighbour_incentives and p.risk_tolerance > 0.3:
            best_neighbour = max(neighbour_incentives, key=neighbour_incentives.get)
            if neighbour_incentives[best_neighbour] > amount * 1.3:
                would_switch = int(best_neighbour)

        reason = (
            f"prob={prob:.2f}, elasticity={p.base_elasticity:.2f}, "
            f"fatigue={fatigue_penalty:.2f}, urgency={urgency_bonus:.2f}"
        )

        decision = {
            "accepted": accepted,
            "reason": reason,
            "would_switch_zone": would_switch,
            "satisfaction": max(1, min(10, int(prob * 10))),
            "acceptance_probability": prob,
            "agent_id": p.agent_id,
            "zone": zone,
            "incentive": amount,
            "earnings": amount if accepted else 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        self.memory.record_decision(decision)
        return decision

    def reset_day(self):
        self.memory.reset_daily()

    def get_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.persona.agent_id,
            "name": self.persona.name,
            "home_zone": self.persona.home_zone,
            "current_zone": self.memory.current_zone,
            "tenure_days": self.persona.tenure_days,
            "total_earnings": self.memory.total_earnings,
            "offers_received": len(self.memory.offers_received),
            "acceptance_rate": self.memory.recent_acceptance_rate(),
            "consecutive_rejections": self.memory.consecutive_rejections,
        }
