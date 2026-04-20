"""
Dasher Swarm Simulation Module (v3)

Multi-agent simulation where LLM-powered Dasher agents autonomously decide
whether to accept/reject incentive offers, which zones to work in, and how
to respond to changing marketplace conditions.

Two execution backends:
  1. Built-in lightweight engine (Python 3.10+, no OASIS dependency)
  2. OASIS bridge (requires Docker sidecar with Python <3.12)
"""

from backend.swarm.dasher_agent import DasherAgent, DasherPersona
from backend.swarm.marketplace_env import MarketplaceEnvironment
from backend.swarm.swarm_runner import DasherSwarmRunner, OASISBridge
from backend.swarm.mirofish_bridge import MiroFishBridge

__all__ = [
    "DasherAgent",
    "DasherPersona",
    "MarketplaceEnvironment",
    "DasherSwarmRunner",
    "OASISBridge",
    "MiroFishBridge",
]
