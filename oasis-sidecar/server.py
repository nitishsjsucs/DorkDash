"""
OASIS Sidecar Server

Runs inside a Python 3.11 Docker container with camel-oasis installed.
Exposes a REST API that the main platform's OASISBridge connects to.

Endpoints:
  GET  /health                       — liveness check
  POST /session/create               — create a simulation session
  POST /session/{id}/step            — run one simulation step
  POST /session/{id}/close           — tear down a session
  GET  /session/{id}/state           — get session state
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- OASIS imports (available in Docker container) ----
try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    import oasis
    from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph
    HAS_OASIS = True
except ImportError:
    HAS_OASIS = False

app = FastAPI(title="OASIS Dasher Sidecar", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Session storage ----
sessions: Dict[str, Dict[str, Any]] = {}


class CreateSessionRequest(BaseModel):
    n_agents: int = 10
    profiles: List[Dict[str, Any]] = []
    model_name: str = "gpt-4o-mini"


class StepRequest(BaseModel):
    incentives: Dict[str, float] = {}


# ---- Health ----

@app.get("/health")
def health():
    return {
        "status": "ok",
        "oasis_available": HAS_OASIS,
        "version": "1.0.0",
    }


# ---- Session Management ----

@app.post("/session/create")
async def create_session(req: CreateSessionRequest):
    """Create a new OASIS simulation session with Dasher agent profiles."""
    session_id = str(uuid.uuid4())[:8]

    if not HAS_OASIS:
        # Fallback: store session metadata without OASIS
        sessions[session_id] = {
            "id": session_id,
            "n_agents": req.n_agents,
            "profiles": req.profiles,
            "step_count": 0,
            "oasis_env": None,
            "mode": "fallback",
        }
        return {"session_id": session_id, "mode": "fallback", "n_agents": req.n_agents}

    # Write profiles to temp file for OASIS
    profile_path = f"/tmp/dasher_profiles_{session_id}.json"
    profiles_to_write = req.profiles
    if not profiles_to_write:
        # Load default profiles
        default_path = Path(__file__).parent / "dasher_profiles.json"
        if default_path.exists():
            profiles_to_write = json.loads(default_path.read_text())
        else:
            profiles_to_write = [
                {
                    "user_id": i,
                    "user_name": f"Dasher_{i}",
                    "bio": f"DoorDash Dasher #{i} with varied preferences and elasticity.",
                    "persona": "generic_dasher",
                }
                for i in range(req.n_agents)
            ]

    with open(profile_path, "w") as f:
        json.dump(profiles_to_write, f)

    try:
        openai_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
        )

        # Create a custom "marketplace" action set
        available_actions = [
            ActionType.CREATE_POST,      # represents ACCEPT incentive
            ActionType.LIKE_POST,        # represents positive signal
            ActionType.DISLIKE_POST,     # represents REJECT incentive
            ActionType.DO_NOTHING,       # no action
            ActionType.CREATE_COMMENT,   # agent commentary
        ]

        agent_graph = await generate_reddit_agent_graph(
            profile_path=profile_path,
            model=openai_model,
            available_actions=available_actions,
        )

        db_path = f"/tmp/oasis_dasher_{session_id}.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        env = oasis.make(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=db_path,
        )
        await env.reset()

        sessions[session_id] = {
            "id": session_id,
            "n_agents": len(profiles_to_write),
            "profiles": profiles_to_write,
            "step_count": 0,
            "oasis_env": env,
            "agent_graph": agent_graph,
            "mode": "oasis",
        }

        return {
            "session_id": session_id,
            "mode": "oasis",
            "n_agents": len(profiles_to_write),
        }

    except Exception as e:
        # Fall back to non-OASIS mode
        sessions[session_id] = {
            "id": session_id,
            "n_agents": req.n_agents,
            "profiles": profiles_to_write,
            "step_count": 0,
            "oasis_env": None,
            "mode": "fallback",
            "error": str(e),
        }
        return {
            "session_id": session_id,
            "mode": "fallback",
            "n_agents": req.n_agents,
            "warning": f"OASIS init failed: {e}",
        }


@app.post("/session/{session_id}/step")
async def step_session(session_id: str, req: StepRequest):
    """Run one simulation step with given incentive offers."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    session["step_count"] += 1

    if session["mode"] == "oasis" and session.get("oasis_env"):
        env = session["oasis_env"]
        agent_graph = session["agent_graph"]

        # Post incentive offers as content (agents will react)
        incentive_msg = "INCENTIVE OFFERS: " + ", ".join(
            f"Zone {k}: ${v:.2f}" for k, v in req.incentives.items()
        )

        # First agent posts the incentive offers
        actions = {}
        actions[agent_graph.get_agent(0)] = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": incentive_msg},
        )
        await env.step(actions)

        # All agents react with LLM decisions
        llm_actions = {
            agent: LLMAction()
            for _, agent in agent_graph.get_agents()
        }
        await env.step(llm_actions)

        return {
            "session_id": session_id,
            "step": session["step_count"],
            "mode": "oasis",
            "incentives_posted": incentive_msg,
            "agents_reacted": session["n_agents"],
        }
    else:
        # Fallback: simple probabilistic simulation
        import random
        decisions = []
        for i in range(session["n_agents"]):
            zone = i % max(1, len(req.incentives))
            zone_key = str(zone)
            amount = req.incentives.get(zone_key, 2.0)
            prob = min(0.95, 0.3 + amount * 0.15)
            accepted = random.random() < prob
            decisions.append({
                "agent_id": i,
                "zone": zone,
                "incentive": amount,
                "accepted": accepted,
            })

        n_accepted = sum(1 for d in decisions if d["accepted"])
        return {
            "session_id": session_id,
            "step": session["step_count"],
            "mode": "fallback",
            "total_agents": session["n_agents"],
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / max(1, session["n_agents"]),
            "decisions": decisions,
        }


@app.get("/session/{session_id}/state")
def get_session_state(session_id: str):
    """Get current session state."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return {
        "session_id": session["id"],
        "mode": session["mode"],
        "n_agents": session["n_agents"],
        "step_count": session["step_count"],
    }


@app.post("/session/{session_id}/close")
async def close_session(session_id: str):
    """Close and clean up a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions.pop(session_id)
    if session.get("oasis_env"):
        try:
            await session["oasis_env"].close()
        except Exception:
            pass

    return {"status": "closed", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
