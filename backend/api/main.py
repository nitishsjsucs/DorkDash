"""
FastAPI backend for the Agentic Causal Bandit Platform.
Exposes REST endpoints for simulation, bandit state, BO, causal insights, and agent chat.
"""

import os
import sys
from pathlib import Path

# Ensure backend package is importable
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import asyncio

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from backend.simulation.experiment_runner import ExperimentRunner
from backend.agent.meta_agent import ExperimentAgent
from backend.swarm.swarm_runner import DasherSwarmRunner, OASISBridge
from backend.swarm.mirofish_bridge import MiroFishBridge
from backend.cache import (
    save_experiment_snapshot,
    save_swarm_result,
    load_experiment_snapshot,
    load_swarm_result,
    cache_status,
)


app = FastAPI(
    title="Agentic Causal Bandit Platform",
    description="DoorDash Dasher Incentive Optimization — Contextual Bandits + BO + Causal ML + LLM Agent",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
runner: Optional[ExperimentRunner] = None
swarm_runner: Optional[DasherSwarmRunner] = None
oasis_bridge = OASISBridge()
mirofish_bridge = MiroFishBridge()
agent: Optional[ExperimentAgent] = None
experiment_results: Optional[Dict] = None
causal_segments: Optional[Dict] = None

# Cached snapshots loaded from disk at startup. These survive backend restarts
# and ship with the repo so a fresh clone can see prior experiment + swarm
# results without having to run anything. Live runs overwrite both the cache
# file and these in-memory copies.
cached_experiment_snapshot: Optional[Dict] = load_experiment_snapshot()
cached_swarm_snapshot: Optional[Dict] = load_swarm_result()

# If a prior snapshot exists, expose its experiment_results immediately so
# /experiment/results works on fresh startup.
if cached_experiment_snapshot and cached_experiment_snapshot.get("experiment_results"):
    experiment_results = cached_experiment_snapshot["experiment_results"]
    causal_segments = experiment_results.get("causal", {}).get("hte_segments")


# ---------- Request/Response Models ----------

class ExperimentConfig(BaseModel):
    n_days: int = 14
    n_dashers_per_zone: int = 10
    seed: int = 42

class BatchConfig(BaseModel):
    n_steps: int = 24
    n_dashers_per_zone: int = 10

class ChatMessage(BaseModel):
    message: str

class BOQueryConfig(BaseModel):
    zone_id: int = 0
    n_points: int = 100

class SwarmConfig(BaseModel):
    n_dashers: int = 50
    n_zones: int = 10
    n_days: int = 3
    seed: int = 42
    model: str = "gpt-4o-mini"
    connect_bandit: bool = False


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "name": "Agentic Causal Bandit Platform",
        "description": "Dasher Incentive Optimization for DoorDash Supply team",
        "version": "2.0",
        "components": [
            "MarketplaceSimulator (10 zones, interference, delayed feedback, regime changes)",
            "VanillaThompsonSampling (DoorDash baseline)",
            "LinThompsonSampling (contextual bandit)",
            "NetLinTS (networked contextual bandit with global+local decomposition)",
            "ShapeConstrainedBO (monotone+concave GP for continuous incentives)",
            "CausalML (DML + CausalForest for HTE)",
            "ConstrainedAllocator (DISCO-style budgets, caps, fairness)",
            "BatchRewardWorker (treatment-effect relative to control)",
            "NonStationarityDetector (CUSUM + posterior drift + sliding window)",
            "LLM MetaAgent (OpenAI function-calling)",
        ],
    }


@app.post("/experiment/init")
def init_experiment(config: ExperimentConfig):
    """Initialize a new experiment."""
    global runner, agent, experiment_results, causal_segments
    runner = ExperimentRunner(seed=config.seed, n_dashers_per_zone=config.n_dashers_per_zone)
    agent = ExperimentAgent()
    experiment_results = None
    causal_segments = None
    return {"status": "initialized", "config": config.model_dump()}


@app.post("/experiment/run")
def run_full_experiment(config: ExperimentConfig):
    """Run a complete multi-day experiment. This may take a few seconds."""
    global runner, experiment_results, causal_segments, agent, cached_experiment_snapshot

    runner = ExperimentRunner(seed=config.seed, n_dashers_per_zone=config.n_dashers_per_zone)
    results = runner.run_full_experiment(
        n_days=config.n_days,
        n_dashers_per_zone=config.n_dashers_per_zone,
    )
    experiment_results = results

    # Extract causal segments for the agent
    causal_segments = results.get("causal", {}).get("hte_segments")

    # Wire up agent data sources
    if agent is None:
        agent = ExperimentAgent()
    agent.set_data_sources({
        "vanilla_ts": runner.vanilla_ts,
        "contextual_ts": runner.contextual_ts,
        "net_ts": runner.net_ts,
        "bo": runner.bo,
        "simulator": runner.simulator,
        "causal_segments": causal_segments,
        "change_detector": runner.change_detector,
        "reward_worker": runner.reward_worker,
        "allocator": runner.allocator,
        "dose_response": runner.dose_response,
        "interference_est": runner.interference_est,
    })

    # ---- Persist everything the frontend needs to render the Overview tab ----
    try:
        sim_snapshot = runner.simulator.get_snapshot()
    except Exception:
        sim_snapshot = None
    try:
        adjacency = {
            "adjacency": runner.simulator.get_adjacency_matrix().tolist(),
            "clusters": runner.simulator.get_cluster_map(),
        }
    except Exception:
        adjacency = None
    bo_posteriors: Dict[int, Any] = {}
    for zid in range(min(3, runner.simulator.n_zones)):
        try:
            bo_posteriors[zid] = runner.bo.get_posterior(zone_id=zid, n_points=100)
        except Exception:
            pass

    cached_experiment_snapshot = save_experiment_snapshot(
        experiment_results=results,
        sim_snapshot=sim_snapshot,
        adjacency=adjacency,
        bo_posteriors=bo_posteriors,
    )

    return results


@app.post("/experiment/step")
def run_step(config: BatchConfig):
    """Run a single batch (e.g., one day)."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized. Call /experiment/init first.")
    summary = runner.run_batch(n_steps=config.n_steps, n_dashers_per_zone=config.n_dashers_per_zone)
    return summary


@app.get("/experiment/state")
def get_experiment_state():
    """Get current experiment state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.get_state()


@app.get("/experiment/results")
def get_experiment_results():
    """Get full experiment results (after /experiment/run)."""
    global experiment_results
    if experiment_results is None:
        raise HTTPException(status_code=400, detail="No experiment results. Run /experiment/run first.")
    return experiment_results


# ---------- Bandit Endpoints ----------

@app.get("/bandits/vanilla_ts")
def get_vanilla_ts():
    """Get vanilla Thompson Sampling state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.vanilla_ts.get_state()


@app.get("/bandits/contextual_ts")
def get_contextual_ts():
    """Get contextual LinTS state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.contextual_ts.get_state()


@app.get("/bandits/net_ts")
def get_net_ts():
    """Get networked LinTS state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.net_ts.get_state()


@app.get("/bandits/regret_comparison")
def get_regret_comparison():
    """Get regret comparison between algorithms."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner._get_regret_comparison()


# ---------- Bayesian Optimization Endpoints ----------

@app.get("/bo/state")
def get_bo_state():
    """Get Bayesian Optimization state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.bo.get_state()


@app.post("/bo/posterior")
def get_bo_posterior(config: BOQueryConfig):
    """Get GP posterior for a specific zone (falls back to cached snapshot)."""
    global runner, cached_experiment_snapshot
    if runner is not None:
        return runner.bo.get_posterior(zone_id=config.zone_id, n_points=config.n_points)
    if cached_experiment_snapshot:
        post = (cached_experiment_snapshot.get("bo_posteriors") or {}).get(str(config.zone_id))
        if post is not None:
            return post
    raise HTTPException(status_code=400, detail="Experiment not initialized and no cached posterior.")


@app.get("/bo/optimal/{zone_id}")
def get_optimal_incentive(zone_id: int):
    """Get current optimal incentive estimate for a zone."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.bo.get_optimal_incentive(zone_id)


# ---------- Causal ML Endpoints ----------

@app.get("/causal/state")
def get_causal_state():
    """Get causal model state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return {
        "dml": runner.causal_dml.get_state(),
        "causal_forest": runner.causal_forest.get_state(),
    }


@app.get("/causal/hte_segments")
def get_hte_segments():
    """Get heterogeneous treatment effect estimates by segment."""
    global causal_segments
    if causal_segments is None:
        raise HTTPException(status_code=400, detail="Causal model not fitted. Run /experiment/run first.")
    return causal_segments


@app.get("/causal/dose_response")
def get_dose_response():
    """Get dose-response curve (incentive amount → expected response)."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.dose_response.get_curve()


@app.get("/causal/interference")
def get_interference():
    """Get interference-aware treatment effect estimates (direct + spillover)."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.interference_est.get_state()


# ---------- Agent Endpoints ----------

@app.post("/agent/chat")
async def agent_chat(msg: ChatMessage):
    """Chat with the LLM experiment management agent."""
    global agent
    if agent is None:
        agent = ExperimentAgent()
        if runner:
            agent.set_data_sources({
                "vanilla_ts": runner.vanilla_ts,
                "contextual_ts": runner.contextual_ts,
                "net_ts": runner.net_ts,
                "bo": runner.bo,
                "simulator": runner.simulator,
                "causal_segments": causal_segments,
                "change_detector": runner.change_detector,
                "reward_worker": runner.reward_worker,
                "allocator": runner.allocator,
            })
    response = await agent.chat(msg.message)
    return response


@app.get("/agent/state")
def get_agent_state():
    """Get agent state including proposed changes and flags."""
    global agent
    if agent is None:
        return {"status": "not_initialized"}
    return agent.get_state()


# ---------- Simulation Endpoints ----------

@app.get("/simulation/snapshot")
def get_simulation_snapshot():
    """Get current marketplace snapshot (falls back to cached snapshot)."""
    global runner, cached_experiment_snapshot
    if runner is not None:
        return runner.simulator.get_snapshot()
    if cached_experiment_snapshot and cached_experiment_snapshot.get("sim_snapshot"):
        return cached_experiment_snapshot["sim_snapshot"]
    raise HTTPException(status_code=400, detail="Experiment not initialized and no cached snapshot.")


@app.get("/simulation/zones")
def get_zones():
    """Get zone configurations."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return [
        {
            "zone_id": z.zone_id,
            "name": z.name,
            "cluster_id": z.cluster_id,
            "base_demand": z.base_demand,
            "base_supply": z.base_supply,
            "elasticity": z.elasticity,
            "peak_hours": z.peak_hours,
            "peak_multiplier": z.peak_multiplier,
        }
        for z in runner.simulator.zones
    ]


@app.get("/simulation/adjacency")
def get_adjacency():
    """Get zone adjacency matrix (falls back to cached snapshot)."""
    global runner, cached_experiment_snapshot
    if runner is not None:
        return {
            "adjacency": runner.simulator.get_adjacency_matrix(),
            "clusters": runner.simulator.get_cluster_map(),
        }
    if cached_experiment_snapshot and cached_experiment_snapshot.get("adjacency"):
        return cached_experiment_snapshot["adjacency"]
    raise HTTPException(status_code=400, detail="Experiment not initialized and no cached adjacency.")


# ---------- v2 Endpoints: Change Detection, Reward Worker, Allocator ----------

@app.get("/change_detection/state")
def get_change_detection_state():
    """Get non-stationarity detection state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.change_detector.get_state()


@app.get("/reward_worker/state")
def get_reward_worker_state():
    """Get batch reward worker state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.reward_worker.get_state()


@app.get("/allocator/state")
def get_allocator_state():
    """Get constraint-aware allocator state."""
    global runner
    if runner is None:
        raise HTTPException(status_code=400, detail="Experiment not initialized.")
    return runner.allocator.get_state()


# ---------- Swarm Simulation Endpoints ----------

@app.post("/swarm/run")
async def run_swarm(config: SwarmConfig):
    """Run a multi-agent Dasher swarm simulation."""
    global swarm_runner, runner
    swarm_runner = DasherSwarmRunner(
        n_zones=config.n_zones,
        n_dashers=config.n_dashers,
        seed=config.seed,
        model=config.model,
    )
    if config.connect_bandit and runner is not None:
        swarm_runner.connect_bandit(runner.net_ts, runner.allocator)

    results = await swarm_runner.run(n_days=config.n_days)
    # Return summary without full decision logs to keep response small
    response = {
        "n_days": results["n_days"],
        "n_dashers": results["n_dashers"],
        "n_zones": results["n_zones"],
        "total_steps": results["total_steps"],
        "aggregate": results["aggregate"],
        "daily_summaries": results["daily_summaries"],
        "final_zones": results["final_zones"],
    }
    global cached_swarm_snapshot
    cached_swarm_snapshot = save_swarm_result(response)
    return response


@app.get("/swarm/state")
def get_swarm_state():
    """Get current swarm simulation state."""
    global swarm_runner
    if swarm_runner is None:
        raise HTTPException(status_code=400, detail="Swarm not initialized. Run /swarm/run first.")
    return swarm_runner.get_state()


@app.get("/swarm/agents")
def get_swarm_agents():
    """Get state of all Dasher agents in the swarm."""
    global swarm_runner
    if swarm_runner is None:
        raise HTTPException(status_code=400, detail="Swarm not initialized.")
    return swarm_runner.env.get_agent_states()


@app.get("/swarm/zones")
def get_swarm_zones():
    """Get zone states from the swarm environment."""
    global swarm_runner
    if swarm_runner is None:
        raise HTTPException(status_code=400, detail="Swarm not initialized.")
    return [z.to_dict() for z in swarm_runner.env.zones]


@app.post("/swarm/step")
async def swarm_step():
    """Run a single swarm step (for interactive mode)."""
    global swarm_runner
    if swarm_runner is None:
        raise HTTPException(status_code=400, detail="Swarm not initialized. Run /swarm/run first.")
    incentives = swarm_runner._bandit_propose_incentives(swarm_runner.env.step_count)
    swarm_runner.env.set_incentives(incentives)
    result = await swarm_runner.env.step()
    return {
        k: v for k, v in result.items() if k != "decisions"
    }


# ---------- OASIS Sidecar Endpoints ----------

@app.get("/oasis/health")
async def oasis_health():
    """Check if OASIS Docker sidecar is available."""
    available = await oasis_bridge.is_available()
    return {"available": available, "url": oasis_bridge.oasis_url}


@app.post("/oasis/session/create")
async def oasis_create_session(n_agents: int = 10):
    """Create an OASIS simulation session."""
    available = await oasis_bridge.is_available()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="OASIS sidecar not available. Run: docker-compose up oasis-sidecar",
        )
    profiles = []
    session_id = await oasis_bridge.create_session(n_agents, profiles)
    return {"session_id": session_id}


@app.post("/oasis/session/{session_id}/step")
async def oasis_step(session_id: str, incentives: Dict[int, float] = {}):
    """Run one OASIS simulation step."""
    result = await oasis_bridge.step(incentives)
    return result


# ---------- MiroFish Endpoints ----------

@app.get("/mirofish/health")
async def mirofish_health():
    """Check if MiroFish is available."""
    available = await mirofish_bridge.is_available()
    return mirofish_bridge.get_status()


@app.get("/mirofish/profiles")
def generate_mirofish_profiles(n_agents: int = 10):
    """Generate Dasher profiles in MiroFish-compatible format."""
    return mirofish_bridge.generate_dasher_profiles(n_agents)


@app.post("/mirofish/scenario")
async def create_mirofish_scenario(
    name: str = "Dasher Incentive Simulation",
    n_agents: int = 10,
    n_rounds: int = 10,
):
    """Create a MiroFish scenario with Dasher agents."""
    available = await mirofish_bridge.is_available()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="MiroFish not available. Run: docker-compose up mirofish",
        )
    profiles = mirofish_bridge.generate_dasher_profiles(n_agents)
    result = await mirofish_bridge.create_scenario(
        name=name,
        description=f"Swarm of {n_agents} Dasher agents simulating incentive responses",
        agent_profiles=profiles,
        n_rounds=n_rounds,
    )
    return result


@app.post("/mirofish/predict")
async def mirofish_predict(
    scenario_id: str,
    question: str,
    context: Dict[str, Any] = {},
):
    """Run a swarm prediction using MiroFish agents."""
    available = await mirofish_bridge.is_available()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="MiroFish not available. Run: docker-compose up mirofish",
        )
    return await mirofish_bridge.run_prediction(scenario_id, question, context)


# ---------- Combined Status ----------

@app.get("/v3/status")
async def v3_status():
    """Get status of all v3 swarm components."""
    oasis_up = await oasis_bridge.is_available()
    mirofish_up = await mirofish_bridge.is_available()
    return {
        "swarm_engine": {
            "status": "ready" if swarm_runner else "not_initialized",
            "type": "built-in",
        },
        "oasis_sidecar": {
            "status": "connected" if oasis_up else "unavailable",
            "url": oasis_bridge.oasis_url,
            "note": "Run: docker-compose up oasis-sidecar",
        },
        "mirofish": {
            "status": "connected" if mirofish_up else "unavailable",
            "url": mirofish_bridge.base_url,
            "note": "Run: docker-compose up mirofish",
        },
    }


# ---------- Cache Endpoints ----------

@app.get("/cache/status")
def get_cache_status():
    """Metadata about what cached results are available on disk."""
    return cache_status()


@app.get("/cache/experiment")
def get_cached_experiment():
    """Return the full cached experiment snapshot (results + sim + adjacency + posteriors)."""
    global cached_experiment_snapshot
    if cached_experiment_snapshot is None:
        raise HTTPException(status_code=404, detail="No cached experiment snapshot.")
    return cached_experiment_snapshot


@app.get("/cache/swarm")
def get_cached_swarm():
    """Return the last cached swarm result."""
    global cached_swarm_snapshot
    if cached_swarm_snapshot is None:
        raise HTTPException(status_code=404, detail="No cached swarm result.")
    return cached_swarm_snapshot


@app.post("/cache/reload")
def reload_cache():
    """Re-read both cache files from disk into memory."""
    global cached_experiment_snapshot, cached_swarm_snapshot, experiment_results, causal_segments
    cached_experiment_snapshot = load_experiment_snapshot()
    cached_swarm_snapshot = load_swarm_result()
    if cached_experiment_snapshot and cached_experiment_snapshot.get("experiment_results"):
        experiment_results = cached_experiment_snapshot["experiment_results"]
        causal_segments = experiment_results.get("causal", {}).get("hte_segments")
    return cache_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
