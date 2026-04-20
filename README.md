# DorkDash — Agentic Causal Bandit Platform for Dasher Incentive Optimization

**v3.0** — A production-realistic full-stack platform demonstrating how to upgrade DoorDash's Multi-Armed Bandit experimentation infrastructure with **Networked Contextual Bandits**, **Shape-Constrained Bayesian Optimization**, **Interference-Aware Causal ML**, **Explicit Change-Point Detection**, a **Constraint-Aware Allocator**, a **Batch Reward Worker**, an **LLM meta-agent** with structured tool-calling, and a **Multi-Agent Swarm Simulation** layer backed by OASIS and MiroFish.

> Built as a direct extension of the DoorDash engineering team's published work:
> - [Arjun Sharma, "Using a MAB with Thompson Sampling to Identify Responsive Dashers" (2022)](https://careersatdoordash.com/blog/using-a-multi-armed-bandit-with-thompson-sampling-to-identify-responsive-dashers/)
> - [Weinstein & Huang, "Accelerating Experimentation with a MAB Platform" (Dec 2025)](https://careersatdoordash.com/blog/experimentation-at-doordash-with-a-multi-armed-bandit-platform/)
> - [Xu et al., "Causal ML for Promotions" (KDD 2025)](https://causal-machine-learning.github.io/kdd2025-workshop/papers/16.pdf)

---

## What's New

### v3 — Multi-Agent Swarm Simulation

| Component | v2 | v3 |
|-----------|----|----|
| Simulated Dashers | Parametric logistic response curves | **LLM-powered autonomous agents** with persona, memory, and fatigue dynamics |
| Simulation engine | Single parametric model | **Three pluggable engines**: built-in lightweight, OASIS (camel-ai, 1M-agent scale), MiroFish (Flask+Zep memory) |
| Deployment | Python backend only | **Docker Compose stack**: backend + OASIS sidecar (port 5050) + MiroFish (port 5001) |
| Frontend | 7 tabs | **8 tabs**: adds Swarm tab with acceptance-trend, zone performance, and daily-summary visualizations |
| Counterfactuals | Re-run parametric model | Run "what-if" policies against a living LLM agent population |

### v2 — Production-Grade Bandit Platform

| Component | v1 | v2 |
|-----------|----|----|
| Simulator | Static zones, immediate feedback | Zone interference/spillover, delayed feedback, cluster structure |
| Bandits | Vanilla TS + Contextual LinTS | + **Networked LinTS** (global + local posterior decomposition) |
| Bayesian Opt | Standard GP | **Shape-constrained GP** (monotonicity + concavity enforced) |
| Non-stationarity | Heuristic weight decay | **CUSUM + sliding window + posterior drift** change-point detection |
| Reward estimation | Immediate observed reward | **Batch reward worker**: delayed feedback, doubly-robust AIPW, treatment-effect-vs-control |
| Causal ML | DML + Causal Forest | + **Dose-response curves** (Nadaraya-Watson) + **interference-aware** decomposition (direct vs spillover) |
| Allocation | Unconstrained | **DISCO-style constrained allocator**: daily/weekly budgets, per-arm caps, exploration floor, sticky cohorts |
| Agent | 5 read tools | **12 tools** covering all v2 components, approval workflow for all write actions |
| Frontend | 5 tabs | **7 tabs**: adds Network (zone graph + spillover) and System (change detection + reward worker + allocator) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Next.js Frontend (v3)                         │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ ┌──┐  │
│  │Overview│ │Bandits │ │Network │ │ BayOpt │ │ Causal │ │System │ │Sw│  │
│  │  KPIs  │ │ Regret │ │ Graph  │ │GP Post │ │Dose-R  │ │ChgDet │ │ar│  │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └───────┘ └──┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                          FastAPI Backend (v3)                           │
│  ┌────────────┐ ┌───────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │ Simulator  │ │  Bandits      │ │  Bayesian    │ │  Causal ML   │     │
│  │ spillover  │ │  VTS / LinTS  │ │  Optimizer   │ │  DML + DR    │     │
│  │ delayed fb │ │  NetLinTS     │ │  Shape-GP    │ │  Dose-Resp   │     │
│  └────────────┘ └───────────────┘ └──────────────┘ └──────────────┘     │
│  ┌────────────┐ ┌───────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │  Change    │ │  Batch Reward │ │  Constrained │ │  LLM Agent   │     │
│  │  Detector  │ │  Worker       │ │  Allocator   │ │  (GPT-4o)    │     │
│  │ CUSUM/PELT │ │  AIPW / DR    │ │  DISCO-style │ │  12 tools    │     │
│  └────────────┘ └───────────────┘ └──────────────┘ └──────────────┘     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Swarm Runner — orchestrates LLM Dasher agents + v2 bandits      │   │
│  │  DasherAgent · MarketplaceEnvironment · OASISBridge · MiroFishBridge │
│  └──────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                         Docker Sidecars (v3)                            │
│  ┌───────────────────────────────┐  ┌──────────────────────────────────┐│
│  │  OASIS sidecar  :5050         │  │  MiroFish (prebuilt)  :5001/3001 ││
│  │  camel-oasis · FastAPI        │  │  Flask + Zep memory + Vue UI     ││
│  │  up to 1M agents              │  │  Social swarm intelligence       ││
│  └───────────────────────────────┘  └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Marketplace Simulator
- 10 Bay Area geographic zones with realistic supply/demand dynamics
- Zone adjacency matrix with **spillover effects** — incentivising zone A pulls supply from neighbours
- **Cluster structure** (3 clusters) reflecting geographic proximity
- **Delayed feedback**: outcomes resolve after a configurable feedback delay
- Dasher incentive-response as logistic function of incentive × zone tightness × time × tenure × spillover pressure
- Ground-truth regime changes logged for change-detection validation

### 2. Vanilla Thompson Sampling (Baseline)
- Replicates DoorDash's current system from Arjun Sharma's 2022 blog
- Beta-Bernoulli TS with weight decay for non-stationarity

### 3. Contextual LinTS
- Linear Thompson Sampling with Bayesian linear regression posterior
- Context features: supply deficit, time encoding, peak indicator, zone elasticity, Dasher tenure, spillover pressure
- Meaningful regret reduction vs vanilla TS in non-stationary environments

### 4. Networked LinTS (v2)
- **Global + local posterior decomposition**: shared global prior across zones + zone-specific deviation
- Zones in the same cluster share information via a configurable `sharing_strength` parameter
- Per-zone regret tracking alongside global cumulative regret
- Mirrors "Decentralized Contextual Bandits with Network Adaptivity" (arXiv 2508.13411)

### 5. Shape-Constrained Bayesian Optimization (v2)
- GP surrogate with **monotonicity** (more incentive ≥ more response) and **concavity** (diminishing returns) enforced
- RBF feature representation for continuous actions
- Expected Improvement acquisition with shape-aware posterior clipping
- BoTorch integration with pure-NumPy fallback

### 6. Explicit Change-Point Detection (v2)
- **CUSUM** (cumulative sum control chart) per zone
- **Sliding window** likelihood ratio test
- **Posterior drift** monitoring via KL divergence on arm posteriors
- Replaces heuristic weight decay — detects regime shifts with timestamps and severities

### 7. Batch Reward Worker (v2)
- Processes outcomes in daily batches (mirrors DoorDash's production batch pipeline)
- Computes **treatment effect relative to control arm** per batch
- **Doubly-robust AIPW** estimation using logged propensity scores
- Produces posterior update signals for bandits

### 8. Constraint-Aware Allocator (v2)
- **Daily and weekly budget caps** with per-zone spend tracking
- **Per-arm frequency caps** — maximum incentive sends per Dasher per week
- **Exploration floor** — guarantees minimum pulls per arm for statistical validity
- **Sticky cohort** logic — Dashers assigned to a treatment stay in it
- Mirrors DISCO (arXiv 2406.06433) combining bandit selection with integer programme constraints

### 9. Causal ML — Dose-Response + Interference (v2)
- **Dose-response curves**: Nadaraya-Watson kernel regression of E[response | incentive = t] with 95% CI
- **Interference-aware estimation**: decomposes treatment effects into direct (own-zone) and indirect (spillover) components using augmented OLS
- Per-zone interference breakdown
- DML + Causal Forest for HTE by Dasher segment (tenure, zone deficit, peak/off-peak, responsiveness)

### 10. LLM Meta-Agent (v2)
- GPT-4o with **12 structured tool calls** covering all v2 components
- New tools: `get_networked_bandit_state`, `get_change_detection_state`, `get_reward_worker_state`, `get_dose_response`, `get_interference_effects`, `get_allocator_state`
- **Approval workflow**: all write actions (arm changes, prior resets) are proposals requiring human confirmation
- Fully functional in mock mode without an OpenAI key

### 11. Multi-Agent Swarm Simulation (v3)
- **`DasherAgent`** — LLM-powered autonomous agent with immutable persona (tenure, home zone, elasticity, earnings goal, fatigue threshold) and episodic memory
- **`MarketplaceEnvironment`** — zone-level simulation with demand fluctuations, incentive offers, acceptance tracking, and zone switching
- **`DasherSwarmRunner`** — orchestrates multi-day, multi-step swarm experiments and wires agent decisions to the v2 bandit + allocator
- **`OASISBridge`** — async REST client to an OASIS Docker sidecar for large-scale simulation (up to 1M agents)
- **`MiroFishBridge`** — REST client to a MiroFish container that ships with Zep-backed memory and social-dynamics primitives
- Rule-based fallback decision path when no OpenAI key is configured, so the swarm works in CI and in offline demos
- 13 new FastAPI endpoints under `/swarm/*`, `/oasis/*`, `/mirofish/*`, plus a combined `/v3/status` health check

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt

# Optional for LLM agent + LLM Dashers — everything works in mock/fallback without it
cat > .env <<'EOF'
OPENAI_API_KEY=sk-your-key
ZEP_API_KEY=your-zep-key    # optional, required only for MiroFish sidecar
EOF

# From repo root
python -m uvicorn backend.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Swarm Sidecars (optional, for v3)

```bash
# Start OASIS sidecar (:5050, camel-oasis) + MiroFish (:5001 backend, :3001 UI)
docker-compose up -d

# Verify all three engines are connected
curl http://localhost:8000/v3/status
```

Open `http://localhost:3000` and click **Run Experiment** (bandits tab) or open the **Swarm** tab to run a live LLM-driven agent simulation.

## Key Results

| Metric | Vanilla TS | Contextual LinTS | Networked LinTS |
|--------|-----------|------------------|-----------------|
| Cumulative Regret (14 days) | Baseline | ~40–60% lower | Lowest (shared zone info) |
| Continuous Optimum | ✗ | ✗ | ✗ (BO finds it) |
| Spillover Correction | ✗ | ✗ | ✓ (network prior) |
| Change-Point Response | Slow decay | Slow decay | CUSUM reset |
| Treatment Effect | None | None | Interference-adjusted |

Empirical from 3-day test run:
- **Dose-response optimal dose**: $5.00
- **Interference**: direct effect +0.081 / indirect (spillover) −0.092 → **114% spillover ratio**
- **Change detector**: events logged with zone, method, and severity

## Frontend Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | KPIs, regret reduction, batch summary table, change-point count |
| **Bandits** | Arm posteriors, regret chart (all 3 algorithms), NetLinTS zone stats |
| **Network** | Zone adjacency graph, cluster colouring, spillover heatmap |
| **Bayesian Opt** | GP posterior per zone, observation history, optimal incentive |
| **Causal** | HTE by segment, dose-response curve with CI, interference decomposition |
| **System** | Change detection alerts, reward worker state, allocator budget utilisation |
| **Agent** | GPT-4o chat with full tool access to all v2 components |
| **Swarm** | LLM Dasher agents: acceptance-rate trend, zone performance table, daily summaries, re-run controls |

## Papers Referenced

| Paper | Relevance |
|-------|-----------|
| Sharma, "Using MAB with TS to Identify Responsive Dashers" (2022) | **Interviewer authored** — DoorDash baseline |
| Weinstein & Huang, "Accelerating Experimentation with MAB" (Dec 2025) | Team roadmap — contextual bandits + BO |
| Xu et al., "Causal ML for Promotions" (KDD 2025) | DoorDash's own causal ML framework |
| Bouneffouf & Feraud, "Bandits, LLMs, and Agentic AI" (AAAI 2026) | Foundation for LLM meta-agent |
| "Agent A/B" (arXiv 2504.09723) | LLM agents for experimentation |
| "Decentralized Contextual Bandits with Network Adaptivity" (arXiv 2508.13411) | NetLinTS global+local decomposition |
| DISCO (arXiv 2406.06433) | Constrained allocator: bandit + integer programming |
| Chernozhukov et al. (2018) | Double/Debiased ML |
| Hirano & Imbens (2004) | GPS for continuous treatments (dose-response) |
| Hudgens & Halloran (2008) | Causal inference under interference (spillover) |

## v3: Multi-Agent Swarm Simulation (Shipped)

The v2 simulator uses parametric equations (logistic response curves, adjacency-weighted spillover). v3 augments this with a **multi-agent swarm** where each simulated Dasher is an autonomous LLM-powered agent with its own persona, memory, and decision-making — producing emergent marketplace dynamics that a single equation cannot capture.

### Why This Helps

| Limitation of v2 Simulator | What Swarm Agents Add |
|---|---|
| Response curves are hand-tuned logistic functions | Each agent develops its own response pattern from persona + history |
| Spillover is a fixed adjacency weight | Agents actively compete for zones, creating emergent spillover |
| Non-stationarity is injected via scheduled regime changes | Agents naturally drift as they accumulate experience and fatigue |
| No agent-to-agent interaction | Dashers can observe and react to each other (herding, coordination) |
| Counterfactuals require re-running the parametric model | Run "what-if" policy changes against a living agent population |

### Architecture (Proposed)

```
┌──────────────────────────────────────────────────┐
│              Swarm Simulation Layer              │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Dasher   │ │ Dasher   │ │  ... × 1000s     │  │
│  │ Agent 1  │ │ Agent 2  │ │  LLM-powered     │  │
│  │ (persona,│ │ (persona,│ │  autonomous      │  │
│  │  memory) │ │  memory) │ │  decision-making │  │
│  └──────────┘ └──────────┘ └──────────────────┘  │
│          ↕ accept/reject incentives ↕            │
├──────────────────────────────────────────────────┤
│         Existing v2 Bandit Platform              │
│  Bandits → Allocator → Causal ML → Agent         │
└──────────────────────────────────────────────────┘
```

Each Dasher agent would have:
- **Persona**: tenure, home zone, elasticity profile, schedule preferences
- **Memory**: past incentive offers, acceptance history, earnings trajectory
- **Decision logic**: LLM-driven accept/reject reasoning conditioned on persona + current state
- **Social awareness**: observe neighbour-zone incentive levels, herd toward high-demand zones

The bandit platform then optimises incentives against this living population instead of (or alongside) the parametric simulator, enabling:
- **Synthetic A/B tests** before real deployment
- **Counterfactual policy evaluation** ("what if we cap incentives at $3 in zone 5?")
- **Emergent behavior discovery** (incentive fatigue, zone herding, strategic waiting)
- **Stress testing** (inject competitor promotions, weather events, holiday surges)

### Open-Source Frameworks

| Repository | Stars | What It Does | How It Fits |
|---|---|---|---|
| [**MiroFish**](https://github.com/666ghj/MiroFish) | 54.7k | Universal swarm intelligence engine — LLM agents with personas, memory, social interactions. Powered by OASIS. | **Primary candidate.** Give each agent a Dasher persona; run marketplace scenarios; observe emergent incentive-response behavior at scale. |
| [**OASIS**](https://github.com/camel-ai/oasis) | 4.2k | Open Agent Social Interaction Simulations — scales to 1M agents. Core engine behind MiroFish. | Underlying simulation engine. Handles agent orchestration, message passing, environment stepping. Apache 2.0. |
| [**Swarms**](https://github.com/kyegomez/swarms) | — | Enterprise-grade multi-agent orchestration with concurrent execution and tool use. | Alternative agent framework if MiroFish/OASIS is too opinionated. More flexible but less turnkey for social simulation. |
| [**VMAS**](https://github.com/proroklab/VectorizedMultiAgentSimulator) | — | Vectorized differentiable multi-agent simulator for MARL benchmarking. PyTorch-native, GPU-accelerated. | For RL-trained Dasher policies instead of LLM-driven ones. Much faster (vectorized) but less interpretable. |
| [**Ride-sharing-Simulator**](https://github.com/HKU-Smart-Mobility-Lab/Ride-sharing-Simulator) | — | High-capacity ride-sharing sim calibrated on real request datasets and road networks. | Domain-specific reference for marketplace supply/demand dynamics. Could provide calibration data. |
| [**RL+LLM Urban Simulations**](https://github.com/lukehollis/rl-llm-urban-simulations) | — | Hybrid RL+LLM approach for city-scale agent behaviors in game engine environments. | Demonstrates the hybrid approach: RL for fast policy learning, LLM for interpretable decision reasoning. |

### Integration (delivered)

1. ✅ **Phase 1**: Built-in swarm engine (`backend/swarm/`) with `DasherAgent`, `MarketplaceEnvironment`, `DasherSwarmRunner` — runs 10–1000 LLM Dashers natively in Python
2. ✅ **Phase 2**: Swarm connected to the v2 allocator via `DasherSwarmRunner.propose_incentive` / `feed_reward` — bandits propose, LLM agents accept/reject, rewards flow back
3. ✅ **Phase 3**: OASIS Docker sidecar (`oasis-sidecar/`) with `camel-oasis`, FastAPI REST bridge, 10 sample Dasher profiles
4. ✅ **Phase 4**: MiroFish integration via the prebuilt `ghcr.io/666ghj/mirofish:latest` image with Zep memory + Vue UI
5. ✅ **Phase 5**: Frontend Swarm tab with live visualization (acceptance-trend bar chart, zone performance table, daily summaries)
6. 🚧 **Phase 6 (next)**: Use swarm-generated data to re-fit causal ML models and compare to parametric ground truth

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, NumPy, SciPy, scikit-learn
- **ML**: econml (DML + Causal Forest), BoTorch/GPyTorch (Bayesian Optimization)
- **LLM**: OpenAI GPT-4o with function calling (mock mode available)
- **Swarm**: built-in engine + `camel-oasis` (OASIS sidecar) + MiroFish (prebuilt image)
- **Frontend**: Next.js 16, React, TailwindCSS, Recharts, shadcn/ui, Lucide icons
- **Infra**: Docker Compose for the OASIS + MiroFish sidecars, aiohttp/httpx async bridges
