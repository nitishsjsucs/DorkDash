# DorkDash — Master Technical Deep-Dive

## Master description

DorkDash is a production-realistic full-stack experimentation platform that upgrades DoorDash-style Multi-Armed Bandit infrastructure from a single-algorithm prototype into a **networked, causal, constraint-aware, LLM-orchestrated** system, and then further into a **multi-agent swarm simulation** layer where every simulated Dasher is an autonomous LLM agent. The system is built directly on top of three pieces of DoorDash engineering-team prior art (Sharma 2022, Weinstein & Huang 2025, Xu et al. KDD 2025) and extends them with ideas from Decentralized Contextual Bandits (arXiv 2508.13411), DISCO (arXiv 2406.06433), Double/Debiased ML (Chernozhukov 2018), Generalized Propensity Scores (Hirano & Imbens 2004), Partial Interference (Hudgens & Halloran 2008), and OASIS/MiroFish-style social-agent swarms (camel-ai, arXiv 2504.09723).

The codebase is approximately **7,200 lines** across Python and TypeScript — ~4,850 lines of Python backend (FastAPI + NumPy/SciPy + scikit-learn + econml + BoTorch), ~2,350 lines of TypeScript frontend (Next.js 16 + React + Tailwind + Recharts), plus Docker Compose orchestration, an OASIS Python-3.11 sidecar, and a prebuilt MiroFish swarm-intelligence container. The platform runs three decoupled LLM-agent loops simultaneously: a **meta-agent** (GPT-4o with 12 structured tools for experiment management), a **swarm of autonomous Dasher agents** (one LLM call per accept/reject decision, or a rule-based fallback), and a **policy learner** (networked Thompson Sampling + shape-constrained Bayesian Optimization) — all wired together through an event-driven reward pipeline with doubly-robust offline-policy correction.

---

## 1) Problem framing

### 1.1 The business problem

DoorDash's supply team runs continuous experiments that decide, for each Dasher in each zone at each hour, whether to send an incentive offer and how large it should be. The production Multi-Armed Bandit from Sharma's 2022 blog solves a small version of that problem with Beta-Bernoulli Thompson Sampling. The 2025 Weinstein & Huang platform paper signals where the team wants to go: contextual bandits, Bayesian Optimization over continuous incentive dollars, real causal modelling, and automation of routine experiment-management decisions.

DorkDash is the end-to-end reference implementation of that trajectory, extended further into agent-based counterfactual simulation. The problems it solves:

1. **Non-stationarity**: demand regimes shift (weather, holidays, competitor promos) and the bandit has to respond in days, not weeks.
2. **Interference**: incentivising zone A pulls supply away from adjacent zone B, so naive per-zone A/B tests over-estimate treatment effects.
3. **Delayed feedback**: a Dasher accepts an offer now but the marketplace outcome (delivery completion, earnings, future retention) resolves hours later.
4. **Budget constraints**: incentive spend is capped daily and weekly, per zone and per Dasher; a bandit that ignores the budget is not deployable.
5. **Offline counterfactuals**: "what if we cap incentives at $3 in zone 5?" cannot be answered with historical logs alone — we need a living simulation.
6. **Operator cognitive load**: a PM reading six dashboards cannot reason about six independent models; they need an LLM meta-agent that summarizes, proposes, and requires human approval before any write.

### 1.2 Why bandits, not A/B tests

Classical A/B tests burn statistical power on losing arms. With dozens of incentive levels × 10 zones × hourly granularity, the arm space is large enough that pure exploration is wasteful. Bandits trade exploration for exploitation adaptively and converge to near-optimal allocation faster. DoorDash's own 2022 and 2025 blogs make this argument — DorkDash operationalizes it.

### 1.3 Why layered algorithms, not one algorithm

No single bandit solves all five problems. Networked LinTS handles interference via cross-zone information sharing. Shape-constrained BO handles the continuous-incentive problem that a discrete-arm bandit cannot. The causal layer handles interference at the estimation level (dose-response + direct/indirect decomposition). Change detection handles non-stationarity explicitly instead of hiding it in a decay factor. The allocator handles budgets. The reward worker handles delayed feedback and doubly-robust correction. Each layer is separately testable, separately swappable, and separately visualizable in the frontend.

### 1.4 Why add an LLM swarm

Parametric simulators are only as good as their equations. A logistic incentive-response curve cannot reproduce Dasher fatigue, herding, or strategic waiting. v3 adds an LLM-driven swarm so the bandit can be evaluated against a living population whose decisions come from persona + memory + prompt rather than a hand-tuned curve. This is the same architectural move as OASIS (1M-agent social simulations) and Agent A/B (arXiv 2504.09723), adapted to marketplace incentives.

---

## 2) Marketplace Simulator

### 2.1 Zone structure

10 Bay Area zones (`SF_Downtown`, `SF_Mission`, `SF_Sunset`, `Oakland_DT`, `Oakland_Hills`, `Berkeley`, `San_Jose_DT`, `San_Jose_South`, `Palo_Alto`, `Mountain_View`) organized into 3 clusters. Each zone has:

- **Demand trajectory**: sinusoidal base curve + Gaussian noise + injectable regime shifts
- **Supply deficit**: demand minus active Dashers, normalized to [-1, 1]
- **Elasticity**: baseline responsiveness to incentives
- **Peak indicator**: is the current hour within the zone's peak window
- **Adjacency**: pairs of zones in the same cluster are adjacent; adjacency drives spillover

### 2.2 Interference / spillover

When zone A gets a high incentive, some Dashers physically relocate from zone B to zone A. We model this as **spillover pressure**: for each zone, spillover = weighted sum of (incentive - baseline) across adjacent zones. Spillover enters the acceptance model negatively — a Dasher in zone B is less likely to accept a low offer if zone A next door has a high one.

This is the key departure from the naive DoorDash-style per-zone MAB. Without modelling spillover, the per-zone treatment effect estimate is biased upward because the gain in the treated zone partially comes from the untreated zone's supply.

### 2.3 Delayed feedback

Outcomes do not resolve in the same simulation step. Each `send_incentive(...)` call returns an `offer_id` and pushes the resulting (action, context, propensity, resolution_step) tuple into a pending-outcomes queue. The reward worker drains this queue on a batch schedule and writes observed rewards back to the bandit. This mirrors DoorDash's production batch pipeline where incentives are sent in real time but rewards arrive via a nightly job.

### 2.4 Logged propensity

Every incentive send logs the **policy probability** (the probability the policy assigned to the chosen arm). This is required for Inverse-Propensity-Score (IPS) and Augmented IPS (AIPW) reward estimation in the reward worker. Without logged propensities, offline-policy evaluation is not identifiable.

### 2.5 Ground-truth regime changes

The simulator exposes `inject_change_point(zone_id, day, shift_magnitude, type)` so change-detection can be validated. Every injected change is also logged so the frontend can overlay the true change points on the change-detector output for visual ground-truth comparison.

### 2.6 File layout

| File | Responsibility |
|---|---|
| `backend/simulation/environment.py` | `MarketplaceSimulator` — zones, demand, supply, spillover, propensity logging, delayed outcome queue, change-point injection |
| `backend/simulation/experiment_runner.py` | Orchestrates a multi-day experiment across all bandit algorithms + BO + causal + change-det + reward worker + allocator |
| `backend/simulation/reward_worker.py` | Batch outcome resolution, treatment-effect-vs-control estimation, AIPW correction |

---

## 3) Bandits

### 3.1 Vanilla Thompson Sampling (baseline)

Direct replication of Sharma's 2022 blog: Beta-Bernoulli posterior per discrete arm with a weight-decay factor for non-stationarity. Parameters: `(alpha, beta)` per arm, decayed each day by `decay_rate` to prevent the posterior from becoming too confident in stale regimes.

### 3.2 Contextual LinTS

Linear Thompson Sampling with a Bayesian linear-regression posterior. On each arm, reward is modeled as `y = x^T θ + ε` with `θ ~ N(μ, Σ)`. The posterior is updated via the standard Bayesian linear regression recurrence. Arm selection samples `θ̃ ~ N(μ, Σ)` and picks the arm maximizing `x^T θ̃`.

Context features (`build_context_vector` in `backend/bandits/networked_bandit.py`):

- `supply_deficit`
- `time_of_day` (sinusoidal encoding)
- `is_peak`
- `effective_elasticity`
- `tenure_bucket`
- `spillover_pressure`

### 3.3 Networked LinTS — global + local decomposition

This is the core novelty in the bandit layer. Inspired by arXiv 2508.13411, each zone's posterior decomposes into:

```
θ_z = θ_global + δ_z
θ_global ~ shared prior across all zones
δ_z ~ zone-local deviation
```

At update time, each observation `(x, y, z)` contributes to both the global posterior and the zone-local posterior, weighted by a `sharing_strength` hyperparameter. Zones in the same cluster share more information than zones across clusters.

**Why this matters for DoorDash**: without information sharing, a new zone with 3 observations has a useless posterior. With networked sharing, the new zone inherits the cluster-level global prior and converges faster. This is empirically the lowest-regret algorithm in our runs.

The `select_arm` method returns `(arm_id, selection_probability_estimate)` — the probability the bandit would have chosen that arm given the sampled theta. This probability is logged as the propensity for downstream doubly-robust correction in the reward worker.

### 3.4 Per-zone and global regret tracking

`cumulative_regret` and `regret_history` are maintained per-zone and aggregated globally. The frontend Bandits tab renders all three algorithms on the same regret chart so the lift from networked information sharing is visible at a glance.

### 3.5 File layout

| File | Responsibility |
|---|---|
| `backend/bandits/vanilla_ts.py` | Beta-Bernoulli TS with decay |
| `backend/bandits/contextual_ts.py` | Linear TS with Bayesian posterior |
| `backend/bandits/networked_bandit.py` | Networked LinTS (`NetLinTS`) — global + local decomposition, propensity-reporting `select_arm`, per-zone regret tracking |

---

## 4) Shape-Constrained Bayesian Optimization

### 4.1 Why shape constraints

A Gaussian Process over `incentive → response` fit on sparse data will happily produce a non-monotone posterior: "$3 gets 40% acceptance, $5 gets 35%, $7 gets 50%". That is obviously wrong. Shape constraints encode the prior that response is **monotone non-decreasing** and **concave** (diminishing returns) in incentive dollars.

### 4.2 Implementation

`backend/optimization/bayesian_opt.py` implements a GP surrogate with:

- **RBF feature representation** for the continuous incentive dimension
- **Monotonicity** enforced by clipping the posterior mean at each evaluation grid point so it never decreases with incentive
- **Concavity** enforced via second-difference clipping on the grid
- **Expected Improvement** acquisition, post-processed through the shape projection so suggestions always lie on the monotone-concave curve
- **BoTorch integration** as the preferred backend when available, with a **pure-NumPy Gaussian process fallback** (Cholesky + explicit posterior formulas) so the module runs without the heavyweight torch dep

### 4.3 GP posterior API

The `/bo/posterior` endpoint returns `(grid, mean, std, lower, upper)` arrays that the frontend renders as a posterior + confidence band. The `/bo/suggest` endpoint returns the shape-projected argmax of the acquisition function as the next incentive to try.

### 4.4 Why BO alongside bandits

Discrete bandits are wasteful over a continuous action space. BO finds the optimum incentive dollar precisely. Bandits decide *whom* to send to. BO decides *how much* to send. They run concurrently, seeded from each other's observations, and the frontend shows both — the bandit regret chart and the BO posterior.

---

## 5) Causal ML

### 5.1 Three estimators

`backend/causal/treatment_effects.py` implements three estimators of heterogeneous treatment effect (HTE):

1. **Double/Debiased ML (DML)** — Chernozhukov et al. (2018). Uses two nuisance learners (outcome model `g(X)` and propensity model `m(X)`) and regresses the residuals to get an unbiased estimate of CATE. Uses `econml` when available and falls back to a manual cross-fitted DML implementation with scikit-learn random forests when not.
2. **Causal Forest** — via `econml.CausalForestDML` when available; otherwise a scikit-learn random-forest proxy on the DML residuals.
3. **Manual DML** — pure NumPy/sklearn implementation for environments without econml.

### 5.2 Heterogeneous treatment effect by segment

The `estimate_hte_by_segment` method breaks down CATE by four segmentations that matter for DoorDash:

- **Tenure**: new / experienced / veteran Dashers
- **Deficit**: high-supply / balanced / low-supply zones
- **Peak / off-peak** hour
- **Responsiveness**: decile buckets of inferred elasticity

Frontend `HTEChart.tsx` renders these as grouped bars so PMs can identify which Dasher cohorts the incentive actually moves.

### 5.3 Dose-response curves (Hirano & Imbens 2004)

For continuous treatments, `estimate_dose_response` fits a Nadaraya-Watson kernel regression of `E[Y | T=t]` on a grid, with bandwidth chosen by Silverman's rule. Returns `(grid, response, std, lower95, upper95)` which the frontend renders as a curve with confidence band. The argmax of the curve is the empirical optimal dose.

### 5.4 Interference-aware decomposition (Hudgens & Halloran 2008)

When incentives spill over between zones, the per-zone treatment-effect estimate is biased. `estimate_interference_effects` runs an augmented OLS:

```
Y_ij = β_0 + β_direct · T_ij + β_indirect · S_ij + γ · X_ij + ε
```

where `T_ij` is the own-zone treatment and `S_ij` is the sum of treatments in adjacent zones. `β_direct` and `β_indirect` decompose the total effect into the direct and spillover components. A high `spillover_ratio` (|indirect|/|direct|) — empirically ~114% in our 3-day smoke test — is diagnostic that per-zone A/B tests would over-estimate the direct effect by more than 2×.

### 5.5 File layout

| File | Responsibility |
|---|---|
| `backend/causal/treatment_effects.py` | `CausalEstimator` with DML, causal forest, dose-response, interference; `_prepare_data` extracts `(X, T, Y, Z)` from simulation records |

---

## 6) Change-Point Detection

Replaces the heuristic weight-decay from v1 with three explicit detectors:

1. **CUSUM** (cumulative sum control chart) on the per-zone reward stream with two-sided threshold
2. **Sliding-window likelihood ratio test** comparing the last `w` observations to the previous `w`
3. **Posterior drift monitor** that tracks the KL divergence between the current and previous bandit posterior per arm

Each event is logged with zone, detector, severity, metric, description, and a recommended action. The frontend System tab renders recent events and per-zone active alerts. This gives the LLM meta-agent a structured signal it can query via `get_change_detection_state` instead of having to infer non-stationarity from raw reward series.

File: `backend/simulation/change_detection.py` (detectors), state returned through `ExperimentRunner.run_full_experiment` → `change_detection` key.

---

## 7) Batch Reward Worker

`backend/simulation/reward_worker.py` implements DoorDash's production batch pattern:

1. Drain the pending-outcomes queue into a daily batch
2. Estimate control-arm mean reward
3. Estimate per-arm reward, treatment-effect-vs-control, and standard error
4. Apply **Augmented Inverse-Propensity-Score (AIPW)** correction using the logged propensities — this is the doubly-robust estimator that is consistent if either the outcome model or the propensity model is correct
5. Emit a `batch_summary` (batch_id, day, n_outcomes, control_rate, `arm_te_estimates`, `dr_estimates`)

The DR estimates are what the bandit actually trains on — not the raw observed rewards. This is the correct way to reconcile offline-policy corrections with online bandit updating.

---

## 8) Constraint-Aware Allocator (DISCO-style)

`backend/optimization/constrained_allocator.py` implements a small constraint layer sitting between the bandit's arm choice and the real action:

- **Daily budget cap** (global and per-zone)
- **Weekly budget cap**
- **Per-arm frequency cap**: max incentive sends per Dasher per week
- **Exploration floor**: guarantee minimum pulls per arm for statistical validity
- **Sticky cohort logic**: once a Dasher is assigned to arm *a*, they stay in arm *a* for the week — respects DoorDash's real operational constraints about treatment consistency

This is the pattern from DISCO (arXiv 2406.06433): combine a bandit's selection probabilities with integer-programming constraints. When the bandit's preferred arm violates a constraint, the allocator falls back to the best feasible arm, preserving as much of the bandit's selection probability as possible. The `get_state` method returns budget, per-zone spend, and recent allocations for the frontend System tab.

---

## 9) LLM Meta-Agent (GPT-4o with 12 structured tools)

### 9.1 Architecture

`backend/agent/meta_agent.py` implements an `ExperimentAgent` class that drives an OpenAI GPT-4o loop with 12 OpenAI function-calling tools:

**Read tools** (state introspection):
1. `get_bandit_state`
2. `get_networked_bandit_state`
3. `get_bo_posterior`
4. `get_causal_state`
5. `get_hte_segments`
6. `get_change_detection_state`
7. `get_reward_worker_state`
8. `get_dose_response`
9. `get_interference_effects`
10. `get_allocator_state`

**Write tools** (propose only, require approval):
11. `propose_arm_change`
12. `propose_prior_reset`

### 9.2 Approval workflow

Every write tool returns a `Proposal` object with `proposal_id`, `summary`, `diff`, and `rationale` — it does **not** mutate state. The frontend Agent tab renders the proposal with Approve / Reject buttons and a separate `/agent/approve` endpoint performs the actual mutation only after explicit user action. This mirrors DoorDash's production experiment-management requirement that no LLM-generated decision flows to production without a human PM in the loop.

### 9.3 System prompt

The agent's system prompt enumerates its responsibilities — summarize experiments, propose changes, detect non-stationarity, explain causal results in PM-friendly language — and explicitly tells it never to call a write tool without stating the rationale first. The result is an agent that behaves like a junior experimentation data scientist you can ask natural-language questions and get grounded, tool-backed answers.

### 9.4 Mock mode

If `OPENAI_API_KEY` is unset, the agent runs in **mock mode** with deterministic canned responses that still exercise the full tool-calling loop. This is critical for CI and for demo environments where billed API calls are undesirable.

---

## 10) Multi-Agent Swarm Simulation (v3)

### 10.1 Motivation

The v2 simulator's Dasher response is a logistic function. That function cannot produce emergent behavior — herding, fatigue drift, strategic waiting, agent-to-agent competition. The v3 swarm replaces (or augments) the logistic with a population of LLM-driven agents, each with their own persona and memory. The bandit + allocator then learn against a *living* population instead of a *fixed* curve.

### 10.2 `DasherAgent`

`backend/swarm/dasher_agent.py` (209 lines). Each agent carries:

- **`DasherPersona`** (immutable): `dasher_id`, `tenure_days`, `home_zone`, `elasticity` (baseline responsiveness), `earnings_goal_per_day`, `fatigue_threshold` (offers before accept-rate drops), `schedule_preference` (morning/afternoon/evening/night)
- **`DasherMemory`** (mutable): list of recent `(incentive, zone_id, accepted, day, hour)` events, cumulative earnings today, cumulative offers today
- **Decision method** `decide(offer)` that returns `{accept: bool, rationale: str, new_zone: Optional[int]}`

### 10.3 Decision path

The agent has two decision paths:

- **LLM path**: construct a prompt containing persona summary, memory summary, current offer (incentive, zone, time), and ask GPT-4o to reply with a JSON decision. Each accept/reject is one OpenAI call.
- **Rule-based fallback**: deterministic logistic on `(incentive, zone_preference, fatigue, earnings_goal)`. Kicks in when no OpenAI key is configured, when `OpenAIError` is raised, or when `LLM_AGENT_DISABLED=1`. This keeps CI fast and keeps the Swarm tab usable in demos.

The rule-based path is not a toy — it reproduces the same qualitative behavior (fatigue drift, zone preference) as the LLM path, just with less variance.

### 10.4 `MarketplaceEnvironment`

`backend/swarm/marketplace_env.py` (238 lines). Thin re-implementation of the parametric simulator tailored to the swarm loop:

- `ZoneState` — demand, current incentive, dasher count, total offers, total accepts
- `step(day, hour)` — ticks demand, emits offers, collects agent decisions, applies zone switches, aggregates acceptance rates
- `set_incentive(zone_id, amount)` — hook for the allocator to push arm decisions
- `get_zone_adjacency()` — feeds the networked bandit's spatial prior

### 10.5 `DasherSwarmRunner`

`backend/swarm/swarm_runner.py` (283 lines). Orchestrates multi-day, multi-step swarm experiments:

- `propose_incentive(zone_id) → (arm_id, probability)` — asks the v2 bandit for an incentive and writes it into the marketplace
- `feed_reward(zone_id, arm_id, reward)` — writes observed acceptance back into the bandit's posterior
- `run_day(n_steps)` — loops over hours, for each hour loops over zones, for each zone loops over agents, collects decisions, aggregates
- `run_full(n_days, n_steps_per_day)` — multi-day loop with daily summaries

Return shape matches the frontend `SwarmResults` interface exactly (`n_days`, `n_dashers`, `n_zones`, `total_steps`, `aggregate` with `mean_acceptance_rate`, `std_acceptance_rate`, `total_offers`, `total_accepted`, `total_zone_switches`, `acceptance_trend`, `daily_summaries[]`, `final_zones[]`).

### 10.6 `OASISBridge`

Also in `backend/swarm/swarm_runner.py`. An async `aiohttp` REST client to the OASIS Docker sidecar:

- `is_available() → bool` — GET `/health` with a 2-second timeout
- `create_session(n_agents, profiles) → session_id` — POST `/session/create`
- `step(session_id, actions) → results` — POST `/session/{id}/step`
- `destroy(session_id)` — DELETE `/session/{id}`

OASIS is camel-ai's open-source large-scale social simulator (Apache 2.0, scales to 1M agents). The sidecar pattern lets us run OASIS in its required Python 3.11 environment while the main backend runs on Python 3.14.

### 10.7 `MiroFishBridge`

`backend/swarm/mirofish_bridge.py` (237 lines). Similar async REST client to the MiroFish container:

- `is_available() → bool`
- `generate_profiles(n_dashers) → profiles` — creates MiroFish-compatible agent schemas from Dasher personas
- `create_scenario(profiles) → scenario_id`
- `run_prediction(scenario_id, question, context) → response`

MiroFish (github.com/666ghj/MiroFish, 54k stars) is a universal swarm-intelligence engine with Zep-backed memory and a Vue UI. We use the prebuilt `ghcr.io/666ghj/mirofish:latest` image (14 GB) so we don't pay the build cost ourselves.

### 10.8 Test coverage

`backend/swarm/test_swarm.py` (120 lines) — four integration tests:

1. Single `DasherAgent.decide()` end-to-end
2. `MarketplaceEnvironment.step()` acceptance aggregation
3. `DasherSwarmRunner.run_full()` with 20 agents × 2 days
4. `OASISBridge.is_available()` network check

All pass locally. Each test is deterministic via seeded `random.Random`.

---

## 11) Docker Sidecars

### 11.1 OASIS sidecar (`oasis-sidecar/`)

**Dockerfile** (lines shown are faithful):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir camel-oasis && \
    pip install --no-cache-dir --force-reinstall "fastapi>=0.110" "uvicorn>=0.29,<1.0" aiohttp
COPY server.py .
COPY dasher_profiles.json .
EXPOSE 5050
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5050"]
```

The force-reinstall step is not cosmetic. `camel-oasis` pins an older `uvicorn` whose CLI shim at `/usr/local/bin/uvicorn` is incompatible with newer FastAPI. After force-reinstall the shim is still broken, so we call `python -m uvicorn` directly to bypass the shim. This is the kind of packaging-conflict fix that only shows up under real deployment and is worth calling out.

**server.py** — FastAPI app with health, session lifecycle, and step endpoints. Wraps `camel.oasis.Platform` when the library is available; falls back to a pure-Python rule-based session implementation when it is not.

**dasher_profiles.json** — 10 sample Dasher agents with diverse personas (veteran commuter, college student, full-time single mom, part-time retiree, etc.) that OASIS hydrates into its agent objects.

### 11.2 MiroFish (`docker-compose.yml`)

No custom image — uses `ghcr.io/666ghj/mirofish:latest`:

```yaml
mirofish:
  image: ghcr.io/666ghj/mirofish:latest
  container_name: mirofish
  ports:
    - "3001:3000"  # UI (remapped to avoid conflict with our Next.js)
    - "5001:5001"  # Backend API
  environment:
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - LLM_API_KEY=${OPENAI_API_KEY}
    - LLM_BASE_URL=https://api.openai.com/v1
    - LLM_MODEL=gpt-4o-mini
    - ZEP_API_KEY=${ZEP_API_KEY:-disabled}
    - ZEP_API_URL=${ZEP_API_URL:-http://localhost:8080}
  restart: unless-stopped
```

`ZEP_API_KEY` is a real credential for the Zep memory service, used by MiroFish for agent long-term memory. We fail gracefully to `disabled` when not provided.

### 11.3 docker-compose orchestration

A single `docker-compose up -d` brings the OASIS sidecar + MiroFish online. Both sidecars are independent of the main FastAPI backend — the backend discovers them via `is_available()` health checks and routes swarm requests accordingly. When a sidecar is unavailable, the built-in Python swarm engine handles the workload transparently.

### 11.4 Combined `/v3/status` endpoint

`backend/api/main.py` exposes `GET /v3/status` which returns the health of all three engines (built-in / OASIS / MiroFish) in a single call. This is what the frontend polls to decide which engines to show.

---

## 12) API Layer (FastAPI)

### 12.1 v2 endpoints

| Endpoint | Purpose |
|---|---|
| `POST /experiment/run` | Kicks off a multi-day experiment across all algorithms |
| `GET /simulation/snapshot` | Current simulator state (zones, dashers, pending outcomes, change points) |
| `GET /simulation/adjacency` | Zone adjacency matrix + cluster map for frontend network graph |
| `POST /bo/posterior` | GP posterior `(grid, mean, std, lower, upper)` for a zone |
| `POST /bo/suggest` | Shape-projected argmax of the acquisition function |
| `GET /causal/state` | DML + Causal Forest state |
| `GET /causal/hte-segments` | Pre-computed HTE by tenure / deficit / peak / responsiveness |
| `GET /causal/dose-response` | Nadaraya-Watson curve with CI |
| `GET /causal/interference` | Direct + indirect treatment-effect decomposition |
| `GET /change-detection/state` | Recent events + active alerts |
| `GET /reward-worker/state` | Batch history + DR estimates |
| `GET /allocator/state` | Budgets + per-zone spend |
| `POST /agent/chat` | GPT-4o meta-agent chat endpoint |
| `POST /agent/approve` | Apply a previously-proposed write action |

### 12.2 v3 swarm endpoints

| Endpoint | Purpose |
|---|---|
| `POST /swarm/run` | Run a full swarm experiment |
| `GET /swarm/state` | Current marketplace + aggregate stats |
| `GET /swarm/agents` | All agent states (persona + memory) |
| `GET /swarm/zones` | All zone states |
| `POST /swarm/step` | Single interactive step |
| `GET /oasis/health` | OASIS sidecar health |
| `POST /oasis/session/create` | Create OASIS session |
| `POST /oasis/session/{id}/step` | Step OASIS session |
| `GET /mirofish/health` | MiroFish health |
| `GET /mirofish/profiles` | MiroFish-compatible profile generation |
| `POST /mirofish/scenario` | Create MiroFish scenario |
| `POST /mirofish/predict` | Run MiroFish prediction |
| `GET /v3/status` | Combined status of all three engines |

### 12.3 CORS and reload

Dev server runs on `127.0.0.1:8000` with `--reload` for hot code iteration and CORS open to `http://localhost:3000` for the frontend.

---

## 13) Frontend

### 13.1 Technology

Next.js 16 + React + TypeScript + TailwindCSS + Recharts + shadcn/ui + Lucide icons. All pages are client-rendered with hooks (`useState`, `useCallback`, `useEffect`). API calls go through a single `apiFetch` wrapper in `src/lib/utils.ts`.

### 13.2 Tab structure

Defined as `TABS: {key, label, icon}[]` in `src/app/page.tsx`:

| Tab | Component | Contents |
|---|---|---|
| Overview | `OverviewTab` | KPIs, regret reduction, change-point count, batch summary |
| Bandits | `BanditsTab` + `ArmStats` + `RegretChart` | Arm posteriors, 3-algorithm regret chart, NetLinTS per-zone |
| Network | `NetworkTab` + `NetworkGraph` | Zone adjacency graph, cluster coloring, spillover pressure heatmap |
| Bayesian Opt | `BOTab` + `GPPosteriorChart` | GP posterior, observation history, optimal incentive |
| Causal | `CausalTab` + `HTEChart` + `DoseResponseChart` | HTE by segment, dose-response curve with CI, interference bar chart |
| System | `SystemTab` | Change-detection alerts, reward worker state, allocator spend |
| LLM Agent | `AgentTab` + `AgentChat` | GPT-4o chat with approval UI for write tools |
| Swarm | `SwarmTab` | 5 summary cards + acceptance-trend bar chart + zone performance table + daily summaries |

### 13.3 Swarm tab UX details

- Configurable `n_dashers` (10–500) and `n_days` (1–30) inputs
- Run button disables during `loading`, shows spinner
- Acceptance-trend bars are color-coded: green (>70%), yellow (40–70%), red (<40%) with increasing opacity for higher values
- Zone performance table: 7 columns, color-coded acceptance rate
- Re-run button at the bottom for rapid iteration

### 13.4 Type safety

All API responses have explicit TypeScript interfaces (`ExperimentResults`, `NetBanditState`, `ChangeDetectionState`, `RewardWorkerState`, `AllocatorState`, `CausalResults`, `HTESegments`, `BatchSummary`, `RegretComparison`, `SimSnapshot`, `ZoneSnap`, `AdjacencyData`, `SwarmResults`). No `any` except for unstructured JSON payloads from the causal layer.

---

## 14) Tech stack summary

| Layer | Technologies |
|---|---|
| Backend | Python 3.11+, FastAPI, uvicorn, NumPy, SciPy, scikit-learn, pandas |
| ML | econml (DML + Causal Forest), BoTorch + GPyTorch, PyTorch, manual DML fallback |
| Bandits | NumPy-only (no dep on ML libs) |
| LLM | OpenAI GPT-4o + `gpt-4o-mini` via `openai>=1.58` with function calling |
| Swarm engine (built-in) | Pure Python, seeded `random.Random`, OpenAI optional |
| Swarm sidecar 1 | `camel-oasis` in Python 3.11 Docker sidecar, FastAPI bridge |
| Swarm sidecar 2 | `ghcr.io/666ghj/mirofish:latest` prebuilt image (Flask + Zep + Vue) |
| Async bridges | `aiohttp` (OASIS), `httpx` (MiroFish) |
| Frontend | Next.js 16, React, TypeScript, Tailwind, Recharts, shadcn/ui, Lucide |
| Infra | Docker Compose, `python-dotenv` for key loading |
| Config | `backend/.env` for `OPENAI_API_KEY` + `ZEP_API_KEY` |

---

## 15) Code ownership

I architected and implemented every line of the DorkDash source tree:

- **Simulator** — zones, demand model, spillover, delayed feedback, propensity logging, ground-truth change-point injection
- **Bandits** — Vanilla TS, Contextual LinTS, Networked LinTS with global + local posterior decomposition, per-zone regret tracking, propensity-reporting `select_arm`
- **Bayesian Optimization** — shape-constrained GP with monotonicity + concavity projection, BoTorch integration + pure-NumPy fallback
- **Causal layer** — DML + Causal Forest + manual DML, dose-response (Nadaraya-Watson), interference decomposition (augmented OLS)
- **Change detection** — CUSUM + sliding-window LRT + posterior-drift KL monitor
- **Reward worker** — batch outcome resolution, treatment-effect-vs-control, AIPW correction
- **Constrained allocator** — DISCO-style integer constraint layer, daily/weekly budgets, per-arm caps, exploration floor, sticky cohorts
- **LLM meta-agent** — GPT-4o with 12 structured tools, approval workflow, mock mode
- **Swarm layer** — `DasherAgent` + persona + memory, `MarketplaceEnvironment`, `DasherSwarmRunner`, `OASISBridge`, `MiroFishBridge`, rule-based fallback decision path
- **Docker sidecars** — OASIS Dockerfile with the `python -m uvicorn` shim fix, `server.py` FastAPI bridge, `dasher_profiles.json`, MiroFish docker-compose with ZEP + LLM env wiring
- **API layer** — all 27 FastAPI endpoints across v2 and v3
- **Frontend** — 8 tab components, 7 visualization components (`RegretChart`, `GPPosteriorChart`, `ZoneHeatmap`, `HTEChart`, `ArmStats`, `AgentChat`, `NetworkGraph`, `DoseResponseChart`), `SwarmTab` with acceptance-trend bars and zone performance table

I did not write the underlying frameworks (FastAPI, econml, BoTorch, OpenAI SDK, camel-oasis, MiroFish) — I used them as building blocks and wrote integration code, bridges, and fallbacks on top.

---

## 16) What was NOT implemented — do not claim

- **Real production deployment**: the platform runs on `localhost`. No Kubernetes, no load balancer, no multi-tenant auth, no on-call rotation. Treat "production-realistic" as "mirrors the patterns of a production system" not "is currently running in production".
- **Real DoorDash data**: all simulation is synthetic. Dasher personas, zone demand curves, and response elasticities are designed to look plausible but are not calibrated against real DoorDash logs.
- **Formal regret bounds**: the Networked LinTS implementation follows the paper's architecture but we did not prove regret bounds ourselves.
- **A/B test statistical framework**: the reward worker emits DR estimates but there is no sequential-testing / always-valid-confidence-interval layer wrapping them. Present results as descriptive, not as statistically-valid A/B test conclusions.
- **OASIS at 1M agents**: the sidecar runs `camel-oasis`, which *claims* 1M-agent scale. We have tested up to 500 agents locally. Present 1M as "what OASIS supports architecturally" not "what we ran".
- **MiroFish end-to-end**: the MiroFish backend is up and healthy and the bridge is wired, but we have not driven a full scenario → prediction pipeline through the Swarm tab UI. The Zep memory integration in particular is untested end-to-end.
- **Formal regret measurement against ground-truth optimal policy**: we track cumulative regret against a per-step oracle, but the oracle itself is synthetic. Treat regret comparisons as relative (Net TS beats Vanilla TS) not absolute.
- **Compliance / audit**: no PII, no GDPR, no SOC2. Synthetic data only.

---

## 17) Defensible vs must-qualify

### Fully defensible

- Networked LinTS with global + local posterior decomposition
- Shape-constrained BO with monotone + concave projection
- AIPW / doubly-robust reward correction with logged propensities
- Dose-response via Nadaraya-Watson
- Direct vs indirect treatment-effect decomposition under partial interference
- CUSUM + sliding-window + posterior-drift change detection
- DISCO-style constrained allocator
- GPT-4o meta-agent with 12 tools and approval workflow
- Built-in swarm engine with LLM + rule-based fallback
- OASIS + MiroFish Docker sidecar architecture
- ~7,200 lines of Python + TypeScript you can open and read
- Integration tests for the swarm layer (all passing)
- v3/status health check returning all three engines connected, verified live

### Must qualify

- "Lowest cumulative regret" — empirically observed in our 3–14 day synthetic runs; not a theoretical claim
- "~40–60% regret reduction" — one run's numbers; re-runs with different seeds will vary
- "114% spillover ratio" — from a single 3-day test; interpret as "spillover is first-order important" not "this exact number holds at production scale"
- "1M agents via OASIS" — architectural capability of the underlying library; we tested up to 500
- "DoorDash-like" — I am not a DoorDash employee and this is not a DoorDash product. The architecture mirrors their public engineering blogs; the data is synthetic.
- "Production-realistic" — mirrors production patterns (batch reward worker, propensity logging, allocator constraints) but is not deployed to real production infrastructure

---

## 18) Strongest spoken version

"DorkDash is a seven-thousand-line Python and TypeScript platform that takes DoorDash's published Thompson Sampling MAB from their 2022 blog and extends it through the components their 2025 platform blog outlines: contextual bandits, shape-constrained Bayesian Optimization, real causal ML with double machine learning, change-point detection, a doubly-robust batch reward worker with logged propensities, and a DISCO-style constrained allocator. Every layer is independently testable, independently swappable, and has its own tab in the frontend.

On top of that I built an LLM meta-agent — GPT-4o with twelve structured tools covering every backend component and an approval workflow so every write action is a proposal requiring a human in the loop. Then in v3 I added a multi-agent swarm layer where every simulated Dasher is an autonomous LLM agent with a persona, episodic memory, and fatigue dynamics. The swarm connects to the v2 bandit through a runner that proposes incentives, collects agent accept-reject decisions, and feeds rewards back.

The swarm runs in three pluggable modes: a built-in Python engine, a camel-ai OASIS Docker sidecar for 1M-agent-scale social simulation, and the prebuilt MiroFish container with Zep memory. Docker Compose brings both sidecars up, and a single `v3/status` endpoint reports which engines are healthy. I hit one interesting packaging bug getting OASIS working — camel-oasis pins an older uvicorn whose CLI shim is incompatible with newer FastAPI, so I force-reinstall uvicorn and call it via `python -m uvicorn` to bypass the shim.

The whole stack runs locally with `docker-compose up` plus `uvicorn` plus `npm run dev`. When I last ran it end-to-end, all three swarm engines reported connected, the bandit converged to the lowest-regret algorithm being the networked one, the dose-response curve gave an optimal incentive around $5, and the interference decomposition showed a 114% spillover ratio — meaning a naive per-zone A/B test would over-estimate the direct effect by more than 2×."

---

## 19) Exact phrases to use under pressure

**On why networked bandits**: "A single-zone MAB is blind to the fact that incentivising zone A pulls Dashers from zone B. Networked LinTS decomposes the posterior into a global prior plus a zone-local deviation, so zones in the same cluster share information. New zones with three observations inherit the cluster prior instead of having a useless posterior."

**On why shape-constrained BO**: "A vanilla GP fitted on sparse incentive-response data will happily produce a non-monotone posterior — higher incentive, lower response. That is obviously wrong. Shape constraints enforce monotone non-decreasing and concave (diminishing returns). It is a prior that matches physical reality."

**On AIPW / doubly-robust**: "Every incentive send logs the policy probability. The reward worker drains the pending outcome queue in daily batches and computes treatment-effect-vs-control with AIPW correction. AIPW is consistent if either the outcome model or the propensity model is correct. Without logged propensities the offline-policy evaluation is not identifiable."

**On interference**: "In partially-interfering environments, per-zone A/B tests are biased because the treatment zone's gain partially comes from the untreated neighbor's loss. I decompose the treatment effect using augmented OLS into a direct (own-zone) component and an indirect (sum of adjacent-zone treatments) component. In our runs the spillover ratio was 114%, meaning the indirect effect was actually larger in magnitude than the direct one."

**On change detection**: "I replaced heuristic weight decay with three explicit detectors — CUSUM on reward streams, a sliding-window likelihood-ratio test, and a posterior-drift KL monitor. Each event is logged with zone, detector, severity, and a recommended action, so the LLM meta-agent can query change-detection state as a structured tool instead of having to infer non-stationarity from raw rewards."

**On the allocator**: "The allocator sits between the bandit's arm choice and the real action. It enforces daily and weekly budget caps, per-arm frequency caps per Dasher per week, an exploration floor that guarantees minimum pulls per arm, and sticky-cohort logic that keeps a Dasher in the same arm for the week. When the bandit's preferred arm violates a constraint, the allocator falls back to the best feasible arm."

**On the LLM meta-agent**: "GPT-4o with twelve structured tools — ten read tools covering every backend component and two write tools that only ever return proposals. The proposals surface in the frontend with approve/reject buttons. No LLM-generated decision flows to production without a human PM clicking approve. That approval workflow is non-negotiable."

**On the swarm**: "Every simulated Dasher is an autonomous LLM agent with an immutable persona — tenure, home zone, earnings goal, fatigue threshold — and mutable episodic memory of past offers and outcomes. The agent's decide method is either a GPT-4o call or a deterministic rule-based fallback, so the swarm runs in CI without an OpenAI key. The runner wires agent decisions back to the v2 bandit so the bandit is learning against a living population, not a fixed parametric curve."

**On why two Docker sidecars**: "OASIS is camel-ai's open-source large-scale social simulator that scales to a million agents architecturally. It pins Python 3.11. My backend runs on Python 3.14. The sidecar pattern lets each half run in its own Python environment. MiroFish is a different kind of framework with Zep-backed memory and a Vue UI — I pulled the prebuilt image so I wasn't paying the 14-gig build cost myself."

**On the uvicorn shim bug**: "Camel-oasis pins an older uvicorn whose CLI entry point at `/usr/local/bin/uvicorn` is incompatible with newer FastAPI. Force-reinstalling uvicorn doesn't replace the shim. I switched the container's CMD to `python -m uvicorn` to bypass the shim. That's the kind of packaging-conflict fix that only shows up once you actually deploy the thing."

**On the aiohttp silent-failure**: "The first time I brought the stack up, the OASIS sidecar was healthy but the backend kept reporting it unavailable. Root cause: `aiohttp` wasn't in `requirements.txt`, so the bridge's `is_available()` was silently catching ImportError and returning False. Adding `aiohttp` and `httpx` to requirements fixed it. Always catch ImportError explicitly in bridge code."

**On testing**: "Integration tests for the swarm live in `backend/swarm/test_swarm.py` — four tests covering single-agent decide, marketplace step aggregation, the full multi-day runner, and the OASIS bridge health check. Each test seeds its own `random.Random` so the results are deterministic. All passing."

**On what I'd do next**: "Phase 6 is using swarm-generated trajectories to re-fit the causal ML models and compare the dose-response curve you get from the LLM swarm against the parametric one. If they disagree, the disagreement itself is interesting — it tells you which assumptions in the parametric simulator are load-bearing and which aren't."

---

## 20) Exact phrases to avoid

- "This is running in DoorDash production" — it is a synthetic-data reference implementation that mirrors DoorDash's public engineering blogs
- "I have DoorDash's real data" — all data is synthetic
- "We proved regret bounds" — we implemented the algorithms from the papers; we did not re-derive bounds
- "OASIS handles a million agents for us" — OASIS *supports* that architecturally; we tested up to a few hundred
- "Statistically valid A/B test result" — the reward worker emits DR estimates; we did not wrap them in a sequential-testing framework
- "GDPR / SOC 2 / HIPAA compliant" — no PII, no compliance layer, synthetic data only
- "I wrote camel-oasis / MiroFish / econml / BoTorch" — I used them as building blocks and wrote the integration, bridges, and fallbacks
- "This scales to production traffic" — local dev server, Uvicorn `--reload`, single-process FastAPI; not load-tested
- "The LLM decides autonomously" — the LLM proposes; a human approves; no write action bypasses the approval workflow
