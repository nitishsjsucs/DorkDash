# DorkDash

A simulator for studying multi-armed bandit approaches to courier incentive allocation,
with a dashboard to watch them run. A synthetic marketplace of ten Bay Area delivery zones
generates couriers who accept or decline dollar incentives; several bandit algorithms
compete to allocate those incentives, and causal-inference code tries to recover the
treatment effects the simulator itself generated.

The name is a play on DoorDash. **This project has no affiliation with DoorDash, uses no
DoorDash data, and has not been run against any real marketplace.** It is a personal
exercise built by reading their public engineering blog posts (linked below). Every number
in it comes from the simulator.

The algorithm code is real — NumPy, scikit-learn, econml, BoTorch — not stubs. What is not
real is the setting.

## What's here

`backend/` — Python 3.11, FastAPI, entry point `backend/api/main.py` (41 routes).

- `simulation/environment.py` — the marketplace: 10 zones with an adjacency graph,
  cross-zone spillover, delayed feedback, and regime shifts.
- `bandits/thompson_sampling.py` — Beta-Bernoulli Thompson sampling with weight decay.
- `bandits/contextual_bandit.py` — Linear Thompson sampling (Agrawal & Goyal), with the
  standard Bayesian linear-regression posterior update.
- `bandits/networked_bandit.py` — a networked LinTS variant that blends a global precision
  matrix with per-zone local ones.
- `bandits/change_detector.py` — CUSUM, sliding-window, and posterior-drift detectors.
- `causal/treatment_effects.py` — econml `LinearDML` and `CausalForestDML` with a
  hand-rolled residual-on-residual DML fallback, Nadaraya-Watson dose-response curves, and
  an interference model that splits direct from spillover effects.
- `optimization/bayesian_opt.py` — GP surrogate with EI/UCB acquisition.
- `optimization/constrained_allocator.py` — greedy allocation under daily and weekly
  budgets, per-arm frequency caps, an exploration floor, and sticky cohorts.
- `simulation/reward_worker.py` — batches delayed outcomes and estimates
  treatment-effect-versus-control.
- `agent/meta_agent.py` — an LLM meta-agent with 12 tools over the components above and an
  approval step for writes. Falls back to a deterministic mock when no API key is set.
- `swarm/` — an optional layer where couriers are LLM-driven agents with personas and
  fatigue, plus HTTP bridges to two third-party containers (OASIS, MiroFish).
- `cache/` — a committed snapshot of one real 21-day run, so the dashboard has something to
  show before you run anything.

`frontend/` — Next.js 16, React 19, Tailwind 4, Recharts. `src/app/page.tsx` is a
hand-written 8-tab dashboard (Overview, Bandits, Network, Bayesian Opt, Causal ML, System,
LLM Agent, Swarm) with nine bespoke chart components. It calls 8 of the backend's 41 routes.

## Running it

```bash
cd backend && pip install -r requirements.txt
python -m uvicorn backend.api.main:app --reload     # from the repo root, port 8000

cd frontend && npm install && npm run dev            # port 3000
```

Env vars by name: `OPENAI_API_KEY` (only for the meta-agent and LLM swarm; both have
non-LLM fallbacks), `LLM_AGENT_DISABLED`, and `NEXT_PUBLIC_API_URL` for the frontend.

The optional sidecars come up with `docker-compose up`. They need `OPENAI_API_KEY` and, for
MiroFish, optionally `ZEP_API_KEY` / `ZEP_API_URL`. Without them the `/oasis/*` and
`/mirofish/*` routes return 503; nothing else is affected.

## Status / limitations

- **All data is synthetic.** The causal estimates recover effects the simulator was
  programmed to produce. Treat the regret and lift numbers as a check that the
  implementations are wired up correctly, not as evidence about real couriers.
- Localhost only. No persistence layer, no queue, no deployment story.
- Three names in the code promise more than the code does, and I have left the names alone
  rather than silently changing behaviour:
  - `reward_worker.py` labels its estimator doubly-robust AIPW. It is self-normalized
    (Hájek) IPW — there is no outcome-regression augmentation term.
  - The "shape-constrained" GP in `bayesian_opt.py` fits an ordinary GP, then monotonizes
    and concavifies the predicted means in a loop and feeds those back as high-noise pseudo
    observations. That is a post-hoc projection, not derivative-constrained inference.
  - `constrained_allocator.py` is described as a DISCO-style integer programme. There is no
    solver in the repo; it is a single greedy pass with constraint guards.
- The committed cache was produced with the NumPy GP fallback, not BoTorch.
- The OASIS sidecar degrades to a random-number stub unless `camel-oasis` is installed, and
  that dependency is commented out in `requirements.txt`. The MiroFish bridge has never been
  driven end to end.
- `Cortex_Enterprise_AI_Research_Platform.md` in the repo root describes an entirely
  different project — a Rust/CQRS/Kubernetes system. There is no Rust, GraphQL, or
  Kubernetes code here. It is a stray file.
  `DorkDash_Agentic_Bandit_Platform.md` does describe this code, accurately.
- `oasis-sidecar/log/` holds four zero-byte log files.
- Single commit, no tests.

## Attribution

The problem framing and the baseline come from DoorDash's public engineering writing, which
this project reimplements on synthetic data and extends:

- Arjun Sharma, [Using a Multi-Armed Bandit with Thompson Sampling to Identify Responsive
  Dashers](https://careersatdoordash.com/blog/using-a-multi-armed-bandit-with-thompson-sampling-to-identify-responsive-dashers/) (2022)
- Weinstein & Huang, [Accelerating Experimentation with a Multi-Armed Bandit
  Platform](https://careersatdoordash.com/blog/experimentation-at-doordash-with-a-multi-armed-bandit-platform/) (2025)
- Xu et al., [Causal ML for Promotions](https://causal-machine-learning.github.io/kdd2025-workshop/papers/16.pdf), KDD 2025

Built on [econml](https://github.com/py-why/EconML),
[BoTorch](https://botorch.org)/[GPyTorch](https://gpytorch.ai),
[scikit-learn](https://scikit-learn.org), [FastAPI](https://fastapi.tiangolo.com),
[Next.js](https://nextjs.org), and [Recharts](https://recharts.org). The optional swarm
sidecars use [OASIS](https://github.com/camel-ai/oasis) (camel-ai, Apache 2.0) and
[MiroFish](https://github.com/666ghj/MiroFish).
