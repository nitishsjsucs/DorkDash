"""
Build the shipped result cache.

Runs a full multi-day experiment + a short swarm simulation, saves the
results under backend/cache/. This cache is what the frontend loads on
a fresh clone so reviewers can see past results without re-running.

Usage:
    python scripts/build_cache.py [--days 21] [--swarm-days 3] [--swarm-dashers 20]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the swarm fast + deterministic by forcing the rule-based fallback.
# (We want the shipped cache to be reproducible without hitting OpenAI.)
os.environ.pop("OPENAI_API_KEY", None)
os.environ["LLM_AGENT_DISABLED"] = "1"

from backend.simulation.experiment_runner import ExperimentRunner
from backend.swarm.swarm_runner import DasherSwarmRunner
from backend.cache import save_experiment_snapshot, save_swarm_result


def build_experiment_cache(n_days: int) -> None:
    print(f"[experiment] running {n_days}-day experiment...", flush=True)
    t0 = time.time()
    runner = ExperimentRunner(seed=42, n_dashers_per_zone=10)
    runner.simulator.reset()

    for day in range(n_days):
        day_t = time.time()
        runner.run_batch(n_steps=24, n_dashers_per_zone=10)
        if (day + 1) % 7 == 0:
            runner.allocator.reset_weekly()
        print(f"  day {day+1}/{n_days} done in {time.time()-day_t:.1f}s  "
              f"(records so far: {len(runner.all_records)})", flush=True)

    print(f"[experiment] simulation done in {time.time()-t0:.1f}s, "
          f"total records={len(runner.all_records)}", flush=True)

    # Fit causal models
    causal_results = {}
    if len(runner.all_records) > 100:
        for label, model in [("dml", runner.causal_dml), ("causal_forest", runner.causal_forest)]:
            try:
                ct = time.time()
                diag = model.fit(runner.all_records)
                causal_results[label] = diag
                segs = model.estimate_hte_by_segment(runner.all_records)
                if label == "dml":
                    causal_results["hte_segments"] = segs
                else:
                    causal_results["cf_hte_segments"] = segs
                print(f"  {label} fit done in {time.time()-ct:.1f}s", flush=True)
            except Exception as e:
                causal_results[f"{label}_error"] = str(e)
                print(f"  {label} error: {e}", flush=True)
        try:
            runner.dose_response.fit(runner.all_records)
            causal_results["dose_response_curve"] = runner.dose_response.get_curve()
        except Exception as e:
            causal_results["dose_response_error"] = str(e)
        try:
            runner.interference_est.fit(runner.all_records)
        except Exception as e:
            causal_results["interference_error"] = str(e)

    # Build results dict matching what run_full_experiment returns
    results = {
        "n_days": n_days,
        "total_steps": runner.simulator.state.total_steps,
        "total_records": len(runner.all_records),
        "vanilla_ts": runner.vanilla_ts.get_state(),
        "contextual_ts": runner.contextual_ts.get_state(),
        "net_ts": runner.net_ts.get_state(),
        "bo": runner.bo.get_state(),
        "causal": causal_results,
        "batch_summaries": runner.batch_summaries,
        "regret_comparison": runner._get_regret_comparison(),
        "change_detection": runner.change_detector.get_state(),
        "reward_worker": runner.reward_worker.get_state(),
        "allocator": runner.allocator.get_state(),
    }
    print(f"[experiment] total time {time.time()-t0:.1f}s", flush=True)

    sim_snapshot = runner.simulator.get_snapshot()
    adjacency = {
        "adjacency": runner.simulator.get_adjacency_matrix(),
        "clusters": runner.simulator.get_cluster_map(),
    }
    bo_posteriors = {}
    for zid in range(min(3, runner.simulator.n_zones)):
        try:
            bo_posteriors[zid] = runner.bo.get_posterior(zone_id=zid, n_points=100)
        except Exception as exc:
            print(f"[experiment] posterior zone {zid} skipped: {exc}")

    save_experiment_snapshot(
        experiment_results=results,
        sim_snapshot=sim_snapshot,
        adjacency=adjacency,
        bo_posteriors=bo_posteriors,
    )
    print("[experiment] snapshot saved to backend/cache/experiment_snapshot.json")


async def build_swarm_cache(n_dashers: int, n_days: int) -> None:
    print(f"[swarm] running {n_days}-day swarm ({n_dashers} dashers, rule-based)...", flush=True)
    t0 = time.time()
    runner = DasherSwarmRunner(n_zones=10, n_dashers=n_dashers, seed=42)
    results = await runner.run(n_days=n_days)
    print(f"[swarm] done in {time.time()-t0:.1f}s, steps={results['total_steps']}")

    response = {
        "n_days": results["n_days"],
        "n_dashers": results["n_dashers"],
        "n_zones": results["n_zones"],
        "total_steps": results["total_steps"],
        "aggregate": results["aggregate"],
        "daily_summaries": results["daily_summaries"],
        "final_zones": results["final_zones"],
    }
    save_swarm_result(response)
    print("[swarm] snapshot saved to backend/cache/swarm_result.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=21, help="Number of simulated days for the main experiment")
    parser.add_argument("--swarm-days", type=int, default=3, help="Number of simulated days for the swarm")
    parser.add_argument("--swarm-dashers", type=int, default=20, help="Number of dashers in the swarm")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--skip-swarm", action="store_true")
    args = parser.parse_args()

    if not args.skip_experiment:
        build_experiment_cache(args.days)
    if not args.skip_swarm:
        asyncio.run(build_swarm_cache(args.swarm_dashers, args.swarm_days))

    print("\nAll caches written under backend/cache/. Commit these files to ship real data with the repo.")


if __name__ == "__main__":
    main()
