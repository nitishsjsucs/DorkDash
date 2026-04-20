"""
Experiment runner v2: orchestrates simulation, bandits (vanilla + contextual + networked),
BO, causal estimation, constraint-aware allocation, batch reward worker, and
non-stationarity detection.

Produces comparative regret curves, business metrics, and constraint satisfaction reports.

v2 upgrades:
  - Networked contextual bandits (NetLinTS) as third algorithm
  - Batch reward worker (treatment-effect relative to control)
  - Constraint-aware allocator (budgets, caps, fairness, sticky assignment)
  - Non-stationarity detection (CUSUM + posterior drift + sliding window)
  - Propensity logging for doubly-robust estimation
  - Delayed feedback resolution
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from backend.simulation.environment import MarketplaceSimulator
from backend.bandits.thompson_sampling import VanillaThompsonSampling
from backend.bandits.contextual_bandit import LinThompsonSampling, build_context_vector
from backend.bandits.networked_bandit import NetLinTS, build_context_vector as net_build_ctx
from backend.bandits.change_detector import NonStationarityDetector
from backend.optimization.bayesian_opt import IncentiveOptimizer
from backend.optimization.constrained_allocator import ConstrainedAllocator, AllocationConstraints
from backend.causal.treatment_effects import CausalEstimator, DoseResponseEstimator, InterferenceAwareEstimator
from backend.simulation.reward_worker import RewardWorker


DISCRETE_ARMS = ["$0", "$1", "$2", "$3", "$4", "$5"]
ARM_VALUES = {a: float(a.replace("$", "")) for a in DISCRETE_ARMS}


class ExperimentRunner:
    """Runs a full comparative experiment across all algorithms."""

    def __init__(self, seed: int = 42, n_dashers_per_zone: int = 100):
        self.seed = seed
        self.simulator = MarketplaceSimulator(seed=seed, n_dashers_per_zone=n_dashers_per_zone)

        # Algorithms
        self.vanilla_ts = VanillaThompsonSampling(
            arm_ids=DISCRETE_ARMS, seed=seed
        )
        self.contextual_ts = LinThompsonSampling(
            arm_ids=DISCRETE_ARMS, seed=seed + 1
        )
        self.net_ts = NetLinTS(
            arm_ids=DISCRETE_ARMS,
            n_zones=self.simulator.n_zones,
            adjacency=self.simulator.adjacency,
            seed=seed + 3,
        )
        self.bo = IncentiveOptimizer(seed=seed + 2)
        self.causal_dml = CausalEstimator(method="dml", seed=seed)
        self.causal_forest = CausalEstimator(method="causal_forest", seed=seed)
        self.dose_response = DoseResponseEstimator(seed=seed)
        self.interference_est = InterferenceAwareEstimator(seed=seed)

        # v2 components
        self.reward_worker = RewardWorker()
        self.change_detector = NonStationarityDetector(n_zones=self.simulator.n_zones)
        self.allocator = ConstrainedAllocator(
            constraints=AllocationConstraints(),
            seed=seed + 4,
        )

        # Tracking
        self.step_results: List[Dict] = []
        self.batch_summaries: List[Dict] = []
        self.all_records: List[Dict] = []
        self.batch_outcomes: List[Dict] = []  # current batch for reward worker

    def run_step(
        self,
        n_dashers_per_zone: int = 10,
        zone_ids: Optional[List[int]] = None,
    ) -> Dict:
        """
        Run one time step of the experiment.
        Each algorithm selects arms for dashers, observes rewards, updates.
        """
        if zone_ids is None:
            zone_ids = list(range(self.simulator.n_zones))

        step_data = {
            "step": self.simulator.state.total_steps,
            "day": self.simulator.state.current_day,
            "hour": self.simulator.state.current_hour,
            "vanilla_ts": {"rewards": [], "regret": 0},
            "contextual_ts": {"rewards": [], "regret": 0},
            "net_ts": {"rewards": [], "regret": 0},
            "bo": {"rewards": [], "suggestions": []},
            "change_events": [],
        }

        for zone_id in zone_ids:
            zone = self.simulator.zones[zone_id]
            zone_ctx = self.simulator.get_zone_context(zone_id)
            zone_dashers = [
                d for d in self.simulator.dashers if d.home_zone == zone_id
            ]

            if len(zone_dashers) == 0:
                continue

            selected_indices = self.simulator.rng.choice(
                len(zone_dashers),
                size=min(n_dashers_per_zone, len(zone_dashers)),
                replace=False,
            )

            # BO sets ONE pay boost per zone per step (matches real-world cadence).
            # Fitting a GP per dasher would be 10x slower and semantically wrong.
            bo_incentive_zone = self.bo.suggest_incentive(zone_id)
            bo_batch_observations = []

            for idx in selected_indices:
                dasher = zone_dashers[idx]
                dasher_ctx = self.simulator.get_dasher_context(dasher)

                # --- Compute oracle optimal for regret calculation ---
                best_reward = 0
                for arm_val in ARM_VALUES.values():
                    te = self.simulator.true_treatment_effect(dasher, arm_val, zone_id)
                    if te > best_reward:
                        best_reward = te

                # --- Vanilla Thompson Sampling ---
                chosen_arm_vts = self.vanilla_ts.select_arm()
                incentive_vts = ARM_VALUES[chosen_arm_vts]
                responded_vts, te_vts, info_vts = self.simulator.send_incentive(
                    dasher.dasher_id, incentive_vts, zone_id
                )
                self.vanilla_ts.update(chosen_arm_vts, float(responded_vts))
                self.vanilla_ts.update_regret(te_vts, best_reward)
                step_data["vanilla_ts"]["rewards"].append(float(responded_vts))

                # --- Contextual LinTS ---
                context = build_context_vector(zone_ctx, dasher_ctx)
                chosen_arm_cts = self.contextual_ts.select_arm(context)
                incentive_cts = ARM_VALUES[chosen_arm_cts]
                responded_cts, te_cts, info_cts = self.simulator.send_incentive(
                    dasher.dasher_id, incentive_cts, zone_id
                )
                self.contextual_ts.update(chosen_arm_cts, context, float(responded_cts))
                self.contextual_ts.update_regret(te_cts, best_reward)
                step_data["contextual_ts"]["rewards"].append(float(responded_cts))

                # --- Networked LinTS ---
                net_context = net_build_ctx(zone_ctx, dasher_ctx)
                chosen_arm_net, net_prob = self.net_ts.select_arm(net_context, zone_id)
                incentive_net = ARM_VALUES[chosen_arm_net]
                responded_net, te_net, info_net = self.simulator.send_incentive(
                    dasher.dasher_id, incentive_net, zone_id, policy_prob=net_prob
                )
                self.net_ts.update(chosen_arm_net, net_context, float(responded_net), zone_id)
                self.net_ts.update_regret(te_net, best_reward, zone_id)
                step_data["net_ts"]["rewards"].append(float(responded_net))

                # --- Non-stationarity detection ---
                events = self.change_detector.observe(zone_id, te_net, self.simulator.state.total_steps)
                for evt in events:
                    step_data["change_events"].append({
                        "zone_id": evt.zone_id,
                        "detector": evt.detector,
                        "severity": evt.severity,
                        "description": evt.description,
                    })

                # --- Bayesian Optimization (continuous) ---
                # Use zone-level suggestion computed once above (not per dasher).
                responded_bo, te_bo, info_bo = self.simulator.send_incentive(
                    dasher.dasher_id, bo_incentive_zone, zone_id
                )
                bo_batch_observations.append((bo_incentive_zone, te_bo, zone_id))
                step_data["bo"]["rewards"].append(float(responded_bo))
                step_data["bo"]["suggestions"].append(bo_incentive_zone)

                # Store for causal model training + reward worker
                record = {**info_net, **zone_ctx, **dasher_ctx, "arm_id": chosen_arm_net}
                self.all_records.append(record)
                self.batch_outcomes.append(record)

            # Flush all BO observations for this zone at once (no re-fit mid-loop).
            for inc, te, zid in bo_batch_observations:
                self.bo.add_observation(inc, te, zid)

        # Resolve any delayed outcomes
        resolved = self.simulator.resolve_pending()

        # Advance simulation
        self.simulator.step()
        self.step_results.append(step_data)

        return step_data

    def run_batch(
        self,
        n_steps: int = 24,  # 1 day
        n_dashers_per_zone: int = 10,
    ) -> Dict:
        """Run a batch of steps (e.g., one day = 24 hours)."""
        self.batch_outcomes = []  # reset batch accumulator
        batch_results = []
        for _ in range(n_steps):
            result = self.run_step(n_dashers_per_zone=n_dashers_per_zone)
            batch_results.append(result)

        # Batch-update vanilla TS (mimics DoorDash's daily batch cadence)
        self.vanilla_ts.batch_update([])  # applies decay only

        # Process batch through reward worker (treatment-effect relative to control)
        if self.batch_outcomes:
            batch_summary_rw = self.reward_worker.process_batch(
                self.batch_outcomes, ARM_VALUES
            )

        # Reset allocator daily budget
        self.allocator.reset_daily()

        summary = self._compute_batch_summary(batch_results)
        self.batch_summaries.append(summary)
        return summary

    def run_full_experiment(
        self,
        n_days: int = 14,
        n_dashers_per_zone: int = 10,
    ) -> Dict:
        """Run a complete multi-day experiment."""
        self.simulator.reset()

        for day in range(n_days):
            self.run_batch(n_steps=24, n_dashers_per_zone=n_dashers_per_zone)
            # Reset weekly caps every 7 days
            if (day + 1) % 7 == 0:
                self.allocator.reset_weekly()

        # Fit causal models on collected data
        causal_results = {}
        if len(self.all_records) > 100:
            try:
                dml_diag = self.causal_dml.fit(self.all_records)
                causal_results["dml"] = dml_diag
                causal_results["hte_segments"] = self.causal_dml.estimate_hte_by_segment(
                    self.all_records
                )
            except Exception as e:
                causal_results["dml_error"] = str(e)

            try:
                cf_diag = self.causal_forest.fit(self.all_records)
                causal_results["causal_forest"] = cf_diag
                causal_results["cf_hte_segments"] = self.causal_forest.estimate_hte_by_segment(
                    self.all_records
                )
            except Exception as e:
                causal_results["cf_error"] = str(e)

            # v2: Dose-response curve
            try:
                dr_diag = self.dose_response.fit(self.all_records)
                causal_results["dose_response"] = dr_diag
                causal_results["dose_response_curve"] = self.dose_response.get_curve()
            except Exception as e:
                causal_results["dose_response_error"] = str(e)

            # v2: Interference-aware estimation
            try:
                intf_diag = self.interference_est.fit(self.all_records)
                causal_results["interference"] = intf_diag
                causal_results["interference_by_zone"] = self.interference_est.estimate_by_zone(
                    self.all_records
                )
            except Exception as e:
                causal_results["interference_error"] = str(e)

        return {
            "n_days": n_days,
            "total_steps": self.simulator.state.total_steps,
            "total_records": len(self.all_records),
            "vanilla_ts": self.vanilla_ts.get_state(),
            "contextual_ts": self.contextual_ts.get_state(),
            "net_ts": self.net_ts.get_state(),
            "bo": self.bo.get_state(),
            "causal": causal_results,
            "batch_summaries": self.batch_summaries,
            "regret_comparison": self._get_regret_comparison(),
            "change_detection": self.change_detector.get_state(),
            "reward_worker": self.reward_worker.get_state(),
            "allocator": self.allocator.get_state(),
            "change_points": self.simulator.state.change_point_log,
        }

    def _compute_batch_summary(self, batch_results: List[Dict]) -> Dict:
        """Compute summary statistics for a batch of steps."""
        vts_rewards = []
        cts_rewards = []
        nts_rewards = []
        bo_rewards = []

        for r in batch_results:
            vts_rewards.extend(r["vanilla_ts"]["rewards"])
            cts_rewards.extend(r["contextual_ts"]["rewards"])
            nts_rewards.extend(r["net_ts"]["rewards"])
            bo_rewards.extend(r["bo"]["rewards"])

        return {
            "day": batch_results[-1]["day"] if batch_results else 0,
            "n_steps": len(batch_results),
            "vanilla_ts": {
                "avg_reward": float(np.mean(vts_rewards)) if vts_rewards else 0,
                "total_reward": float(np.sum(vts_rewards)),
                "cumulative_regret": self.vanilla_ts.cumulative_regret,
            },
            "contextual_ts": {
                "avg_reward": float(np.mean(cts_rewards)) if cts_rewards else 0,
                "total_reward": float(np.sum(cts_rewards)),
                "cumulative_regret": self.contextual_ts.cumulative_regret,
            },
            "net_ts": {
                "avg_reward": float(np.mean(nts_rewards)) if nts_rewards else 0,
                "total_reward": float(np.sum(nts_rewards)),
                "cumulative_regret": self.net_ts.cumulative_regret,
            },
            "bo": {
                "avg_reward": float(np.mean(bo_rewards)) if bo_rewards else 0,
                "total_reward": float(np.sum(bo_rewards)),
            },
        }

    def _get_regret_comparison(self) -> Dict:
        """Get regret histories for all algorithms for plotting."""
        vts_regret = self.vanilla_ts.regret_history
        cts_regret = self.contextual_ts.regret_history
        nts_regret = self.net_ts.regret_history
        n = min(len(vts_regret), len(cts_regret), len(nts_regret)) if nts_regret else min(len(vts_regret), len(cts_regret))

        # Downsample for API (max 500 points)
        if n > 500:
            indices = np.linspace(0, n - 1, 500, dtype=int)
            vts_regret = [vts_regret[i] for i in indices]
            cts_regret = [cts_regret[i] for i in indices]
            nts_regret = [nts_regret[i] for i in indices] if nts_regret else []
        else:
            vts_regret = vts_regret[:n]
            cts_regret = cts_regret[:n]
            nts_regret = nts_regret[:n] if nts_regret else []

        return {
            "steps": list(range(len(vts_regret))),
            "vanilla_ts_regret": vts_regret,
            "contextual_ts_regret": cts_regret,
            "net_ts_regret": nts_regret,
            "regret_reduction_pct": (
                (vts_regret[-1] - cts_regret[-1]) / max(vts_regret[-1], 1e-6) * 100
                if vts_regret and cts_regret else 0
            ),
            "net_ts_reduction_pct": (
                (vts_regret[-1] - nts_regret[-1]) / max(vts_regret[-1], 1e-6) * 100
                if vts_regret and nts_regret else 0
            ),
        }

    def get_state(self) -> Dict:
        """Get current experiment state for API."""
        return {
            "simulation": self.simulator.get_snapshot(),
            "vanilla_ts": self.vanilla_ts.get_state(),
            "contextual_ts": self.contextual_ts.get_state(),
            "net_ts": self.net_ts.get_state(),
            "bo": self.bo.get_state(),
            "n_records": len(self.all_records),
            "n_batches": len(self.batch_summaries),
            "change_detection": self.change_detector.get_state(),
            "reward_worker": self.reward_worker.get_state(),
            "allocator": self.allocator.get_state(),
        }
