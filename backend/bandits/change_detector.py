"""
Non-stationarity detection for bandit environments.
Replaces heuristic decay with principled change-point detection.

Implements three complementary approaches:
1. CUSUM-based change-point detection (lightweight, online)
2. Bayesian posterior drift monitoring (works with TS posteriors)
3. Sliding-window reward divergence (KL/chi-squared on reward distributions)

References:
  - Catoni-style change-point detection for non-stationary bandits (arXiv 2025)
  - SCAPA-UCB: non-stationary contextual bandits with change/anomaly detection (2025)
  - DoorDash: treatment-effect modelling to reduce non-stationary bias
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ChangeEvent:
    step: int
    zone_id: int
    detector: str   # which method flagged it
    severity: str   # low / medium / high
    metric: float   # the statistic that triggered detection
    threshold: float
    description: str
    recommendation: str


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change-point detector.
    Monitors the mean treatment effect per zone and detects shifts.
    Works online with O(1) per-step computation.
    """

    def __init__(
        self,
        n_zones: int = 10,
        threshold: float = 4.0,
        drift_allowance: float = 0.02,
        min_observations: int = 50,
    ):
        self.n_zones = n_zones
        self.threshold = threshold
        self.drift_allowance = drift_allowance
        self.min_observations = min_observations

        # Per-zone CUSUM statistics
        self.S_pos: Dict[int, float] = {z: 0.0 for z in range(n_zones)}
        self.S_neg: Dict[int, float] = {z: 0.0 for z in range(n_zones)}
        self.running_mean: Dict[int, float] = {z: 0.0 for z in range(n_zones)}
        self.n_obs: Dict[int, int] = {z: 0 for z in range(n_zones)}

    def update(self, zone_id: int, reward: float) -> Optional[ChangeEvent]:
        """
        Feed a new observation. Returns ChangeEvent if shift detected, else None.
        """
        self.n_obs[zone_id] += 1
        n = self.n_obs[zone_id]

        if n < self.min_observations:
            # Update running mean only, no detection yet
            self.running_mean[zone_id] += (reward - self.running_mean[zone_id]) / n
            return None

        # CUSUM update
        deviation = reward - self.running_mean[zone_id]
        self.S_pos[zone_id] = max(0, self.S_pos[zone_id] + deviation - self.drift_allowance)
        self.S_neg[zone_id] = max(0, self.S_neg[zone_id] - deviation - self.drift_allowance)

        stat = max(self.S_pos[zone_id], self.S_neg[zone_id])

        if stat > self.threshold:
            direction = "increase" if self.S_pos[zone_id] > self.S_neg[zone_id] else "decrease"
            severity = "high" if stat > self.threshold * 2 else ("medium" if stat > self.threshold * 1.3 else "low")

            event = ChangeEvent(
                step=n,
                zone_id=zone_id,
                detector="CUSUM",
                severity=severity,
                metric=float(stat),
                threshold=self.threshold,
                description=f"Detected mean {direction} in zone {zone_id}: CUSUM stat={stat:.3f}",
                recommendation=f"Reset priors for zone {zone_id}, increase exploration for 24h",
            )

            # Reset after detection
            self.S_pos[zone_id] = 0.0
            self.S_neg[zone_id] = 0.0
            self.running_mean[zone_id] = reward  # reset mean to current
            return event

        # Slowly adapt mean (exponential moving average for robustness)
        alpha = 0.01
        self.running_mean[zone_id] = (1 - alpha) * self.running_mean[zone_id] + alpha * reward
        return None

    def reset(self, zone_id: Optional[int] = None):
        """Reset detector state."""
        zones = [zone_id] if zone_id is not None else range(self.n_zones)
        for z in zones:
            self.S_pos[z] = 0.0
            self.S_neg[z] = 0.0
            self.running_mean[z] = 0.0
            self.n_obs[z] = 0


class PosteriorDriftMonitor:
    """
    Monitors posterior distribution drift between batches.
    If the posterior mean shifts significantly between consecutive batch updates,
    flags a potential regime change.
    """

    def __init__(
        self,
        n_zones: int = 10,
        drift_threshold: float = 0.15,
        window_size: int = 5,
    ):
        self.n_zones = n_zones
        self.drift_threshold = drift_threshold
        self.window_size = window_size

        # Store posterior means over time
        self.posterior_history: Dict[int, deque] = {
            z: deque(maxlen=window_size) for z in range(n_zones)
        }

    def record_posterior(self, zone_id: int, arm_means: Dict[str, float]):
        """Record current posterior means for a zone (after batch update)."""
        self.posterior_history[zone_id].append(arm_means)

    def check_drift(self, zone_id: int) -> Optional[ChangeEvent]:
        """Check if posterior has drifted significantly since last batch."""
        history = self.posterior_history[zone_id]
        if len(history) < 2:
            return None

        current = history[-1]
        previous = history[-2]

        # Compute L2 norm of drift across all arms
        drift = 0.0
        n_arms = 0
        for arm_id in current:
            if arm_id in previous:
                drift += (current[arm_id] - previous[arm_id]) ** 2
                n_arms += 1

        if n_arms == 0:
            return None

        drift = np.sqrt(drift / n_arms)

        if drift > self.drift_threshold:
            severity = "high" if drift > self.drift_threshold * 2 else "medium"
            return ChangeEvent(
                step=0,
                zone_id=zone_id,
                detector="PosteriorDrift",
                severity=severity,
                metric=float(drift),
                threshold=self.drift_threshold,
                description=f"Posterior drift={drift:.4f} in zone {zone_id} exceeds threshold",
                recommendation=f"Increase exploration weight for zone {zone_id}, consider prior reweighting",
            )

        return None


class SlidingWindowDetector:
    """
    Sliding-window test: compares reward distribution in recent window
    vs historical window using a two-sample test statistic.
    """

    def __init__(
        self,
        n_zones: int = 10,
        window_size: int = 100,
        significance: float = 2.0,
    ):
        self.n_zones = n_zones
        self.window_size = window_size
        self.significance = significance
        self.windows: Dict[int, deque] = {
            z: deque(maxlen=window_size * 2) for z in range(n_zones)
        }

    def update(self, zone_id: int, reward: float) -> Optional[ChangeEvent]:
        """Add observation and check for distribution shift."""
        self.windows[zone_id].append(reward)

        window = self.windows[zone_id]
        if len(window) < self.window_size * 2:
            return None

        data = np.array(window)
        recent = data[-self.window_size:]
        historical = data[:self.window_size]

        # Two-sample Z-test for means
        mean_diff = recent.mean() - historical.mean()
        pooled_std = np.sqrt(
            (recent.var() + historical.var()) / 2 / self.window_size
        )

        if pooled_std < 1e-8:
            return None

        z_stat = abs(mean_diff) / pooled_std

        if z_stat > self.significance:
            direction = "increase" if mean_diff > 0 else "decrease"
            severity = "high" if z_stat > self.significance * 2 else "medium"
            return ChangeEvent(
                step=len(window),
                zone_id=zone_id,
                detector="SlidingWindow",
                severity=severity,
                metric=float(z_stat),
                threshold=self.significance,
                description=f"Reward distribution shift in zone {zone_id}: z={z_stat:.2f}, direction={direction}",
                recommendation=f"Trigger posterior reset or increase decay rate for zone {zone_id}",
            )

        return None


class NonStationarityDetector:
    """
    Composite detector that runs all three methods and aggregates events.
    Replaces the v1 heuristic decay approach with principled detection.
    """

    def __init__(self, n_zones: int = 10):
        self.n_zones = n_zones
        self.cusum = CUSUMDetector(n_zones=n_zones)
        self.drift_monitor = PosteriorDriftMonitor(n_zones=n_zones)
        self.sliding_window = SlidingWindowDetector(n_zones=n_zones)
        self.events: List[Dict] = []
        self.active_alerts: Dict[int, List[ChangeEvent]] = {z: [] for z in range(n_zones)}

    def observe(self, zone_id: int, reward: float, step: int = 0) -> List[ChangeEvent]:
        """Feed observation to all detectors, collect any events."""
        new_events = []

        evt1 = self.cusum.update(zone_id, reward)
        if evt1:
            evt1.step = step
            new_events.append(evt1)

        evt2 = self.sliding_window.update(zone_id, reward)
        if evt2:
            evt2.step = step
            new_events.append(evt2)

        for evt in new_events:
            self.events.append({
                "step": evt.step,
                "zone_id": evt.zone_id,
                "detector": evt.detector,
                "severity": evt.severity,
                "metric": evt.metric,
                "description": evt.description,
                "recommendation": evt.recommendation,
            })
            self.active_alerts[zone_id].append(evt)
            # Keep only recent alerts
            self.active_alerts[zone_id] = self.active_alerts[zone_id][-5:]

        return new_events

    def record_batch_posteriors(self, zone_id: int, arm_means: Dict[str, float], step: int = 0) -> Optional[ChangeEvent]:
        """Record batch posteriors and check for drift."""
        self.drift_monitor.record_posterior(zone_id, arm_means)
        evt = self.drift_monitor.check_drift(zone_id)
        if evt:
            evt.step = step
            self.events.append({
                "step": evt.step,
                "zone_id": evt.zone_id,
                "detector": evt.detector,
                "severity": evt.severity,
                "metric": evt.metric,
                "description": evt.description,
                "recommendation": evt.recommendation,
            })
            return evt
        return None

    def get_state(self) -> Dict:
        """Get detector state for API."""
        return {
            "total_events": len(self.events),
            "recent_events": self.events[-20:],
            "active_alerts_by_zone": {
                str(z): [
                    {"detector": e.detector, "severity": e.severity, "description": e.description}
                    for e in alerts
                ]
                for z, alerts in self.active_alerts.items()
                if alerts
            },
        }
