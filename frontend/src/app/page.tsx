"use client";

import { useState, useCallback, useEffect } from "react";
import {
  Activity,
  BarChart3,
  Brain,
  Bot,
  Play,
  Loader2,
  MapPin,
  TrendingUp,
  Zap,
  FlaskConical,
  Network,
  Shield,
  Users,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { apiFetch } from "@/lib/utils";
import RegretChart from "@/components/RegretChart";
import GPPosteriorChart from "@/components/GPPosteriorChart";
import ZoneHeatmap from "@/components/ZoneHeatmap";
import HTEChart from "@/components/HTEChart";
import ArmStats from "@/components/ArmStats";
import AgentChat from "@/components/AgentChat";
import NetworkGraph from "@/components/NetworkGraph";
import ChangeDetectionPanel from "@/components/ChangeDetectionPanel";
import DoseResponseChart from "@/components/DoseResponseChart";

type TabKey = "overview" | "bandits" | "network" | "bayesian_opt" | "causal" | "system" | "agent" | "swarm";

interface ExperimentResults {
  n_days: number;
  total_steps: number;
  total_records: number;
  vanilla_ts: BanditState;
  contextual_ts: BanditState;
  net_ts: NetBanditState;
  bo: BOState;
  causal: CausalResults;
  batch_summaries: BatchSummary[];
  regret_comparison: RegretComparison;
  change_detection: ChangeDetectionState;
  reward_worker: RewardWorkerState;
  allocator: AllocatorState;
  change_points: ChangePointEntry[];
}

interface NetBanditState {
  algorithm: string;
  cumulative_regret: number;
  regret_history: number[];
  zone_regret: Record<string, number>;
  global_stats: { arm_id: string; total_pulls_all_zones: number; global_mean_norm: number }[];
  zone_stats: Record<string, ArmStat[]>;
  total_pulls: number;
}

interface ChangeDetectionState {
  total_events: number;
  recent_events: { step: number; zone_id: number; detector: string; severity: string; metric: number; description: string; recommendation: string }[];
  active_alerts_by_zone: Record<string, { detector: string; severity: string; description: string }[]>;
}

interface RewardWorkerState {
  batch_count: number;
  total_outcomes_processed: number;
  latest_batch: {
    batch_id: number;
    n_outcomes: number;
    control_rate: number;
    arm_te_estimates: Record<string, number>;
    dr_estimates: Record<string, number> | null;
  } | null;
  history: { batch_id: number; day: number; n_outcomes: number; control_rate: number; arm_te_estimates: Record<string, number> }[];
}

interface AllocatorState {
  constraints: Record<string, number | boolean>;
  global_spend_today: number;
  per_zone_spend: Record<string, number>;
  recent_allocations: Record<string, unknown>[];
}

interface ChangePointEntry {
  day: number;
  zone_id: number;
  zone_name: string;
  shift: number;
  type: string;
}

interface BanditState {
  algorithm: string;
  arms: ArmStat[];
  cumulative_regret: number;
  regret_history: number[];
  total_pulls: number;
}

interface ArmStat {
  arm_id: string;
  mean?: number;
  avg_reward: number;
  n_pulls: number;
  variance?: number;
  uncertainty?: number;
}

interface BOState {
  algorithm: string;
  backend: string;
  total_observations: number;
  iteration: number;
  zones: Record<string, { optimal: { optimal_incentive: number | null; expected_reward: number | null; uncertainty?: number }; n_observations: number }>;
  suggestion_history: { iteration: number; zone_id: number; suggested_incentive: number; method: string }[];
}

interface CausalResults {
  dml?: Record<string, unknown>;
  causal_forest?: Record<string, unknown>;
  hte_segments?: HTESegments;
  cf_hte_segments?: HTESegments;
  dml_error?: string;
  cf_error?: string;
  dose_response?: Record<string, unknown>;
  dose_response_curve?: { fitted: boolean; grid?: number[]; response?: number[]; std?: number[]; lower?: number[]; upper?: number[] };
  dose_response_error?: string;
  interference?: Record<string, number>;
  interference_by_zone?: Record<string, Record<string, number>>;
  interference_error?: string;
}

interface HTESegments {
  by_tenure: HTESeg[];
  by_deficit: HTESeg[];
  by_peak: HTESeg[];
  by_responsiveness: HTESeg[];
  overall: { mean_cate: number; std_cate: number; median_cate: number; p10_cate: number; p90_cate: number };
}

interface HTESeg {
  tenure_bucket?: string;
  deficit_bucket?: string;
  resp_bucket?: string;
  is_peak?: number;
  mean: number;
  std: number;
  count: number;
}

interface BatchSummary {
  day: number;
  n_steps: number;
  vanilla_ts: { avg_reward: number; total_reward: number; cumulative_regret: number };
  contextual_ts: { avg_reward: number; total_reward: number; cumulative_regret: number };
  net_ts: { avg_reward: number; total_reward: number; cumulative_regret: number };
  bo: { avg_reward: number; total_reward: number };
}

interface SwarmResults {
  n_days: number;
  n_dashers: number;
  n_zones: number;
  total_steps: number;
  aggregate: {
    mean_acceptance_rate: number;
    std_acceptance_rate: number;
    total_offers: number;
    total_accepted: number;
    total_zone_switches: number;
    acceptance_trend: number[];
  };
  daily_summaries: {
    day: number;
    total_offers: number;
    total_accepted: number;
    acceptance_rate: number;
    zone_switches: number;
    zone_acceptance_rates: Record<string, number>;
  }[];
  final_zones: {
    zone_id: number;
    name: string;
    current_demand: number;
    current_incentive: number;
    dasher_count: number;
    total_offers: number;
    total_accepts: number;
    acceptance_rate: number;
  }[];
}

interface RegretComparison {
  steps: number[];
  vanilla_ts_regret: number[];
  contextual_ts_regret: number[];
  net_ts_regret: number[];
  regret_reduction_pct: number;
  net_ts_reduction_pct: number;
}

interface SimSnapshot {
  day: number;
  hour: number;
  total_steps: number;
  zones: ZoneSnap[];
  n_dashers: number;
  clusters: Record<string, number[]>;
  change_points: ChangePointEntry[];
  pending_outcomes: number;
}

interface ZoneSnap {
  zone_id: number;
  name: string;
  cluster_id: number;
  supply_deficit: number;
  effective_elasticity: number;
  is_peak: boolean;
  demand: number;
  supply: number;
  spillover_pressure: number;
  daily_spend: number;
  total_spend: number;
}

interface AdjacencyData {
  adjacency: number[][];
  clusters: Record<string, number[]>;
}

const TABS: { key: TabKey; label: string; icon: React.ReactNode }[] = [
  { key: "overview", label: "Overview", icon: <Activity className="h-4 w-4" /> },
  { key: "bandits", label: "Bandits", icon: <BarChart3 className="h-4 w-4" /> },
  { key: "network", label: "Network", icon: <Network className="h-4 w-4" /> },
  { key: "bayesian_opt", label: "Bayesian Opt", icon: <TrendingUp className="h-4 w-4" /> },
  { key: "causal", label: "Causal ML", icon: <FlaskConical className="h-4 w-4" /> },
  { key: "system", label: "System", icon: <Shield className="h-4 w-4" /> },
  { key: "agent", label: "LLM Agent", icon: <Bot className="h-4 w-4" /> },
  { key: "swarm", label: "Swarm", icon: <Users className="h-4 w-4" /> },
];

export default function Home() {
  const [tab, setTab] = useState<TabKey>("overview");
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [snapshot, setSnapshot] = useState<SimSnapshot | null>(null);
  const [posteriorData, setPosteriorData] = useState<Record<number, unknown>>({});
  const [adjacencyData, setAdjacencyData] = useState<AdjacencyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [nDays, setNDays] = useState(14);
  const [error, setError] = useState<string | null>(null);
  const [swarmData, setSwarmData] = useState<SwarmResults | null>(null);
  const [swarmLoading, setSwarmLoading] = useState(false);
  const [viewingCache, setViewingCache] = useState(false);
  const [cacheTimestamp, setCacheTimestamp] = useState<string | null>(null);
  const [swarmFromCache, setSwarmFromCache] = useState(false);
  const [swarmCacheTimestamp, setSwarmCacheTimestamp] = useState<string | null>(null);

  // On mount, auto-hydrate from the on-disk cache so a freshly cloned repo
  // shows prior experiment + swarm results without having to run anything.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const snap = await apiFetch("/cache/experiment");
        if (cancelled || !snap) return;
        if (snap.experiment_results) setResults(snap.experiment_results);
        if (snap.sim_snapshot) setSnapshot(snap.sim_snapshot);
        if (snap.adjacency) setAdjacencyData(snap.adjacency);
        if (snap.bo_posteriors) {
          const post: Record<number, unknown> = {};
          for (const [k, v] of Object.entries(snap.bo_posteriors)) post[Number(k)] = v;
          setPosteriorData(post);
        }
        if (snap.experiment_results?.n_days) setNDays(snap.experiment_results.n_days);
        setViewingCache(true);
        setCacheTimestamp(snap.saved_at_iso || null);
      } catch {
        /* no cache yet — leave empty */
      }
    })();

    (async () => {
      try {
        const swm = await apiFetch("/cache/swarm");
        if (cancelled || !swm?.result) return;
        setSwarmData(swm.result);
        setSwarmFromCache(true);
        setSwarmCacheTimestamp(swm.saved_at_iso || null);
      } catch {
        /* no cached swarm yet */
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const runExperiment = useCallback(async () => {
    setLoading(true);
    setError(null);
    setViewingCache(false);
    setCacheTimestamp(null);
    try {
      const res: ExperimentResults = await apiFetch("/experiment/run", {
        method: "POST",
        body: JSON.stringify({ n_days: nDays, n_dashers_per_zone: 10, seed: 42 }),
      });
      setResults(res);

      // Fetch snapshot
      const snap: SimSnapshot = await apiFetch("/simulation/snapshot");
      setSnapshot(snap);

      // Fetch GP posteriors for first 3 zones
      const posteriors: Record<number, unknown> = {};
      for (const zoneId of [0, 1, 2]) {
        try {
          const p = await apiFetch("/bo/posterior", {
            method: "POST",
            body: JSON.stringify({ zone_id: zoneId, n_points: 100 }),
          });
          posteriors[zoneId] = p;
        } catch {
          // Zone may not have enough data
        }
      }
      setPosteriorData(posteriors);

      // Fetch adjacency matrix
      try {
        const adj: AdjacencyData = await apiFetch("/simulation/adjacency");
        setAdjacencyData(adj);
      } catch {
        // non-critical
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run experiment");
    } finally {
      setLoading(false);
    }
  }, [nDays]);

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Header */}
      <header className="border-b border-[var(--border)] bg-[var(--card)]">
        <div className="mx-auto max-w-7xl px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--primary)]">
              <Zap className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl tracking-tight" style={{ fontFamily: "var(--font-instrument-serif)" }}>
                DorkDash
              </h1>
              <p className="text-xs text-[var(--muted-foreground)]">
                Agentic Causal Bandit Platform · DoorDash Supply Team
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <label className="text-xs text-[var(--muted-foreground)]">Days:</label>
              <input
                type="number"
                min={1}
                max={60}
                value={nDays}
                onChange={(e) => setNDays(parseInt(e.target.value) || 14)}
                className="w-16 rounded-md bg-[var(--muted)] border border-[var(--border)] px-2 py-1.5 text-xs text-center outline-none focus:border-[var(--primary)]"
              />
            </div>
            <button
              onClick={runExperiment}
              disabled={loading}
              className="flex items-center gap-2 rounded-lg bg-[var(--primary)] px-4 py-2 text-sm font-medium text-white hover:bg-[var(--primary)]/80 disabled:opacity-50 transition-colors"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {loading ? "Running..." : "Run Experiment"}
            </button>
          </div>
        </div>
      </header>

      {/* Cached-result banner */}
      {(viewingCache || (tab === "swarm" && swarmFromCache)) && (
        <div className="border-b border-[var(--border)] bg-amber-500/10">
          <div className="mx-auto max-w-7xl px-4 py-2 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2 text-amber-600">
              <Activity className="h-3.5 w-3.5" />
              <span>
                {tab === "swarm" && swarmFromCache
                  ? `Viewing cached swarm result${swarmCacheTimestamp ? ` from ${swarmCacheTimestamp}` : ""}`
                  : `Viewing cached experiment result${cacheTimestamp ? ` from ${cacheTimestamp}` : ""}`}
                {" "}— run a new experiment to refresh.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-[var(--border)] bg-[var(--card)]">
        <div className="mx-auto max-w-7xl px-4">
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  tab === t.key
                    ? "border-[var(--primary)] text-[var(--primary)]"
                    : "border-transparent text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
                }`}
              >
                {t.icon}
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-auto max-w-7xl px-4 pt-4">
          <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-sm text-red-400">
            {error}
          </div>
        </div>
      )}

      {/* Content */}
      <main className="mx-auto max-w-7xl px-4 py-6">
        {tab === "overview" && (
          <OverviewTab results={results} snapshot={snapshot} />
        )}
        {tab === "bandits" && <BanditsTab results={results} />}
        {tab === "network" && (
          <NetworkTab results={results} snapshot={snapshot} adjacencyData={adjacencyData} />
        )}
        {tab === "bayesian_opt" && (
          <BOTab results={results} posteriorData={posteriorData} />
        )}
        {tab === "causal" && <CausalTab results={results} />}
        {tab === "system" && <SystemTab results={results} />}
        {tab === "agent" && <AgentTab />}
        {tab === "swarm" && (
          <SwarmTab
            swarmData={swarmData}
            loading={swarmLoading}
            onRunSwarm={async (nDashers: number, nDays: number) => {
              setSwarmLoading(true);
              setSwarmFromCache(false);
              setSwarmCacheTimestamp(null);
              try {
                const res = await apiFetch("/swarm/run", {
                  method: "POST",
                  body: JSON.stringify({ n_dashers: nDashers, n_zones: 10, n_days: nDays, seed: 42 }),
                });
                setSwarmData(res);
              } catch (e) {
                setError(e instanceof Error ? e.message : "Swarm failed");
              } finally {
                setSwarmLoading(false);
              }
            }}
          />
        )}
      </main>
    </div>
  );
}

/* ============= Tab Components ============= */

function OverviewTab({
  results,
  snapshot,
}: {
  results: ExperimentResults | null;
  snapshot: SimSnapshot | null;
}) {
  if (!results) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <Brain className="h-16 w-16 text-[var(--muted-foreground)] mb-4" />
        <h2 className="text-xl font-semibold mb-2">Ready to Run</h2>
        <p className="text-[var(--muted-foreground)] max-w-md">
          Click &quot;Run Experiment&quot; to simulate a multi-day comparison of Vanilla Thompson
          Sampling vs. Contextual LinTS vs. Networked LinTS vs. Shape-Constrained BO
          across 10 DoorDash zones with interference, delayed feedback, and regime changes.
        </p>
      </div>
    );
  }

  const regretPct = results.regret_comparison.regret_reduction_pct;
  const netPct = results.regret_comparison.net_ts_reduction_pct || 0;
  const boZones = Object.entries(results.bo.zones || {});
  const avgSaving = boZones.length > 0
    ? boZones.reduce((s, [, z]) => s + (z.optimal?.optimal_incentive || 0), 0) / boZones.length
    : 0;

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-5">
            <div className="text-xs text-[var(--muted-foreground)]">Experiment Duration</div>
            <div className="text-2xl font-bold mt-1">{results.n_days} days</div>
            <div className="text-xs text-[var(--muted-foreground)]">{results.total_records.toLocaleString()} observations</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5">
            <div className="text-xs text-[var(--muted-foreground)]">NetTS Regret Reduction</div>
            <div className="text-2xl font-bold mt-1 text-cyan-400">{netPct.toFixed(1)}%</div>
            <div className="text-xs text-[var(--muted-foreground)]">Networked vs Vanilla TS</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5">
            <div className="text-xs text-[var(--muted-foreground)]">BO Optimal Incentive</div>
            <div className="text-2xl font-bold mt-1 text-blue-400">${avgSaving.toFixed(2)}</div>
            <div className="text-xs text-[var(--muted-foreground)]">avg across zones</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5">
            <div className="text-xs text-[var(--muted-foreground)]">Change Events</div>
            <div className="text-2xl font-bold mt-1 text-yellow-400">
              {results.change_detection?.total_events || 0}
            </div>
            <div className="text-xs text-[var(--muted-foreground)]">
              {results.change_points?.length || 0} regime changes
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Zone heatmap */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="h-4 w-4" /> Zone Supply/Demand Balance
          </CardTitle>
          <CardDescription>
            Live supply deficit across 10 Bay Area zones (Day {snapshot?.day}, Hour {snapshot?.hour})
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ZoneHeatmap zones={snapshot?.zones || []} />
        </CardContent>
      </Card>

      {/* Regret comparison */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" /> Cumulative Regret Comparison
          </CardTitle>
          <CardDescription>
            Vanilla TS vs. Contextual LinTS vs. Networked LinTS over {results.n_days} days
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RegretChart data={results.regret_comparison} />
        </CardContent>
      </Card>

      {/* Daily batch summaries */}
      <Card>
        <CardHeader>
          <CardTitle>Daily Performance Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-2 px-3 text-[var(--muted-foreground)]">Day</th>
                  <th className="text-right py-2 px-3 text-orange-400">VTS Reward</th>
                  <th className="text-right py-2 px-3 text-orange-400">VTS Regret</th>
                  <th className="text-right py-2 px-3 text-emerald-400">CTS Reward</th>
                  <th className="text-right py-2 px-3 text-emerald-400">CTS Regret</th>
                  <th className="text-right py-2 px-3 text-cyan-400">NetTS Reward</th>
                  <th className="text-right py-2 px-3 text-cyan-400">NetTS Regret</th>
                  <th className="text-right py-2 px-3 text-blue-400">BO Reward</th>
                </tr>
              </thead>
              <tbody>
                {results.batch_summaries.map((b, i) => (
                  <tr key={i} className="border-b border-[var(--border)]/50 hover:bg-[var(--muted)]/50">
                    <td className="py-2 px-3">{b.day}</td>
                    <td className="py-2 px-3 text-right tabular-nums">{b.vanilla_ts.avg_reward.toFixed(3)}</td>
                    <td className="py-2 px-3 text-right tabular-nums">{b.vanilla_ts.cumulative_regret.toFixed(1)}</td>
                    <td className="py-2 px-3 text-right tabular-nums">{b.contextual_ts.avg_reward.toFixed(3)}</td>
                    <td className="py-2 px-3 text-right tabular-nums">{b.contextual_ts.cumulative_regret.toFixed(1)}</td>
                    <td className="py-2 px-3 text-right tabular-nums">{b.net_ts?.avg_reward?.toFixed(3) || "—"}</td>
                    <td className="py-2 px-3 text-right tabular-nums">{b.net_ts?.cumulative_regret?.toFixed(1) || "—"}</td>
                    <td className="py-2 px-3 text-right tabular-nums">{b.bo.avg_reward.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function BanditsTab({ results }: { results: ExperimentResults | null }) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" /> Bandit Algorithm Comparison
          </CardTitle>
          <CardDescription>
            Vanilla TS (baseline) vs. Contextual LinTS vs. Networked LinTS (global+local decomposition)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ArmStats
            vanilla={results?.vanilla_ts || null}
            contextual={results?.contextual_ts || null}
          />
          {results?.net_ts && (
            <div className="mt-4 rounded-lg bg-cyan-500/5 border border-cyan-500/20 p-4">
              <h4 className="text-sm font-medium text-cyan-400 mb-2">Networked LinTS (NetLinTS)</h4>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div>
                  <div className="text-[var(--muted-foreground)]">Total Pulls</div>
                  <div className="text-lg font-bold">{results.net_ts.total_pulls}</div>
                </div>
                <div>
                  <div className="text-[var(--muted-foreground)]">Cumulative Regret</div>
                  <div className="text-lg font-bold">{results.net_ts.cumulative_regret.toFixed(1)}</div>
                </div>
                <div>
                  <div className="text-[var(--muted-foreground)]">Zones</div>
                  <div className="text-lg font-bold">{Object.keys(results.net_ts.zone_regret).length}</div>
                </div>
              </div>
              {results.net_ts.global_stats && (
                <div className="mt-3 flex gap-2 flex-wrap">
                  {results.net_ts.global_stats.map((gs) => (
                    <span key={gs.arm_id} className="rounded bg-cyan-500/10 px-2 py-1 text-[10px]">
                      {gs.arm_id}: {gs.total_pulls_all_zones} pulls
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Cumulative Regret Over Time</CardTitle>
          <CardDescription>
            Lower is better — networked bandits share information across zones for faster convergence
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RegretChart data={results?.regret_comparison || null} />
        </CardContent>
      </Card>
    </div>
  );
}

function NetworkTab({
  results,
  snapshot,
  adjacencyData,
}: {
  results: ExperimentResults | null;
  snapshot: SimSnapshot | null;
  adjacencyData: AdjacencyData | null;
}) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-4 w-4" /> Zone Network &amp; Interference
          </CardTitle>
          <CardDescription>
            Geographic zone clusters, adjacency weights, and spillover pressure.
            Incentivising one zone pulls Dashers from neighbouring zones.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <NetworkGraph
            zones={snapshot?.zones || []}
            adjacency={adjacencyData?.adjacency || []}
            clusters={adjacencyData?.clusters || snapshot?.clusters || {}}
          />
        </CardContent>
      </Card>

      {/* Per-zone regret for NetLinTS */}
      {results?.net_ts?.zone_regret && (
        <Card>
          <CardHeader>
            <CardTitle>Per-Zone Regret (NetLinTS)</CardTitle>
            <CardDescription>
              How much regret accumulated in each zone — zones with more interference or drift accumulate more
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {Object.entries(results.net_ts.zone_regret).map(([zoneId, regret]) => {
                const zone = snapshot?.zones.find((z) => z.zone_id === parseInt(zoneId));
                return (
                  <div key={zoneId} className="rounded-lg bg-cyan-500/10 border border-cyan-500/20 p-3 text-center">
                    <div className="text-[10px] text-cyan-300/70 truncate">{zone?.name || `Zone ${zoneId}`}</div>
                    <div className="text-lg font-bold text-cyan-400">{(regret as number).toFixed(1)}</div>
                    <div className="text-[10px] text-cyan-300/50">regret</div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function SystemTab({ results }: { results: ExperimentResults | null }) {
  return (
    <div className="space-y-6">
      {/* Change Detection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-4 w-4" /> Non-Stationarity Detection
          </CardTitle>
          <CardDescription>
            CUSUM + Sliding Window + Posterior Drift monitoring detects regime changes in real-time.
            Ground truth changes shown alongside detected events.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChangeDetectionPanel
            changeDetection={results?.change_detection || null}
            changePoints={results?.change_points || []}
          />
        </CardContent>
      </Card>

      {/* Reward Worker */}
      {results?.reward_worker && (
        <Card>
          <CardHeader>
            <CardTitle>Batch Reward Worker</CardTitle>
            <CardDescription>
              Treatment-effect relative to control (avoids Simpson&apos;s paradox under non-stationarity)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="rounded-lg bg-[var(--muted)] p-3">
                <div className="text-xs text-[var(--muted-foreground)]">Batches Processed</div>
                <div className="text-xl font-bold">{results.reward_worker.batch_count}</div>
              </div>
              <div className="rounded-lg bg-[var(--muted)] p-3">
                <div className="text-xs text-[var(--muted-foreground)]">Total Outcomes</div>
                <div className="text-xl font-bold">{results.reward_worker.total_outcomes_processed?.toLocaleString()}</div>
              </div>
              <div className="rounded-lg bg-[var(--muted)] p-3">
                <div className="text-xs text-[var(--muted-foreground)]">Control Rate</div>
                <div className="text-xl font-bold">
                  {(results.reward_worker.latest_batch?.control_rate ?? 0 * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Treatment effect estimates */}
            {results.reward_worker.latest_batch?.arm_te_estimates && (
              <div>
                <h4 className="text-sm font-medium mb-2">Latest Batch: Treatment Effects (vs Control)</h4>
                <div className="flex gap-2 flex-wrap">
                  {Object.entries(results.reward_worker.latest_batch.arm_te_estimates).map(([arm, te]) => (
                    <div key={arm} className="rounded bg-emerald-500/10 border border-emerald-500/20 px-3 py-2 text-center">
                      <div className="text-xs text-emerald-300/70">{arm}</div>
                      <div className={`text-sm font-bold ${te > 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {te > 0 ? "+" : ""}{(te * 100).toFixed(2)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Allocator Constraints */}
      {results?.allocator && (
        <Card>
          <CardHeader>
            <CardTitle>Constraint-Aware Allocator</CardTitle>
            <CardDescription>
              DISCO-style budget caps, per-Dasher frequency limits, exploration/control floors, sticky assignment
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
              {Object.entries(results.allocator.constraints || {}).map(([key, val]) => (
                <div key={key} className="rounded-lg bg-[var(--muted)] p-3">
                  <div className="text-[var(--muted-foreground)]">{key.replace(/_/g, " ")}</div>
                  <div className="text-sm font-bold mt-1">{typeof val === "boolean" ? (val ? "Yes" : "No") : String(val)}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function BOTab({
  results,
  posteriorData,
}: {
  results: ExperimentResults | null;
  posteriorData: Record<number, unknown>;
}) {
  const zoneNames = [
    "Downtown SF", "Mission District", "SoMa", "Marina", "Sunset",
    "Richmond", "Oakland Downtown", "Berkeley", "San Jose Central", "Palo Alto",
  ];

  return (
    <div className="space-y-6">
      {/* Optimal incentives by zone */}
      {results?.bo && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" /> Optimal Incentive by Zone
            </CardTitle>
            <CardDescription>
              Bayesian Optimization finds the continuous optimum — no discrete arm guessing
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {Object.entries(results.bo.zones || {}).map(([zoneId, data]) => (
                <div key={zoneId} className="rounded-lg bg-blue-500/10 border border-blue-500/20 p-3 text-center">
                  <div className="text-[10px] text-blue-300/70 truncate">{zoneNames[parseInt(zoneId)] || `Zone ${zoneId}`}</div>
                  <div className="text-xl font-bold text-blue-400 mt-1">
                    ${data.optimal?.optimal_incentive?.toFixed(2) || "—"}
                  </div>
                  <div className="text-[10px] text-blue-300/50">{data.n_observations} obs</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* GP Posterior plots */}
      {[0, 1, 2].map((zoneId) => (
        <Card key={zoneId}>
          <CardHeader>
            <CardTitle>GP Posterior — {zoneNames[zoneId]}</CardTitle>
            <CardDescription>
              Gaussian Process model of incentive → treatment effect with 95% confidence interval
            </CardDescription>
          </CardHeader>
          <CardContent>
            <GPPosteriorChart
              data={posteriorData[zoneId] as Parameters<typeof GPPosteriorChart>[0]["data"]}
              zoneName={zoneNames[zoneId]}
            />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function CausalTab({ results }: { results: ExperimentResults | null }) {
  const hteData = results?.causal?.hte_segments || results?.causal?.cf_hte_segments || null;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4" /> Heterogeneous Treatment Effects
          </CardTitle>
          <CardDescription>
            Double Machine Learning estimates of who responds most to incentives — by tenure, zone deficit, responsiveness, and peak timing
          </CardDescription>
        </CardHeader>
        <CardContent>
          <HTEChart data={hteData as Parameters<typeof HTEChart>[0]["data"]} />
        </CardContent>
      </Card>

      {/* Model diagnostics */}
      {results?.causal?.dml && (
        <Card>
          <CardHeader>
            <CardTitle>Model Diagnostics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg bg-[var(--muted)] p-4">
                <h4 className="text-sm font-medium mb-2">Double Machine Learning</h4>
                <pre className="text-xs text-[var(--muted-foreground)] whitespace-pre-wrap">
                  {JSON.stringify(results.causal.dml, null, 2)}
                </pre>
              </div>
              {results.causal.causal_forest && (
                <div className="rounded-lg bg-[var(--muted)] p-4">
                  <h4 className="text-sm font-medium mb-2">Causal Forest</h4>
                  <pre className="text-xs text-[var(--muted-foreground)] whitespace-pre-wrap">
                    {JSON.stringify(results.causal.causal_forest, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Dose-Response Curve */}
      <Card>
        <CardHeader>
          <CardTitle>Dose-Response Curve</CardTitle>
          <CardDescription>
            Non-parametric estimate of E[response | incentive = t] — the causal curve mapping incentive amount to expected response probability
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DoseResponseChart
            data={results?.causal?.dose_response_curve as Parameters<typeof DoseResponseChart>[0]["data"] || null}
          />
        </CardContent>
      </Card>

      {/* Interference Effects */}
      {results?.causal?.interference && (
        <Card>
          <CardHeader>
            <CardTitle>Interference / Spillover Effects</CardTitle>
            <CardDescription>
              Decomposition of treatment effect into direct (own-zone incentive) and indirect (neighbour spillover) components
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-4 text-center">
                <div className="text-xs text-emerald-300/70">Direct Effect</div>
                <div className="text-xl font-bold text-emerald-400">
                  {((results.causal.interference as Record<string, number>).direct_effect * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] text-emerald-300/50">per $ incentive</div>
              </div>
              <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4 text-center">
                <div className="text-xs text-red-300/70">Indirect (Spillover)</div>
                <div className="text-xl font-bold text-red-400">
                  {((results.causal.interference as Record<string, number>).indirect_effect * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] text-red-300/50">from neighbours</div>
              </div>
              <div className="rounded-lg bg-yellow-500/10 border border-yellow-500/20 p-4 text-center">
                <div className="text-xs text-yellow-300/70">Spillover Ratio</div>
                <div className="text-xl font-bold text-yellow-400">
                  {((results.causal.interference as Record<string, number>).spillover_ratio * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-yellow-300/50">|indirect| / |direct|</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {results?.causal?.dml_error && (
        <Card>
          <CardContent className="pt-5">
            <div className="text-sm text-yellow-400">
              DML Error: {results.causal.dml_error}
            </div>
            <div className="text-xs text-[var(--muted-foreground)] mt-1">
              Install econml for full causal estimation: pip install econml
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function AgentTab() {
  return (
    <div className="grid grid-cols-1 gap-6">
      <Card className="h-[calc(100vh-260px)] flex flex-col">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-4 w-4" /> Experiment Management Agent
          </CardTitle>
          <CardDescription>
            GPT-4o agent with function-calling access to bandit state, BO posteriors, and causal insights.
            Ask it to summarize results, propose arm changes, or detect non-stationarity.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden">
          <AgentChat />
        </CardContent>
      </Card>
    </div>
  );
}

function SwarmTab({
  swarmData,
  loading,
  onRunSwarm,
}: {
  swarmData: SwarmResults | null;
  loading: boolean;
  onRunSwarm: (nDashers: number, nDays: number) => void;
}) {
  const [nDashers, setNDashers] = useState(50);
  const [nDays, setNDays] = useState(3);

  if (!swarmData && !loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <Users className="h-16 w-16 text-[var(--muted-foreground)] mb-4" />
        <h2 className="text-xl font-semibold mb-2">Multi-Agent Swarm Simulation</h2>
        <p className="text-[var(--muted-foreground)] max-w-lg mb-6">
          Simulate a swarm of LLM-powered Dasher agents that autonomously accept or reject
          incentive offers based on their unique personas, memory, and marketplace conditions.
        </p>
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <label className="text-xs text-[var(--muted-foreground)]">Dashers:</label>
            <input
              type="number" min={10} max={500} value={nDashers}
              onChange={(e) => setNDashers(parseInt(e.target.value) || 50)}
              className="w-20 rounded-md bg-[var(--muted)] border border-[var(--border)] px-2 py-1.5 text-xs text-center outline-none focus:border-[var(--primary)]"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-[var(--muted-foreground)]">Days:</label>
            <input
              type="number" min={1} max={30} value={nDays}
              onChange={(e) => setNDays(parseInt(e.target.value) || 3)}
              className="w-16 rounded-md bg-[var(--muted)] border border-[var(--border)] px-2 py-1.5 text-xs text-center outline-none focus:border-[var(--primary)]"
            />
          </div>
        </div>
        <button
          onClick={() => onRunSwarm(nDashers, nDays)}
          className="flex items-center gap-2 rounded-lg bg-purple-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-purple-500 transition-colors"
        >
          <Play className="h-4 w-4" />
          Run Swarm Simulation
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <Loader2 className="h-12 w-12 animate-spin text-purple-400 mb-4" />
        <p className="text-[var(--muted-foreground)]">
          Simulating {nDashers} Dasher agents over {nDays} days...
        </p>
      </div>
    );
  }

  if (!swarmData) return null;

  const agg = swarmData.aggregate;

  return (
    <div className="grid grid-cols-1 gap-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="pt-5 text-center">
            <div className="text-xs text-[var(--muted-foreground)]">Agents</div>
            <div className="text-2xl font-bold">{swarmData.n_dashers}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5 text-center">
            <div className="text-xs text-[var(--muted-foreground)]">Zones</div>
            <div className="text-2xl font-bold">{swarmData.n_zones}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5 text-center">
            <div className="text-xs text-[var(--muted-foreground)]">Acceptance Rate</div>
            <div className="text-2xl font-bold text-green-400">
              {(agg.mean_acceptance_rate * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-[var(--muted-foreground)]">
              ±{(agg.std_acceptance_rate * 100).toFixed(1)}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5 text-center">
            <div className="text-xs text-[var(--muted-foreground)]">Total Offers</div>
            <div className="text-2xl font-bold">{agg.total_offers.toLocaleString()}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5 text-center">
            <div className="text-xs text-[var(--muted-foreground)]">Zone Switches</div>
            <div className="text-2xl font-bold text-yellow-400">{agg.total_zone_switches}</div>
          </CardContent>
        </Card>
      </div>

      {/* Acceptance Trend */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Acceptance Rate Over Time (per step)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48 flex items-end gap-[2px]">
            {agg.acceptance_trend.map((rate, i) => (
              <div
                key={i}
                className="flex-1 rounded-t transition-all"
                style={{
                  height: `${rate * 100}%`,
                  backgroundColor: rate > 0.7
                    ? "rgb(34, 197, 94)"
                    : rate > 0.4
                    ? "rgb(234, 179, 8)"
                    : "rgb(239, 68, 68)",
                  opacity: 0.7 + rate * 0.3,
                }}
                title={`Step ${i + 1}: ${(rate * 100).toFixed(1)}%`}
              />
            ))}
          </div>
          <div className="flex justify-between text-[10px] text-[var(--muted-foreground)] mt-1">
            <span>Step 1</span>
            <span>Step {agg.acceptance_trend.length}</span>
          </div>
        </CardContent>
      </Card>

      {/* Zone Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Zone Performance (Final State)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-2 px-2 text-[var(--muted-foreground)]">Zone</th>
                  <th className="text-right py-2 px-2 text-[var(--muted-foreground)]">Incentive</th>
                  <th className="text-right py-2 px-2 text-[var(--muted-foreground)]">Demand</th>
                  <th className="text-right py-2 px-2 text-[var(--muted-foreground)]">Dashers</th>
                  <th className="text-right py-2 px-2 text-[var(--muted-foreground)]">Offers</th>
                  <th className="text-right py-2 px-2 text-[var(--muted-foreground)]">Accepts</th>
                  <th className="text-right py-2 px-2 text-[var(--muted-foreground)]">Rate</th>
                </tr>
              </thead>
              <tbody>
                {swarmData.final_zones.map((z) => (
                  <tr key={z.zone_id} className="border-b border-[var(--border)]/50 hover:bg-[var(--muted)]/30">
                    <td className="py-1.5 px-2 font-medium">{z.name}</td>
                    <td className="py-1.5 px-2 text-right text-green-400">${z.current_incentive.toFixed(2)}</td>
                    <td className="py-1.5 px-2 text-right">{(z.current_demand * 100).toFixed(0)}%</td>
                    <td className="py-1.5 px-2 text-right">{z.dasher_count}</td>
                    <td className="py-1.5 px-2 text-right">{z.total_offers}</td>
                    <td className="py-1.5 px-2 text-right">{z.total_accepts}</td>
                    <td className="py-1.5 px-2 text-right font-medium" style={{
                      color: z.acceptance_rate > 0.7 ? "rgb(34, 197, 94)" : z.acceptance_rate > 0.4 ? "rgb(234, 179, 8)" : "rgb(239, 68, 68)"
                    }}>
                      {(z.acceptance_rate * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Daily Summaries */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Daily Summaries</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {swarmData.daily_summaries.map((d) => (
              <div key={d.day} className="rounded-lg bg-[var(--muted)]/40 border border-[var(--border)] p-3">
                <div className="text-xs font-semibold mb-1">Day {d.day}</div>
                <div className="text-[10px] text-[var(--muted-foreground)] space-y-0.5">
                  <div>Offers: {d.total_offers} · Accepted: {d.total_accepted}</div>
                  <div>
                    Rate:{" "}
                    <span style={{
                      color: d.acceptance_rate > 0.7 ? "rgb(34, 197, 94)" : d.acceptance_rate > 0.4 ? "rgb(234, 179, 8)" : "rgb(239, 68, 68)"
                    }}>
                      {(d.acceptance_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>Zone switches: {d.zone_switches}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Run Again */}
      <div className="flex justify-center">
        <button
          onClick={() => onRunSwarm(nDashers, nDays)}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-purple-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-purple-500 disabled:opacity-50 transition-colors"
        >
          <Play className="h-4 w-4" />
          Re-run Swarm
        </button>
      </div>
    </div>
  );
}
