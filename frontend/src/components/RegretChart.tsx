"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface RegretData {
  steps: number[];
  vanilla_ts_regret: number[];
  contextual_ts_regret: number[];
  net_ts_regret?: number[];
  regret_reduction_pct: number;
  net_ts_reduction_pct?: number;
}

export default function RegretChart({ data }: { data: RegretData | null }) {
  if (!data || !data.steps.length) {
    return (
      <div className="flex items-center justify-center h-64 text-[var(--muted-foreground)]">
        Run an experiment to see regret comparison
      </div>
    );
  }

  const hasNet = data.net_ts_regret && data.net_ts_regret.length > 0;
  const chartData = data.steps.map((step, i) => ({
    step,
    "Vanilla TS": data.vanilla_ts_regret[i],
    "Contextual LinTS": data.contextual_ts_regret[i],
    ...(hasNet ? { "Networked LinTS": data.net_ts_regret![i] } : {}),
  }));

  return (
    <div>
      <div className="mb-3 flex items-center gap-4 flex-wrap">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400">
          {data.regret_reduction_pct.toFixed(1)}% CTS reduction
        </span>
        {data.net_ts_reduction_pct !== undefined && (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-400">
            {data.net_ts_reduction_pct.toFixed(1)}% NetTS reduction
          </span>
        )}
        <span className="text-xs text-[var(--muted-foreground)]">
          vs. vanilla Thompson Sampling
        </span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="step"
            stroke="var(--muted-foreground)"
            fontSize={11}
            tickLine={false}
          />
          <YAxis
            stroke="var(--muted-foreground)"
            fontSize={11}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="Vanilla TS"
            stroke="#f97316"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="Contextual LinTS"
            stroke="#22c55e"
            strokeWidth={2}
            dot={false}
          />
          {hasNet && (
            <Line
              type="monotone"
              dataKey="Networked LinTS"
              stroke="#06b6d4"
              strokeWidth={2}
              dot={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
