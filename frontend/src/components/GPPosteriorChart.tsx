"use client";

import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ComposedChart,
} from "recharts";

interface PosteriorData {
  incentive_amounts: number[];
  mean: number[];
  std: number[];
  lower: number[];
  upper: number[];
  observations_x: number[];
  observations_y: number[];
}

export default function GPPosteriorChart({
  data,
  zoneName,
}: {
  data: PosteriorData | null;
  zoneName?: string;
}) {
  if (!data || !data.incentive_amounts.length) {
    return (
      <div className="flex items-center justify-center h-64 text-[var(--muted-foreground)]">
        No posterior data available yet
      </div>
    );
  }

  const chartData = data.incentive_amounts.map((x, i) => ({
    incentive: parseFloat(x.toFixed(2)),
    mean: parseFloat(data.mean[i].toFixed(4)),
    lower: parseFloat(data.lower[i].toFixed(4)),
    upper: parseFloat(data.upper[i].toFixed(4)),
    range: [
      parseFloat(data.lower[i].toFixed(4)),
      parseFloat(data.upper[i].toFixed(4)),
    ],
  }));

  // Find optimal
  const bestIdx = data.mean.indexOf(Math.max(...data.mean));
  const optimalIncentive = data.incentive_amounts[bestIdx];

  return (
    <div>
      <div className="mb-3 flex items-center gap-4">
        {zoneName && (
          <span className="text-xs font-medium text-[var(--muted-foreground)]">
            {zoneName}
          </span>
        )}
        <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-400">
          Optimal: ${optimalIncentive.toFixed(2)}
        </span>
        <span className="text-xs text-[var(--muted-foreground)]">
          {data.observations_x.length} observations
        </span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="incentive"
            stroke="var(--muted-foreground)"
            fontSize={11}
            label={{
              value: "Incentive ($)",
              position: "insideBottom",
              offset: -5,
              fontSize: 11,
              fill: "var(--muted-foreground)",
            }}
          />
          <YAxis
            stroke="var(--muted-foreground)"
            fontSize={11}
            label={{
              value: "Treatment Effect",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
              fill: "var(--muted-foreground)",
            }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Area
            type="monotone"
            dataKey="range"
            stroke="none"
            fill="#3b82f6"
            fillOpacity={0.15}
          />
          <Line
            type="monotone"
            dataKey="mean"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            name="GP Mean"
          />
          <Line
            type="monotone"
            dataKey="lower"
            stroke="#3b82f6"
            strokeWidth={1}
            strokeDasharray="4 4"
            dot={false}
            name="95% CI Lower"
          />
          <Line
            type="monotone"
            dataKey="upper"
            stroke="#3b82f6"
            strokeWidth={1}
            strokeDasharray="4 4"
            dot={false}
            name="95% CI Upper"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
