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
} from "recharts";

interface DoseResponseData {
  fitted: boolean;
  grid?: number[];
  response?: number[];
  std?: number[];
  lower?: number[];
  upper?: number[];
}

export default function DoseResponseChart({ data }: { data: DoseResponseData | null }) {
  if (!data || !data.fitted || !data.grid) {
    return (
      <div className="flex items-center justify-center h-48 text-[var(--muted-foreground)] text-sm">
        Run experiment to see dose-response curve
      </div>
    );
  }

  const chartData = data.grid.map((t, i) => ({
    incentive: parseFloat(t.toFixed(2)),
    response: data.response![i],
    lower: data.lower![i],
    upper: data.upper![i],
  }));

  return (
    <div>
      <div className="mb-2 text-xs text-[var(--muted-foreground)]">
        Kernel-weighted dose-response: E[response | incentive = t] with 95% CI
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="incentive"
            stroke="var(--muted-foreground)"
            fontSize={10}
            tickLine={false}
            label={{ value: "Incentive ($)", position: "insideBottom", offset: -2, fontSize: 10, fill: "var(--muted-foreground)" }}
          />
          <YAxis
            stroke="var(--muted-foreground)"
            fontSize={10}
            tickLine={false}
            label={{ value: "P(respond)", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--muted-foreground)" }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 11,
            }}
            formatter={(value: unknown) => typeof value === "number" ? value.toFixed(4) : String(value)}
          />
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="#a855f7"
            fillOpacity={0.08}
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="var(--background)"
            fillOpacity={1}
          />
          <Line
            type="monotone"
            dataKey="response"
            stroke="#a855f7"
            strokeWidth={2}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
