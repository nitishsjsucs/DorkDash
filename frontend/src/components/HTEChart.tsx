"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
} from "recharts";

interface HTESegment {
  tenure_bucket?: string;
  deficit_bucket?: string;
  resp_bucket?: string;
  is_peak?: number;
  mean: number;
  std: number;
  count: number;
}

interface HTEData {
  by_tenure: HTESegment[];
  by_deficit: HTESegment[];
  by_peak: HTESegment[];
  by_responsiveness: HTESegment[];
  overall: {
    mean_cate: number;
    std_cate: number;
    median_cate: number;
    p10_cate: number;
    p90_cate: number;
  };
}

function SegmentBarChart({
  data,
  labelKey,
  title,
}: {
  data: HTESegment[];
  labelKey: string;
  title: string;
}) {
  if (!data || data.length === 0) return null;

  const chartData = data.map((seg) => ({
    name: String(
      seg[labelKey as keyof HTESegment] ??
      (seg.is_peak !== undefined ? (seg.is_peak ? "Peak" : "Off-Peak") : "Unknown")
    ),
    CATE: parseFloat(seg.mean.toFixed(4)),
    error: parseFloat(seg.std.toFixed(4)),
    count: seg.count,
  }));

  return (
    <div>
      <h4 className="text-sm font-medium text-[var(--muted-foreground)] mb-2">
        {title}
      </h4>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={chartData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis type="number" stroke="var(--muted-foreground)" fontSize={10} />
          <YAxis
            type="category"
            dataKey="name"
            stroke="var(--muted-foreground)"
            fontSize={10}
            width={80}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value) => typeof value === "number" ? value.toFixed(4) : String(value)}
          />
          <Bar dataKey="CATE" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function HTEChart({ data }: { data: HTEData | null }) {
  if (!data) {
    return (
      <div className="flex items-center justify-center h-48 text-[var(--muted-foreground)]">
        Run experiment to see heterogeneous treatment effects
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {data.overall && (
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-lg bg-[var(--muted)] p-3 text-center">
            <div className="text-[10px] text-[var(--muted-foreground)]">Avg Treatment Effect</div>
            <div className="text-lg font-bold text-purple-400">
              {data.overall.mean_cate.toFixed(4)}
            </div>
          </div>
          <div className="rounded-lg bg-[var(--muted)] p-3 text-center">
            <div className="text-[10px] text-[var(--muted-foreground)]">P10-P90 Range</div>
            <div className="text-sm font-medium text-purple-300">
              {data.overall.p10_cate.toFixed(4)} – {data.overall.p90_cate.toFixed(4)}
            </div>
          </div>
          <div className="rounded-lg bg-[var(--muted)] p-3 text-center">
            <div className="text-[10px] text-[var(--muted-foreground)]">Std Dev</div>
            <div className="text-lg font-bold text-purple-400">
              {data.overall.std_cate.toFixed(4)}
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SegmentBarChart data={data.by_tenure} labelKey="tenure_bucket" title="By Dasher Tenure" />
        <SegmentBarChart data={data.by_deficit} labelKey="deficit_bucket" title="By Zone Deficit" />
        <SegmentBarChart data={data.by_responsiveness} labelKey="resp_bucket" title="By Responsiveness" />
        <SegmentBarChart data={data.by_peak} labelKey="is_peak" title="Peak vs Off-Peak" />
      </div>
    </div>
  );
}
