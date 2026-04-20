"use client";

import { useMemo } from "react";

interface ZoneNode {
  zone_id: number;
  name: string;
  cluster_id: number;
  supply_deficit: number;
  spillover_pressure: number;
  daily_spend: number;
}

interface NetworkGraphProps {
  zones: ZoneNode[];
  adjacency: number[][];
  clusters: Record<string, number[]>;
}

const CLUSTER_COLORS: Record<number, string> = {
  0: "#f97316", // SF Core — orange
  1: "#22c55e", // SF Outer — green
  2: "#3b82f6", // East Bay — blue
  3: "#a855f7", // South Bay — purple
};

const CLUSTER_NAMES: Record<number, string> = {
  0: "SF Core",
  1: "SF Outer",
  2: "East Bay",
  3: "South Bay",
};

export default function NetworkGraph({ zones, adjacency, clusters }: NetworkGraphProps) {
  // Position nodes in a force-directed-ish layout (static positions for 10 zones)
  const positions = useMemo(() => {
    const pos: Record<number, { x: number; y: number }> = {
      0: { x: 200, y: 80 },   // Downtown SF
      1: { x: 170, y: 160 },  // Mission
      2: { x: 260, y: 130 },  // SoMa
      3: { x: 120, y: 50 },   // Marina
      4: { x: 60, y: 140 },   // Sunset
      5: { x: 50, y: 60 },    // Richmond
      6: { x: 380, y: 100 },  // Oakland DT
      7: { x: 400, y: 40 },   // Berkeley
      8: { x: 350, y: 260 },  // San Jose
      9: { x: 260, y: 260 },  // Palo Alto
    };
    return pos;
  }, []);

  // Compute edges from adjacency
  const edges = useMemo(() => {
    const result: { from: number; to: number; weight: number }[] = [];
    if (!adjacency || adjacency.length === 0) return result;
    for (let i = 0; i < adjacency.length; i++) {
      for (let j = i + 1; j < adjacency[i].length; j++) {
        if (adjacency[i][j] > 0.05) {
          result.push({ from: i, to: j, weight: adjacency[i][j] });
        }
      }
    }
    return result;
  }, [adjacency]);

  if (!zones || zones.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-[var(--muted-foreground)]">
        Run an experiment to see zone network
      </div>
    );
  }

  const maxSpillover = Math.max(...zones.map((z) => z.spillover_pressure || 0), 0.01);

  return (
    <div>
      {/* Legend */}
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        {Object.entries(CLUSTER_NAMES).map(([cid, name]) => (
          <span key={cid} className="flex items-center gap-1.5 text-xs">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: CLUSTER_COLORS[parseInt(cid)] }}
            />
            {name}
          </span>
        ))}
        <span className="text-xs text-[var(--muted-foreground)] ml-2">
          Edge thickness = adjacency weight · Node size = supply deficit
        </span>
      </div>

      <svg viewBox="0 0 460 310" className="w-full h-auto" style={{ maxHeight: 350 }}>
        {/* Cluster backgrounds */}
        {Object.entries(clusters).map(([cid, zoneIds]) => {
          const pts = zoneIds.map((zid) => positions[zid]).filter(Boolean);
          if (pts.length === 0) return null;
          const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
          const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
          const r = Math.max(60, ...pts.map((p) => Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2) + 40));
          return (
            <g key={cid}>
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={CLUSTER_COLORS[parseInt(cid)] || "#666"}
                opacity={0.06}
                stroke={CLUSTER_COLORS[parseInt(cid)] || "#666"}
                strokeWidth={1}
                strokeDasharray="4 4"
                strokeOpacity={0.3}
              />
              <text x={cx} y={cy + r - 8} textAnchor="middle" fontSize={9} fill={CLUSTER_COLORS[parseInt(cid)]} opacity={0.5}>
                {CLUSTER_NAMES[parseInt(cid)]}
              </text>
            </g>
          );
        })}

        {/* Edges */}
        {edges.map((e, i) => {
          const p1 = positions[e.from];
          const p2 = positions[e.to];
          if (!p1 || !p2) return null;
          return (
            <line
              key={i}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke="var(--muted-foreground)"
              strokeWidth={Math.max(0.5, e.weight * 3)}
              strokeOpacity={0.2 + e.weight * 0.3}
            />
          );
        })}

        {/* Nodes */}
        {zones.map((zone) => {
          const pos = positions[zone.zone_id];
          if (!pos) return null;
          const deficit = Math.max(0.1, Math.abs(zone.supply_deficit));
          const r = 10 + deficit * 20;
          const color = CLUSTER_COLORS[zone.cluster_id] || "#888";
          const spillNorm = (zone.spillover_pressure || 0) / maxSpillover;
          return (
            <g key={zone.zone_id}>
              {/* Spillover glow */}
              {spillNorm > 0.1 && (
                <circle cx={pos.x} cy={pos.y} r={r + 6} fill="red" opacity={spillNorm * 0.2} />
              )}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={r}
                fill={color}
                opacity={0.8}
                stroke={color}
                strokeWidth={1.5}
              />
              <text
                x={pos.x}
                y={pos.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={7}
                fill="white"
                fontWeight={600}
              >
                Z{zone.zone_id}
              </text>
              <text
                x={pos.x}
                y={pos.y + r + 10}
                textAnchor="middle"
                fontSize={8}
                fill="var(--muted-foreground)"
              >
                {zone.name.split(" ")[0]}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Spillover stats table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="text-left py-1.5 px-2 text-[var(--muted-foreground)]">Zone</th>
              <th className="text-left py-1.5 px-2 text-[var(--muted-foreground)]">Cluster</th>
              <th className="text-right py-1.5 px-2 text-[var(--muted-foreground)]">Deficit</th>
              <th className="text-right py-1.5 px-2 text-[var(--muted-foreground)]">Spillover</th>
              <th className="text-right py-1.5 px-2 text-[var(--muted-foreground)]">Daily $</th>
            </tr>
          </thead>
          <tbody>
            {zones.map((z) => (
              <tr key={z.zone_id} className="border-b border-[var(--border)]/50">
                <td className="py-1.5 px-2">{z.name}</td>
                <td className="py-1.5 px-2">
                  <span className="inline-block h-2 w-2 rounded-full mr-1" style={{ background: CLUSTER_COLORS[z.cluster_id] }} />
                  {CLUSTER_NAMES[z.cluster_id]}
                </td>
                <td className="py-1.5 px-2 text-right tabular-nums">
                  {(z.supply_deficit * 100).toFixed(1)}%
                </td>
                <td className="py-1.5 px-2 text-right tabular-nums">
                  {(z.spillover_pressure || 0).toFixed(3)}
                </td>
                <td className="py-1.5 px-2 text-right tabular-nums">
                  ${(z.daily_spend || 0).toFixed(0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
