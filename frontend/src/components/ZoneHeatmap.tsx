"use client";

import { cn } from "@/lib/utils";

interface ZoneSnapshot {
  zone_id: number;
  name: string;
  supply_deficit: number;
  effective_elasticity: number;
  is_peak: boolean;
  demand: number;
  supply: number;
}

export default function ZoneHeatmap({ zones }: { zones: ZoneSnapshot[] }) {
  if (!zones || zones.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-[var(--muted-foreground)]">
        No zone data
      </div>
    );
  }

  const getDeficitColor = (deficit: number) => {
    if (deficit <= 0) return "bg-emerald-500/20 border-emerald-500/40 text-emerald-300";
    if (deficit < 0.1) return "bg-yellow-500/20 border-yellow-500/40 text-yellow-300";
    if (deficit < 0.2) return "bg-orange-500/20 border-orange-500/40 text-orange-300";
    return "bg-red-500/20 border-red-500/40 text-red-300";
  };

  const getDeficitLabel = (deficit: number) => {
    if (deficit <= 0) return "Balanced";
    if (deficit < 0.1) return "Mild";
    if (deficit < 0.2) return "Moderate";
    return "Severe";
  };

  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
      {zones.map((zone) => (
        <div
          key={zone.zone_id}
          className={cn(
            "rounded-lg border p-3 transition-all hover:scale-[1.02]",
            getDeficitColor(zone.supply_deficit)
          )}
        >
          <div className="text-[10px] font-medium opacity-70 truncate">
            {zone.name}
          </div>
          <div className="mt-1 text-lg font-bold tabular-nums">
            {(zone.supply_deficit * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] opacity-60">
            {getDeficitLabel(zone.supply_deficit)} deficit
          </div>
          <div className="mt-1.5 flex items-center gap-1.5 text-[10px] opacity-50">
            <span>D:{zone.demand.toFixed(0)}</span>
            <span>S:{zone.supply.toFixed(0)}</span>
          </div>
          {zone.is_peak && (
            <span className="mt-1 inline-block rounded bg-white/10 px-1.5 py-0.5 text-[9px] font-medium">
              PEAK
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
