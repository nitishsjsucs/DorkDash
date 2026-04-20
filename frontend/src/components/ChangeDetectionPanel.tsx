"use client";

import { AlertTriangle, Activity, Shield } from "lucide-react";

interface ChangeEvent {
  step: number;
  zone_id: number;
  detector: string;
  severity: string;
  metric: number;
  description: string;
  recommendation: string;
}

interface ChangeDetectionState {
  total_events: number;
  recent_events: ChangeEvent[];
  active_alerts_by_zone: Record<string, { detector: string; severity: string; description: string }[]>;
}

interface ChangePoint {
  day: number;
  zone_id: number;
  zone_name: string;
  shift: number;
  type: string;
}

interface Props {
  changeDetection: ChangeDetectionState | null;
  changePoints: ChangePoint[];
}

const SEVERITY_STYLES: Record<string, string> = {
  high: "bg-red-500/10 border-red-500/30 text-red-400",
  medium: "bg-yellow-500/10 border-yellow-500/30 text-yellow-400",
  low: "bg-blue-500/10 border-blue-500/30 text-blue-300",
};

export default function ChangeDetectionPanel({ changeDetection, changePoints }: Props) {
  if (!changeDetection) {
    return (
      <div className="text-sm text-[var(--muted-foreground)]">
        Run an experiment to see non-stationarity detection results
      </div>
    );
  }

  const activeZones = Object.entries(changeDetection.active_alerts_by_zone || {});
  const recentEvents = changeDetection.recent_events?.slice(-10) || [];

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 rounded-lg bg-[var(--muted)] px-3 py-2">
          <Activity className="h-4 w-4 text-cyan-400" />
          <span className="text-sm font-medium">{changeDetection.total_events}</span>
          <span className="text-xs text-[var(--muted-foreground)]">total events detected</span>
        </div>
        <div className="flex items-center gap-2 rounded-lg bg-[var(--muted)] px-3 py-2">
          <AlertTriangle className="h-4 w-4 text-yellow-400" />
          <span className="text-sm font-medium">{activeZones.length}</span>
          <span className="text-xs text-[var(--muted-foreground)]">zones with active alerts</span>
        </div>
        <div className="flex items-center gap-2 rounded-lg bg-[var(--muted)] px-3 py-2">
          <Shield className="h-4 w-4 text-purple-400" />
          <span className="text-sm font-medium">{changePoints?.length || 0}</span>
          <span className="text-xs text-[var(--muted-foreground)]">regime changes (ground truth)</span>
        </div>
      </div>

      {/* Active alerts by zone */}
      {activeZones.length > 0 && (
        <div>
          <h4 className="text-sm font-medium mb-2">Active Alerts</h4>
          <div className="space-y-2">
            {activeZones.map(([zoneId, alerts]) => (
              <div key={zoneId}>
                {alerts.map((alert, i) => (
                  <div
                    key={i}
                    className={`rounded-lg border px-3 py-2 text-xs ${SEVERITY_STYLES[alert.severity] || SEVERITY_STYLES.low}`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Zone {zoneId} · {alert.detector}</span>
                      <span className="uppercase text-[10px] tracking-wide">{alert.severity}</span>
                    </div>
                    <p className="mt-1 opacity-80">{alert.description}</p>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Regime change log (ground truth) */}
      {changePoints && changePoints.length > 0 && (
        <div>
          <h4 className="text-sm font-medium mb-2">Regime Changes (Ground Truth)</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-1.5 px-2 text-[var(--muted-foreground)]">Day</th>
                  <th className="text-left py-1.5 px-2 text-[var(--muted-foreground)]">Zone</th>
                  <th className="text-right py-1.5 px-2 text-[var(--muted-foreground)]">Shift</th>
                  <th className="text-left py-1.5 px-2 text-[var(--muted-foreground)]">Type</th>
                </tr>
              </thead>
              <tbody>
                {changePoints.slice(-10).map((cp, i) => (
                  <tr key={i} className="border-b border-[var(--border)]/50">
                    <td className="py-1.5 px-2">{cp.day}</td>
                    <td className="py-1.5 px-2">{cp.zone_name}</td>
                    <td className={`py-1.5 px-2 text-right tabular-nums ${cp.shift > 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {cp.shift > 0 ? "+" : ""}{cp.shift.toFixed(2)}
                    </td>
                    <td className="py-1.5 px-2 text-[var(--muted-foreground)]">{cp.type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent detection events */}
      {recentEvents.length > 0 && (
        <div>
          <h4 className="text-sm font-medium mb-2">Recent Detection Events</h4>
          <div className="space-y-1">
            {recentEvents.map((evt, i) => (
              <div key={i} className="flex items-center gap-2 text-xs py-1 border-b border-[var(--border)]/30">
                <span className={`inline-block h-1.5 w-1.5 rounded-full ${
                  evt.severity === "high" ? "bg-red-400" : evt.severity === "medium" ? "bg-yellow-400" : "bg-blue-400"
                }`} />
                <span className="text-[var(--muted-foreground)]">Step {evt.step}</span>
                <span className="font-medium">Zone {evt.zone_id}</span>
                <span className="text-[var(--muted-foreground)]">{evt.detector}:</span>
                <span className="flex-1 truncate">{evt.description}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
