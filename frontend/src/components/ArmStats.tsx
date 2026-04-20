"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface ArmStat {
  arm_id: string;
  mean?: number;
  avg_reward: number;
  n_pulls: number;
  variance?: number;
  uncertainty?: number;
}

interface BanditState {
  algorithm: string;
  arms: ArmStat[];
  cumulative_regret: number;
  total_pulls: number;
}

export default function ArmStats({
  vanilla,
  contextual,
}: {
  vanilla: BanditState | null;
  contextual: BanditState | null;
}) {
  if (!vanilla && !contextual) {
    return (
      <div className="flex items-center justify-center h-48 text-[var(--muted-foreground)]">
        No bandit data yet
      </div>
    );
  }

  const chartData = (vanilla?.arms || []).map((arm) => {
    const ctxArm = contextual?.arms.find((a) => a.arm_id === arm.arm_id);
    return {
      arm: arm.arm_id,
      "Vanilla TS": parseFloat(arm.avg_reward.toFixed(4)),
      "Contextual LinTS": parseFloat((ctxArm?.avg_reward || 0).toFixed(4)),
      "VTS Pulls": arm.n_pulls,
      "CTS Pulls": ctxArm?.n_pulls || 0,
    };
  });

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3">
        {vanilla && (
          <div className="rounded-lg bg-orange-500/10 border border-orange-500/20 p-3">
            <div className="text-[10px] text-orange-300/70 font-medium">Vanilla TS</div>
            <div className="mt-1 text-xl font-bold text-orange-400 tabular-nums">
              {vanilla.cumulative_regret.toFixed(1)}
            </div>
            <div className="text-[10px] text-orange-300/50">
              cumulative regret · {vanilla.total_pulls} pulls
            </div>
          </div>
        )}
        {contextual && (
          <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-3">
            <div className="text-[10px] text-emerald-300/70 font-medium">Contextual LinTS</div>
            <div className="mt-1 text-xl font-bold text-emerald-400 tabular-nums">
              {contextual.cumulative_regret.toFixed(1)}
            </div>
            <div className="text-[10px] text-emerald-300/50">
              cumulative regret · {contextual.total_pulls} pulls
            </div>
          </div>
        )}
      </div>

      {/* Arm comparison chart */}
      <div>
        <h4 className="text-sm font-medium text-[var(--muted-foreground)] mb-2">
          Average Reward by Arm
        </h4>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="arm" stroke="var(--muted-foreground)" fontSize={11} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Bar dataKey="Vanilla TS" fill="#f97316" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Contextual LinTS" fill="#22c55e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Pull distribution */}
      <div>
        <h4 className="text-sm font-medium text-[var(--muted-foreground)] mb-2">
          Pull Distribution
        </h4>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="arm" stroke="var(--muted-foreground)" fontSize={11} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Bar dataKey="VTS Pulls" fill="#f97316" opacity={0.6} radius={[4, 4, 0, 0]} />
            <Bar dataKey="CTS Pulls" fill="#22c55e" opacity={0.6} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
