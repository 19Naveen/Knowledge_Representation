import { StatusPill } from "./StatusPill";

type StatCardProps = {
  title: string;
  value: string;
  subtitle?: string;
  trend?: { value: string; direction: "up" | "down" };
  status?: { label: string; tone?: "success" | "warning" | "danger" | "info" | "neutral" };
};

export function StatCard({ title, value, subtitle, trend, status }: StatCardProps) {
  return (
    <article className="card">
      <div className="flex items-start justify-between gap-3">
        <span className="text-sm font-medium text-text-secondary">{title}</span>
        {status && <StatusPill label={status.label} tone={status.tone} size="xs" />}
      </div>
      <div className="mt-3 flex items-baseline gap-2">
        <p className="text-3xl font-heading font-semibold tracking-tight text-text tabular-nums">{value}</p>
        {trend && (
          <span className={cn("text-xs font-medium", trend.direction === "up" ? "text-success" : "text-danger")}>
            {trend.direction === "up" ? "↑" : "↓"} {trend.value}
          </span>
        )}
      </div>
      {subtitle && <p className="mt-1.5 text-xs text-text-tertiary">{subtitle}</p>}
    </article>
  );
}

import { cn } from "../../lib/cn";