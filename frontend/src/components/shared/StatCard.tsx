import { cn } from "../../lib/cn";
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
    <article className="card p-4">
      <div className="flex items-start justify-between gap-3">
        <span className="text-[11px] font-medium text-text-tertiary uppercase tracking-wider">{title}</span>
        {status && <StatusPill label={status.label} tone={status.tone} size="xs" />}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <p className="text-2xl font-semibold tracking-tight text tabular-nums">{value}</p>
        {trend && (
          <span className={cn("text-xs font-medium", trend.direction === "up" ? "text-success" : "text-danger")}>
            {trend.direction === "up" ? "↑" : "↓"} {trend.value}
          </span>
        )}
      </div>
      {subtitle && <p className="mt-1 text-xs text-text-tertiary">{subtitle}</p>}
    </article>
  );
}