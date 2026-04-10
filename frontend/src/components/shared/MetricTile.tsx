import { cn } from "../../lib/cn";

type MetricTileProps = {
  label: string;
  value: string;
  hint?: string;
  icon?: React.ReactNode;
  trend?: { value: string; direction: "up" | "down" };
};

export function MetricTile({ label, value, hint, icon, trend }: MetricTileProps) {
  return (
    <div className="card flex items-center gap-4 p-4">
      {icon && <div className="flex size-10 items-center justify-center rounded-lg bg-surface-2 text-text-secondary">{icon}</div>}
      <div className="flex-1">
        <p className="text-xs font-medium text-text-tertiary uppercase tracking-wide">{label}</p>
        <div className="mt-1 flex items-baseline gap-2">
          <p className="text-2xl font-heading font-semibold tracking-tight text-text tabular-nums">{value}</p>
          {trend && (
            <span className={cn("text-xs font-medium", trend.direction === "up" ? "text-success" : "text-danger")}>
              {trend.direction === "up" ? "↑" : "↓"} {trend.value}
            </span>
          )}
        </div>
        {hint && <p className="mt-0.5 text-xs text-text-tertiary">{hint}</p>}
      </div>
    </div>
  );
}