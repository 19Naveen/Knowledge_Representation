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
    <div className="card flex items-center gap-3 p-4">
      {icon && <div className="flex h-9 w-9 items-center justify-center rounded-md bg-surface-2 text-text-secondary">{icon}</div>}
      <div className="flex-1">
        <p className="text-[11px] font-medium text-text-tertiary uppercase tracking-wider">{label}</p>
        <div className="mt-0.5 flex items-baseline gap-2">
          <p className="text-xl font-semibold tracking-tight text tabular-nums">{value}</p>
          {trend && (
            <span className={cn("text-xs font-medium", trend.direction === "up" ? "text-success" : "text-danger")}>
              {trend.direction === "up" ? "↑" : "↓"} {trend.value}
            </span>
          )}
        </div>
        {hint && <p className="mt-0.5 text-[11px] text-text-tertiary">{hint}</p>}
      </div>
    </div>
  );
}