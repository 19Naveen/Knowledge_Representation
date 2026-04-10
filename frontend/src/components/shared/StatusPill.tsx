import { cn } from "../../lib/cn";

const toneMap = {
  success: { bg: "bg-success-muted", text: "text-success", dot: "bg-success" },
  warning: { bg: "bg-warning-muted", text: "text-warning", dot: "bg-warning" },
  danger: { bg: "bg-danger-muted", text: "text-danger", dot: "bg-danger" },
  info: { bg: "bg-info-muted", text: "text-info", dot: "bg-info" },
  neutral: { bg: "bg-surface-2", text: "text-text-secondary", dot: "bg-text-tertiary" }
};

export function StatusPill({ label, tone = "neutral", size = "sm" }: { label: string; tone?: keyof typeof toneMap; size?: "sm" | "xs" }) {
  const styles = toneMap[tone];
  const sizeClasses = size === "sm" ? "px-2.5 py-1 text-xs" : "px-2 py-0.5 text-[10px]";
  
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full font-medium", styles.bg, styles.text, sizeClasses)}>
      <span className={cn("size-1.5 rounded-full", styles.dot)} />
      {label}
    </span>
  );
}