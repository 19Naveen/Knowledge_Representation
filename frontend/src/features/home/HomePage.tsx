import { useState } from "react";
import { Link } from "react-router-dom";
import { StatusPill } from "../../components/shared/StatusPill";
import { PageHeader } from "../../components/shared/PageHeader";
import { cn } from "../../lib/cn";
import { actionAdvices, datasets, deployments, leaderboard, workspace } from "../../lib/mocks/data";

const progressSteps = [
  { id: 1, label: "Data Connected", status: "completed", date: "Jan 15, 2026" },
  { id: 2, label: "Exploration", status: "completed", date: "Jan 22, 2026" },
  { id: 3, label: "Model Training", status: "active", date: "In Progress" },
  { id: 4, label: "Deployment", status: "pending", date: "Est. Feb 5, 2026" }
];

const recentActivity = [
  { id: 1, action: "Model trained", target: "XGBoost v2.1", time: "2h ago", type: "success" },
  { id: 2, action: "Data synced", target: "Customer_Churn_2024", time: "4h ago", type: "info" },
  { id: 3, action: "Alert triggered", target: "Drift detected", time: "6h ago", type: "warning" },
  { id: 4, action: "Deployment", target: "Canary promoted", time: "1d ago", type: "success" }
];

function Sparkline({ data }: { data: number[] }) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const height = 32;
  const width = 80;
  
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg className="w-20 h-8" viewBox={`0 0 ${width} ${height}`}>
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-primary"
      />
    </svg>
  );
}

function MetricCard({ label, value, change, trend, sparkData }: { label: string; value: string; change: string; trend: "up" | "down"; sparkData?: number[] }) {
  const isPositive = trend === "up";
  const changeColor = label === "Latency" ? (isPositive ? "text-success" : "text-danger") : isPositive ? "text-success" : "text-danger";
  
  return (
    <div className="card card-hover group relative overflow-hidden">
      <div className="flex items-start justify-between">
        <span className="text-xs font-medium text-text-tertiary">{label}</span>
        <span className={cn("text-[11px] font-semibold tabular-nums", changeColor)}>
          {change}
        </span>
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-2xl font-semibold tracking-tight text-text">{value}</span>
      </div>
      {sparkData && (
        <div className="absolute bottom-2 right-2 opacity-40 transition-opacity group-hover:opacity-70">
          <Sparkline data={sparkData} />
        </div>
      )}
    </div>
  );
}

function ProgressStep({ step, isLast }: { step: typeof progressSteps[0]; isLast: boolean }) {
  const statusClasses = {
    completed: "bg-success text-white",
    active: "bg-primary text-white ring-4 ring-primary/10",
    pending: "bg-surface-2 text-text-tertiary border border-dashed"
  };
  
  const icon = step.status === "completed" ? "✓" : step.status === "active" ? step.id : String(step.id);
  
  return (
    <div className="flex items-center gap-3">
      <div className={cn("flex size-8 items-center justify-center rounded-full text-sm font-semibold transition-all", statusClasses[step.status as keyof typeof statusClasses])}>
        {icon}
      </div>
      <div className="flex flex-col">
        <span className={cn("text-sm font-medium", step.status === "pending" ? "text-text-tertiary" : "text-text")}>
          {step.label}
        </span>
        <span className="text-[11px] text-text-tertiary">{step.date}</span>
      </div>
      {isLast && <div className="h-px flex-1 bg-border" />}
    </div>
  );
}

export function HomePage() {
  const [activeTab, setActiveTab] = useState<"overview" | "models" | "data">("overview");
  const [searchQuery, setSearchQuery] = useState("");
  
  const activeDataset = datasets.find((item) => item.id === workspace.activeDatasetId) ?? datasets[0];
  const champion = leaderboard[0];
  
  const filteredAdvices = actionAdvices.filter(adv => 
    adv.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    adv.why.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  return (
    <section className="space-y-6 animate-in fade-in duration-500">
      <PageHeader
        title="Project Home"
        subtitle="Real-time overview of your ML project. Track progress, monitor metrics, and manage deployments."
        actions={
          <div className="flex items-center gap-2">
            <div className="relative">
              <input
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input h-8 w-48 pl-8"
              />
              <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <Link to="/workspace-settings" className="btn btn-secondary text-xs">
              Settings
            </Link>
            <Link to="/data-import" className="btn btn-primary text-xs">
              Connect Data
            </Link>
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard label="Model Accuracy" value="94.2%" change="+2.1%" trend="up" sparkData={[88, 89, 91, 90, 92, 94]} />
        <MetricCard label="Prediction Rate" value="1.2K/hr" change="+8%" trend="up" sparkData={[800, 950, 880, 1100, 1050, 1200]} />
        <MetricCard label="Latency" value="118ms" change="-12ms" trend="down" sparkData={[150, 140, 135, 130, 125, 118]} />
        <MetricCard label="Active Models" value="3" change="+1" trend="up" sparkData={[1, 1, 2, 2, 2, 3]} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card lg:col-span-1">
          <div className="border-b border-border pb-4 mb-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-text">Project Progress</h2>
              <StatusPill label="In Progress" tone="info" />
            </div>
          </div>
          <div className="space-y-1">
            {progressSteps.map((step, i) => (
              <ProgressStep key={step.id} step={step} isLast={i < progressSteps.length - 1} />
            ))}
          </div>
          
          <div className="mt-5 rounded-lg bg-surface-2/50 p-4">
            <div className="flex items-center justify-between text-xs">
              <span className="text-text-tertiary">Overall Completion</span>
              <span className="font-semibold text-text">75%</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-2">
              <div 
                className="h-full rounded-full bg-gradient-to-r from-primary to-primary/80 transition-all duration-1000"
                style={{ width: "75%" }}
              />
            </div>
          </div>
        </div>

        <div className="card lg:col-span-1">
          <div className="flex items-center justify-between border-b border-border pb-4 mb-4">
            <div>
              <h2 className="text-sm font-semibold text-text">Active Dataset</h2>
              <p className="mt-0.5 text-xs text-text-tertiary">Primary data source for model training.</p>
            </div>
            <Link to="/data-import" className="text-xs font-medium text-primary hover:underline">
              Manage
            </Link>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between rounded-lg border bg-surface-2/30 p-3">
              <div>
                <p className="text-xs text-text-tertiary">Dataset Name</p>
                <p className="mt-0.5 text-sm font-semibold text-text">{activeDataset.name}</p>
              </div>
              <div className="text-right">
                <p className="text-xs text-text-tertiary">Quality Score</p>
                <p className={cn("text-sm font-bold", activeDataset.qualityScore >= 90 ? "text-success" : activeDataset.qualityScore >= 80 ? "text-warning" : "text-danger")}>
                  {activeDataset.qualityScore}%
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border bg-surface-2/20 p-3">
                <p className="text-[11px] text-text-tertiary uppercase tracking-wider">Total Rows</p>
                <p className="mt-1 text-lg font-semibold text-text tabular-nums">{activeDataset.rows.toLocaleString()}</p>
              </div>
              <div className="rounded-lg border bg-surface-2/20 p-3">
                <p className="text-[11px] text-text-tertiary uppercase tracking-wider">Columns</p>
                <p className="mt-1 text-lg font-semibold text-text tabular-nums">{activeDataset.columns}</p>
              </div>
            </div>

            {activeDataset.issues.length > 0 && (
              <div className="rounded-lg border border-warning/20 bg-warning/5 p-3">
                <div className="flex items-center gap-2">
                  <svg className="size-4 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <p className="text-xs font-semibold text-warning">{activeDataset.issues.length} Issues Detected</p>
                </div>
                <ul className="mt-2 space-y-1">
                  {activeDataset.issues.slice(0, 2).map((issue, i) => (
                    <li key={i} className="text-[11px] text-text-tertiary">• {issue}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        <div className="card lg:col-span-1">
          <div className="flex items-center justify-between border-b border-border pb-4 mb-4">
            <div>
              <h2 className="text-sm font-semibold text-text">Champion Model</h2>
              <p className="mt-0.5 text-xs text-text-tertiary">Best performing model in production.</p>
            </div>
            <Link to="/automl-lab" className="text-xs font-medium text-primary hover:underline">
              Leaderboard
            </Link>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between rounded-lg border bg-surface-2/30 p-3">
              <div>
                <p className="text-xs text-text-tertiary">Model</p>
                <p className="mt-0.5 text-lg font-bold text-text">{champion.model}</p>
              </div>
              <div className="text-right">
                <p className="text-xs text-text-tertiary">Validation Score</p>
                <p className="text-xl font-bold text-success tabular-nums">{(champion.score * 100).toFixed(1)}%</p>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-text-tertiary">Model Rank</span>
                <span className="font-medium text-text">#1 of {leaderboard.length}</span>
              </div>
              <div className="flex gap-1">
                {leaderboard.map((m, i) => (
                  <div 
                    key={m.id}
                    className={cn(
                      "h-1.5 flex-1 rounded-full transition-all",
                      i === 0 ? "bg-success" : i === 1 ? "bg-primary/50" : "bg-surface-2"
                    )}
                    title={`${m.model}: ${(m.score * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
            </div>

            <div className="flex gap-2">
              <div className="flex-1 rounded-md border border-success/20 bg-success/5 px-3 py-2 text-center">
                <p className="text-[10px] uppercase tracking-wider text-success">Leakage</p>
                <p className="mt-0.5 text-xs font-semibold text-success">Pass</p>
              </div>
              <div className="flex-1 rounded-md border border-info/20 bg-info/5 px-3 py-2 text-center">
                <p className="text-[10px] uppercase tracking-wider text-info">Status</p>
                <p className="mt-0.5 text-xs font-semibold text-info">Prod</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center gap-6 border-b border-border pb-4 mb-4">
          {(["overview", "models", "data"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={cn(
                "border-b-2 py-4 text-sm font-medium capitalize transition-colors",
                activeTab === tab 
                  ? "border-primary text-text" 
                  : "border-transparent text-text-tertiary hover:text-text"
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        <div>
          {activeTab === "overview" && (
            <div className="grid gap-6 md:grid-cols-2">
              <div>
                <h3 className="mb-3 text-sm font-semibold text-text">Recommended Actions</h3>
                <div className="space-y-3">
                  {filteredAdvices.slice(0, 3).map((advice) => (
                    <div 
                      key={advice.id}
                      className="group relative overflow-hidden rounded-lg border bg-surface-2/30 p-4 transition-all hover:border-primary/20 hover:bg-surface-2/50"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="text-sm font-medium text-text">{advice.title}</h4>
                        <StatusPill 
                          label={advice.impact} 
                          tone={advice.impact === "high" ? "danger" : advice.impact === "medium" ? "warning" : "info"} 
                        />
                      </div>
                      <p className="mt-2 text-xs text-text-tertiary">{advice.why}</p>
                      <p className="mt-2 text-[11px] text-primary font-medium">{advice.nextStep}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="mb-3 text-sm font-semibold text-text">Recent Activity</h3>
                <div className="space-y-3">
                  {recentActivity.map((activity) => (
                    <div 
                      key={activity.id}
                      className="flex items-center gap-3 rounded-lg border bg-surface-2/30 p-3 transition-colors hover:bg-surface-2/50"
                    >
                      <div className={cn(
                        "flex size-8 shrink-0 items-center justify-center rounded-full",
                        activity.type === "success" && "bg-success/10 text-success",
                        activity.type === "info" && "bg-info/10 text-info",
                        activity.type === "warning" && "bg-warning/10 text-warning"
                      )}>
                        {activity.type === "success" && (
                          <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                        )}
                        {activity.type === "info" && (
                          <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        )}
                        {activity.type === "warning" && (
                          <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-text truncate">{activity.action}</p>
                        <p className="text-xs text-text-tertiary">{activity.target}</p>
                      </div>
                      <span className="text-[11px] text-text-tertiary tabular-nums">{activity.time}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "models" && (
            <div className="space-y-3">
              {leaderboard.map((model, i) => (
                <div 
                  key={model.id}
                  className={cn(
                    "flex items-center gap-4 rounded-lg border p-4 transition-colors hover:bg-surface-2/30",
                    i === 0 ? "border-success/20 bg-success/5" : "bg-surface"
                  )}
                >
                  <div className={cn(
                    "flex size-8 items-center justify-center rounded-full text-sm font-bold",
                    i === 0 ? "bg-success text-white" : "bg-surface-2 text-text-tertiary"
                  )}>
                    #{i + 1}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-semibold text-text">{model.model}</h4>
                      {i === 0 && <StatusPill label="Champion" tone="success" />}
                    </div>
                    <p className="mt-0.5 text-xs text-text-tertiary">{model.status}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xl font-bold text-text tabular-nums">{(model.score * 100).toFixed(1)}%</p>
                    <p className="text-[11px] text-text-tertiary">Validation Score</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === "data" && (
            <div className="space-y-3">
              {datasets.map((dataset) => (
                <div 
                  key={dataset.id}
                  className="flex items-center gap-4 rounded-lg border bg-surface p-4 transition-colors hover:bg-surface-2/30"
                >
                  <div className="flex size-10 items-center justify-center rounded-lg bg-surface-2">
                    <svg className="size-5 text-text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-text">{dataset.name}</h4>
                    <div className="mt-0.5 flex items-center gap-3 text-xs text-text-tertiary">
                      <span>{dataset.owner}</span>
                      <span>•</span>
                      <span>{dataset.rows.toLocaleString()} rows</span>
                      <span>•</span>
                      <span>{dataset.columns} cols</span>
                      <span>•</span>
                      <span>{dataset.freshness}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <p className={cn(
                        "text-sm font-bold tabular-nums",
                        dataset.qualityScore >= 90 ? "text-success" : dataset.qualityScore >= 80 ? "text-warning" : "text-danger"
                      )}>
                        {dataset.qualityScore}%
                      </p>
                      <p className="text-[11px] text-text-tertiary">Quality</p>
                    </div>
                    <button className="btn btn-secondary text-xs">
                      Select
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between border-b border-border pb-4 mb-4">
          <h2 className="text-sm font-semibold text-text">Active Deployments</h2>
          <Link to="/deployments" className="text-xs font-medium text-primary hover:underline">
            View All
          </Link>
        </div>
        <div className="grid gap-px divide-x divide-border md:grid-cols-2">
          {deployments.map((deployment) => (
            <div key={deployment.id} className="flex items-center gap-4 p-4">
              <div className={cn(
                "flex size-10 items-center justify-center rounded-lg",
                deployment.env === "production" ? "bg-success/10 text-success" : "bg-warning/10 text-warning"
              )}>
                <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-semibold text-text truncate">{deployment.modelName}</h4>
                <div className="mt-0.5 flex items-center gap-2 text-xs text-text-tertiary">
                  <StatusPill 
                    label={deployment.env} 
                    tone={deployment.env === "production" ? "success" : "warning"} 
                  />
                  <span>{deployment.strategy} strategy</span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold text-text tabular-nums">{deployment.latencyMs}ms</p>
                <p className="text-[11px] text-text-tertiary">latency</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold text-success tabular-nums">+{deployment.expectedLift}%</p>
                <p className="text-[11px] text-text-tertiary">lift</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}