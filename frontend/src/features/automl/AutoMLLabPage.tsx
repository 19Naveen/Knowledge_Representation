import { useState } from "react";
import { MetricTile } from "../../components/shared/MetricTile";
import { PageHeader } from "../../components/shared/PageHeader";
import { StatusPill } from "../../components/shared/StatusPill";
import { cn } from "../../lib/cn";

const extendedLeaderboard = [
  {
    id: "m-1",
    model: "XGBoost",
    score: 0.942,
    status: "completed" as const,
    accuracy: 0.948,
    precision: 0.912,
    recall: 0.876,
    auc: 0.981,
    trainingTime: "4m 32s",
    features: 24,
    createdAt: "2024-01-15T10:23:00Z"
  },
  {
    id: "m-2",
    model: "Random Forest",
    score: 0.918,
    status: "completed" as const,
    accuracy: 0.923,
    precision: 0.889,
    recall: 0.854,
    auc: 0.962,
    trainingTime: "2m 18s",
    features: 24,
    createdAt: "2024-01-15T10:18:00Z"
  },
  {
    id: "m-3",
    model: "Logistic Regression",
    score: 0.854,
    status: "completed" as const,
    accuracy: 0.861,
    precision: 0.832,
    recall: 0.798,
    auc: 0.912,
    trainingTime: "0m 45s",
    features: 24,
    createdAt: "2024-01-15T10:15:00Z"
  },
  {
    id: "m-4",
    model: "LightGBM",
    score: 0.891,
    status: "running" as const,
    accuracy: 0.894,
    precision: 0.867,
    recall: 0.831,
    auc: 0.945,
    trainingTime: "1m 12s",
    features: 24,
    createdAt: "2024-01-15T10:28:00Z"
  },
  {
    id: "m-5",
    model: "CatBoost",
    score: 0.0,
    status: "queued" as const,
    accuracy: 0,
    precision: 0,
    recall: 0,
    auc: 0,
    trainingTime: "-",
    features: 24,
    createdAt: "2024-01-15T10:30:00Z"
  }
];

type SortField = "score" | "accuracy" | "precision" | "recall" | "auc" | "trainingTime";
type ViewMode = "leaderboard" | "cards" | "compare";

const configOptions = {
  problemTypes: ["Classification", "Regression", "Ranking"],
  optimizationMetrics: ["F1 Score", "Accuracy", "AUC-ROC", "Precision", "Recall", "RMSE", "MAE"],
  algorithms: ["Auto (All)", "XGBoost", "LightGBM", "CatBoost", "Random Forest", "Logistic Regression", "Neural Network"]
};

function ProgressBar({ progress, tone = "primary" }: { progress: number; tone?: "primary" | "success" | "warning" }) {
  const colors = {
    primary: "bg-accent",
    success: "bg-success",
    warning: "bg-warning"
  };
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
      <div
        className={cn("h-full rounded-full transition-all duration-500 ease-out", colors[tone])}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}

function ModelCard({ entry, isSelected, onSelect }: { entry: typeof extendedLeaderboard[0]; isSelected: boolean; onSelect: () => void }) {
  const scoreColor = entry.score >= 0.9 ? "text-success" : entry.score >= 0.85 ? "text-warning" : "text-danger";
  
  return (
    <div 
      className={cn(
        "card card-hover relative cursor-pointer p-4 transition-all duration-300",
        isSelected ? "ring-2 ring-accent border-accent" : ""
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-medium text">{entry.model}</h3>
          <p className="mt-0.5 text-xs text-tertiary">
            {entry.status === "completed" ? `${entry.features} features` : entry.status}
          </p>
        </div>
        <div className="text-right">
          <p className={cn("text-2xl font-bold tabular-nums", scoreColor)}>
            {entry.score > 0 ? (entry.score * 100).toFixed(1) : "—"}
            <span className="text-xs font-normal text-tertiary">%</span>
          </p>
          <p className="text-[10px] uppercase tracking-wider text-tertiary">F1 Score</p>
        </div>
      </div>
      
      {entry.status === "running" && (
        <div className="mt-3">
          <ProgressBar progress={72} tone="warning" />
          <p className="mt-1 text-[10px] text-tertiary">Training in progress...</p>
        </div>
      )}
      
      {entry.status === "completed" && (
        <div className="mt-4 grid grid-cols-4 gap-2 border-t border-border-subtle pt-3">
          <div className="text-center">
            <p className="text-xs font-semibold text tabular-nums">{(entry.accuracy * 100).toFixed(1)}%</p>
            <p className="text-[9px] text-tertiary">Accuracy</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-semibold text tabular-nums">{(entry.precision * 100).toFixed(1)}%</p>
            <p className="text-[9px] text-tertiary">Precision</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-semibold text tabular-nums">{(entry.recall * 100).toFixed(1)}%</p>
            <p className="text-[9px] text-tertiary">Recall</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-semibold text tabular-nums">{(entry.auc * 100).toFixed(1)}%</p>
            <p className="text-[9px] text-tertiary">AUC</p>
          </div>
        </div>
      )}
      
      <div className="mt-3 flex items-center justify-between border-t border-border-subtle pt-3">
        <span className="text-[10px] text-tertiary">{entry.trainingTime}</span>
        <button className="btn btn-ghost text-[10px] font-medium opacity-0 transition-opacity group-hover:opacity-100">
          Deploy
        </button>
      </div>
    </div>
  );
}

function ComparisonTable({ entries }: { entries: typeof extendedLeaderboard }) {
  const metrics = [
    { key: "score", label: "F1 Score", format: (v: number) => `${(v * 100).toFixed(1)}%`, higher: true },
    { key: "accuracy", label: "Accuracy", format: (v: number) => `${(v * 100).toFixed(1)}%`, higher: true },
    { key: "precision", label: "Precision", format: (v: number) => `${(v * 100).toFixed(1)}%`, higher: true },
    { key: "recall", label: "Recall", format: (v: number) => `${(v * 100).toFixed(1)}%`, higher: true },
    { key: "auc", label: "AUC-ROC", format: (v: number) => `${(v * 100).toFixed(1)}%`, higher: true },
    { key: "trainingTime", label: "Train Time", format: (v: string) => v, higher: false }
  ] as const;
  
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="pb-3 pr-4 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Model</th>
            {metrics.map(m => (
              <th key={m.key} className="pb-3 px-4 text-[10px] font-semibold uppercase tracking-wider text-tertiary">{m.label}</th>
            ))}
            <th className="pb-3 pl-4 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border-subtle">
          {entries.map((entry, idx) => (
            <tr key={entry.id} className="table-row group">
              <td className="py-3 pr-4">
                <div className="flex items-center gap-2">
                  {idx === 0 && <span className="inline-flex size-5 items-center justify-center rounded-full bg-amber-400 text-[9px] font-bold text-amber-900">★</span>}
                  <span className="font-medium text">{entry.model}</span>
                </div>
              </td>
              {metrics.map(m => {
                const value = entry[m.key as keyof typeof entry] as unknown as number | string;
                const isBest = idx === 0 && m.higher;
                return (
                  <td key={m.key} className="py-3 px-4 tabular-nums">
                    <span className={cn("font-medium", isBest ? "text-success" : "text")}>
                      {m.format(value as never)}
                    </span>
                  </td>
                );
              })}
              <td className="py-3 pl-4">
                <StatusPill label={entry.status} tone={entry.status === "completed" ? "success" : entry.status === "running" ? "warning" : "info"} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FeatureImportance({ model }: { model: string }) {
  const features = [
    { name: "Support Tickets", importance: 0.342, trend: "up" },
    { name: "Account Age", importance: 0.218, trend: "up" },
    { name: "Monthly Charges", importance: 0.156, trend: "down" },
    { name: "Payment History", importance: 0.134, trend: "stable" },
    { name: "Customer Service Calls", importance: 0.089, trend: "up" },
    { name: "Plan Type", importance: 0.061, trend: "stable" }
  ];
  
  const maxImportance = Math.max(...features.map(f => f.importance));
  
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold text">Top Features - {model}</h4>
        <span className="text-[10px] text-tertiary">Gini Importance</span>
      </div>
      <div className="space-y-2">
        {features.map((feature, idx) => (
          <div key={feature.name} className="group">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs text">
                <span className="text-tertiary">{idx + 1}.</span>
                {feature.name}
              </span>
              <span className="text-xs font-medium tabular-nums text">
                {(feature.importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 h-1 overflow-hidden rounded-full bg-surface-2">
              <div
                className="h-full rounded-full bg-gradient-to-r from-accent to-accent/60 transition-all duration-500 group-hover:from-accent/80 group-hover:to-accent"
                style={{ width: `${(feature.importance / maxImportance) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function AutoMLLabPage() {
  const [sortField, setSortField] = useState<SortField>("score");
  const [sortAsc, setSortAsc] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("leaderboard");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    problemType: "Classification",
    optimizationMetric: "F1 Score",
    algorithm: "Auto (All)",
    targetVariable: "Churn",
    trainSplit: 80,
    crossValidation: 5,
    earlyStopping: true
  });
  
  const completedModels = extendedLeaderboard.filter(m => m.status === "completed");
  const sortedLeaderboard = [...extendedLeaderboard].sort((a, b) => {
    const aVal = a[sortField] as number;
    const bVal = b[sortField] as number;
    return sortAsc ? aVal - bVal : bVal - aVal;
  });
  
  const handleModelSelect = (id: string) => {
    setSelectedModels(prev => 
      prev.includes(id) 
        ? prev.filter(m => m !== id)
        : [...prev, id]
    );
  };
  
  const topModel = extendedLeaderboard.find(m => m.status === "completed") || extendedLeaderboard[0];
  
  return (
    <section className="animate-in fade-in duration-300">
      <PageHeader
        title="AutoML Experiments"
        subtitle="Train, compare, and evaluate machine learning models using automated leaderboard tracking."
        actions={
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setShowConfigPanel(!showConfigPanel)}
              className="btn btn-secondary"
            >
              <span className="flex items-center gap-1.5">
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Configure
              </span>
            </button>
            <button className="btn btn-primary">
              <span className="flex items-center gap-1.5">
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Training Run
              </span>
            </button>
          </div>
        }
      />

      {showConfigPanel && (
        <div className="mb-6 animate-in slide-in-from-top-2 duration-200">
          <div className="card overflow-hidden">
            <div className="border-b border-border-subtle bg-surface-2/30 px-5 py-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text">Training Configuration</h3>
              <button onClick={() => setShowConfigPanel(false)} className="text-tertiary hover:text transition-colors">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="grid gap-6 p-5 md:grid-cols-3 lg:grid-cols-6">
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Target Variable</label>
                <input 
                  type="text" 
                  value={trainingConfig.targetVariable}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, targetVariable: e.target.value })}
                  className="input w-full"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Problem Type</label>
                <select 
                  value={trainingConfig.problemType}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, problemType: e.target.value })}
                  className="input w-full"
                >
                  {configOptions.problemTypes.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Optimization Metric</label>
                <select 
                  value={trainingConfig.optimizationMetric}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, optimizationMetric: e.target.value })}
                  className="input w-full"
                >
                  {configOptions.optimizationMetrics.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Algorithm</label>
                <select 
                  value={trainingConfig.algorithm}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, algorithm: e.target.value })}
                  className="input w-full"
                >
                  {configOptions.algorithms.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Train/Val Split</label>
                <div className="flex items-center gap-2">
                  <input 
                    type="range" 
                    min={50}
                    max={95}
                    step={5}
                    value={trainingConfig.trainSplit}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, trainSplit: parseInt(e.target.value) })}
                    className="flex-1 h-2 appearance-none rounded-full bg-surface-2 cursor-pointer accent-accent"
                  />
                  <span className="w-10 text-sm font-medium tabular-nums text">{trainingConfig.trainSplit}%</span>
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Cross Validation</label>
                <div className="flex items-center gap-3">
                  <button 
                    onClick={() => setTrainingConfig({ ...trainingConfig, crossValidation: Math.max(3, trainingConfig.crossValidation - 1) })}
                    className="btn btn-secondary px-2"
                  >
                    −
                  </button>
                  <span className="w-8 text-center text-sm font-medium tabular-nums text">{trainingConfig.crossValidation}</span>
                  <button 
                    onClick={() => setTrainingConfig({ ...trainingConfig, crossValidation: Math.min(10, trainingConfig.crossValidation + 1) })}
                    className="btn btn-secondary px-2"
                  >
                    +
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-4">
        <div className="flex flex-col gap-6 lg:col-span-1">
          <article className="card p-5">
            <h2 className="mb-4 text-sm font-semibold text border-b border-border pb-2">Experiment Summary</h2>
            <div className="space-y-3">
              <MetricTile label="Best Model" value={topModel?.model || "—"} hint={topModel ? `F1: ${(topModel.score * 100).toFixed(1)}%` : undefined} />
              <div className="grid grid-cols-2 gap-3">
                <MetricTile label="Total Runs" value={extendedLeaderboard.length.toString()} />
                <MetricTile label="Completed" value={completedModels.length.toString()} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <MetricTile label="Running" value={extendedLeaderboard.filter(m => m.status === "running").length.toString()} />
                <MetricTile label="Queued" value={extendedLeaderboard.filter(m => m.status === "queued").length.toString()} />
              </div>
            </div>
          </article>
          
          <article className="card p-5">
            <h2 className="mb-4 text-sm font-semibold text border-b border-border pb-2">Configuration</h2>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Target Variable</p>
                <p className="mt-1 font-medium text">{trainingConfig.targetVariable}</p>
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Problem Type</p>
                <p className="mt-1 font-medium text">{trainingConfig.problemType}</p>
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Optimization Metric</p>
                <p className="mt-1 font-medium text">{trainingConfig.optimizationMetric}</p>
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-tertiary">Feature Count</p>
                <p className="mt-1 font-medium text">{topModel?.features || 0} features</p>
              </div>
            </div>
          </article>
          
          <article className="card p-5">
            <h2 className="mb-4 text-sm font-semibold text border-b border-border pb-2">Top Features</h2>
            <FeatureImportance model={topModel?.model || "XGBoost"} />
          </article>
        </div>
        
        <article className="card flex flex-col overflow-hidden lg:col-span-3">
          <div className="border-b border-border bg-surface-2/30 px-5 py-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-sm font-semibold text">Model Leaderboard</h2>
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex rounded-md border border-border overflow-hidden">
                <button
                  onClick={() => setViewMode("leaderboard")}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium transition-colors",
                    viewMode === "leaderboard" ? "bg-accent text-white" : "bg-surface text hover:bg-surface-2"
                  )}
                >
                  Table
                </button>
                <button
                  onClick={() => setViewMode("cards")}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium transition-colors",
                    viewMode === "cards" ? "bg-accent text-white" : "bg-surface text hover:bg-surface-2"
                  )}
                >
                  Cards
                </button>
                <button
                  onClick={() => setViewMode("compare")}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium transition-colors",
                    viewMode === "compare" ? "bg-accent text-white" : "bg-surface text hover:bg-surface-2"
                  )}
                >
                  Compare
                </button>
              </div>
              <select
                value={sortField}
                onChange={(e) => setSortField(e.target.value as SortField)}
                className="input py-1.5 text-xs"
              >
                <option value="score">Sort by F1</option>
                <option value="accuracy">Sort by Accuracy</option>
                <option value="precision">Sort by Precision</option>
                <option value="recall">Sort by Recall</option>
                <option value="auc">Sort by AUC</option>
                <option value="trainingTime">Sort by Time</option>
              </select>
              <button
                onClick={() => setSortAsc(!sortAsc)}
                className="btn btn-ghost p-1.5"
                title={sortAsc ? "Ascending" : "Descending"}
              >
                <svg className={cn("h-4 w-4 transition-transform", sortAsc && "rotate-180")} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
                </svg>
              </button>
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            {viewMode === "leaderboard" && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm tabular-nums whitespace-nowrap">
                  <thead className="sticky top-0 bg-surface/95 backdrop-blur-sm">
                    <tr className="border-b border-border bg-surface-2/50">
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Rank</th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Algorithm</th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">F1 Score</th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Accuracy</th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">AUC-ROC</th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Train Time</th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Status</th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-tertiary">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-subtle">
                    {sortedLeaderboard.map((item, idx) => (
                      <tr 
                        key={item.id} 
                        className={cn(
                          "table-row group transition-colors hover:bg-surface-2/20",
                          selectedModels.includes(item.id) && "bg-accent/5"
                        )}
                      >
                        <td className="px-5 py-3">
                          {idx === 0 ? (
                            <span className="inline-flex size-6 items-center justify-center rounded-full bg-gradient-to-br from-amber-300 to-amber-500 text-[10px] font-bold text-amber-950 shadow-sm">
                              ★
                            </span>
                          ) : (
                            <span className="flex size-6 items-center justify-center rounded-full bg-surface-2 text-xs font-medium text-tertiary">
                              {idx + 1}
                            </span>
                          )}
                        </td>
                        <td className="px-5 py-3">
                          <div className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={selectedModels.includes(item.id)}
                              onChange={() => handleModelSelect(item.id)}
                              className="h-3.5 w-3.5 rounded border-border-subtle text-accent accent-accent"
                            />
                            <span className="font-medium text">{item.model}</span>
                          </div>
                        </td>
                        <td className="px-5 py-3">
                          <span className={cn(
                            "font-semibold tabular-nums",
                            item.score >= 0.9 ? "text-success" : item.score >= 0.85 ? "text-warning" : item.score > 0 ? "text-danger" : "text-tertiary"
                          )}>
                            {item.score > 0 ? `${(item.score * 100).toFixed(1)}%` : "—"}
                          </span>
                        </td>
                        <td className="px-5 py-3 text-tertiary tabular-nums">
                          {item.accuracy > 0 ? `${(item.accuracy * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-5 py-3 text-tertiary tabular-nums">
                          {item.auc > 0 ? `${(item.auc * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-5 py-3 text-tertiary tabular-nums">{item.trainingTime}</td>
                        <td className="px-5 py-3">
                          <StatusPill 
                            label={item.status} 
                            tone={item.status === "completed" ? "success" : item.status === "running" ? "warning" : "info"} 
                          />
                        </td>
                        <td className="px-5 py-3">
                          <div className="flex items-center gap-2">
                            <button className="text-xs font-medium text-accent hover:underline">Deploy</button>
                            <button className="text-xs font-medium text-tertiary hover:text transition-colors">Details</button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            
            {viewMode === "cards" && (
              <div className="p-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  {sortedLeaderboard.map(entry => (
                    <ModelCard
                      key={entry.id}
                      entry={entry}
                      isSelected={selectedModels.includes(entry.id)}
                      onSelect={() => handleModelSelect(entry.id)}
                    />
                  ))}
                </div>
              </div>
            )}
            
            {viewMode === "compare" && selectedModels.length < 2 && (
              <div className="flex h-full flex-col items-center justify-center p-8 text-center">
                <div className="mb-4 rounded-full bg-surface-2 p-4">
                  <svg className="h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="text-base font-semibold text">Select models to compare</h3>
                <p className="mt-1 max-w-xs text-sm text-tertiary">Check the boxes next to at least two models in the leaderboard table to enable comparison view.</p>
              </div>
            )}
            
            {viewMode === "compare" && selectedModels.length >= 2 && (
              <div className="p-4">
                <ComparisonTable entries={extendedLeaderboard.filter(e => selectedModels.includes(e.id))} />
              </div>
            )}
          </div>
        </article>
      </div>
    </section>
  );
}