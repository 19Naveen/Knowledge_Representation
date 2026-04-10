import { useState, useMemo, useCallback } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { StatusPill } from "../../components/shared/StatusPill";
import { datasets, workspace } from "../../lib/mocks/data";

type SortField = "name" | "type" | "nulls" | "unique" | "quality";
type SortDir = "asc" | "desc";

interface ColumnDef {
  name: string;
  type: string;
  nullRate: number;
  unique: string;
  distribution: string;
  quality: "good" | "warning" | "danger";
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
}

const mockColumnData: ColumnDef[] = [
  { name: "customer_id", type: "UUID", nullRate: 0, unique: "100%", distribution: "Unique", quality: "good" },
  { name: "age", type: "Int64", nullRate: 1.2, unique: "78%", distribution: "Normal (μ=34, σ=12)", quality: "warning", min: 18, max: 89, mean: 34, std: 12 },
  { name: "churned", type: "Boolean", nullRate: 0, unique: "2%", distribution: "Imbalanced (82/18)", quality: "good" },
  { name: "tenure_months", type: "Int64", nullRate: 0.3, unique: "65%", distribution: "Right-skewed", quality: "good", min: 1, max: 72, mean: 24, std: 18 },
  { name: "monthly_charge", type: "Float64", nullRate: 0, unique: "45%", distribution: " multimodal", quality: "good", min: 20, max: 320, mean: 85, std: 45 },
  { name: "total_charges", type: "Float64", nullRate: 2.1, unique: "92%", distribution: "Right-skewed", quality: "warning", min: 0, max: 12000, mean: 2200, std: 1800 },
  { name: "payment_method", type: "Category", nullRate: 0.1, unique: "4%", distribution: "Credit Card (58%)", quality: "good" },
  { name: "contract_type", type: "Category", nullRate: 0, unique: "3%", distribution: "Month-to-month (55%)", quality: "good" },
  { name: "num_support_tickets", type: "Int64", nullRate: 0.5, unique: "15%", distribution: "Pareto", quality: "warning", min: 0, max: 45, mean: 2.3, std: 4.1 },
  { name: "last_interaction_days", type: "Int64", nullRate: 0.2, unique: "88%", distribution: "Exponential", quality: "good", min: 1, max: 180, mean: 28, std: 35 },
];

function QualityBadge({ quality }: { quality: ColumnDef["quality"] }) {
  return (
    <StatusPill 
      label={quality === "good" ? "Healthy" : quality === "warning" ? "Review" : "Critical"} 
      tone={quality === "good" ? "success" : quality === "warning" ? "warning" : "danger"}
      size="xs"
    />
  );
}

function ColumnVisibilityMenu({
  columns,
  visible,
  onToggle,
}: {
  columns: ColumnDef[];
  visible: Set<string>;
  onToggle: (name: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="btn-ghost flex items-center gap-2"
      >
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 7h13M3 12h16M3 17h10" />
        </svg>
        Columns
        <span className="rounded bg-surface-2 px-1.5 py-0.5 text-xs font-medium text-text-tertiary">{visible.size}/{columns.length}</span>
      </button>
      {isOpen && (
        <div className="absolute right-0 top-full z-50 mt-2 w-56 rounded-xl border border-border bg-surface shadow-xl animate-in fade-in slide-in-from-top-2 duration-150">
          <div className="border-b border-border px-3 py-2">
            <p className="text-xs font-semibold text-text-tertiary">Toggle Columns</p>
          </div>
          <div className="max-h-64 overflow-y-auto p-1">
            {columns.map((col) => (
              <label
                key={col.name}
                className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-surface-2"
              >
                <input
                  type="checkbox"
                  checked={visible.has(col.name)}
                  onChange={() => onToggle(col.name)}
                  className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                />
                <span className="flex-1 truncate text-sm font-medium text-text">{col.name}</span>
                <QualityBadge quality={col.quality} />
              </label>
            ))}
          </div>
          <div className="border-t border-border px-2 py-2">
            <button
              onClick={() => columns.forEach(c => onToggle(c.name))}
              className="w-full rounded-lg bg-surface-2 px-3 py-1.5 text-xs font-medium text-text transition-colors hover:bg-surface-2/80"
            >
              Select All
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function StatisticBar({
  label,
  value,
  subValue,
  color = "bg-primary",
}: {
  label: string;
  value: string;
  subValue?: string;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border-subtle last:border-0">
      <span className="text-sm text-text-tertiary">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-text tabular-nums">{value}</span>
        {subValue && <span className="text-xs text-text-tertiary">({subValue})</span>}
        <div className={`h-1.5 w-8 rounded-full ${color}`} />
      </div>
    </div>
  );
}

function DistributionBar({ percentage, color = "bg-primary" }: { percentage: number; color?: string }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
      <div
        className={`h-full rounded-full ${color} transition-all duration-500 ease-out`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}

export function DataStudioPage() {
  const [selectedDatasetId, setSelectedDatasetId] = useState(workspace.activeDatasetId);
  const [sortField, setSortField] = useState<SortField>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [filter, setFilter] = useState("");
  const [hiddenColumns, setHiddenColumns] = useState<Set<string>>(new Set());
  const [selectedColumn, setSelectedColumn] = useState<ColumnDef | null>(null);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  const selectedDataset = datasets.find(d => d.id === selectedDatasetId) ?? datasets[0];

  const visibleColumns = useMemo(() => {
    const visible = new Set<string>();
    mockColumnData.forEach(col => {
      if (!hiddenColumns.has(col.name)) visible.add(col.name);
    });
    return visible;
  }, [hiddenColumns]);

  const toggleColumn = useCallback((name: string) => {
    setHiddenColumns(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const sortedColumns = useMemo(() => {
    let cols = [...mockColumnData];
    if (filter) {
      cols = cols.filter(c => c.name.toLowerCase().includes(filter.toLowerCase()) || c.type.toLowerCase().includes(filter.toLowerCase()));
    }
    cols.sort((a, b) => {
      let cmp = 0;
      if (sortField === "name") cmp = a.name.localeCompare(b.name);
      else if (sortField === "type") cmp = a.type.localeCompare(b.type);
      else if (sortField === "nulls") cmp = a.nullRate - b.nullRate;
      else if (sortField === "unique") cmp = a.unique.localeCompare(b.unique);
      else if (sortField === "quality") {
        const q = { good: 0, warning: 1, danger: 2 };
        cmp = q[a.quality] - q[b.quality];
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return cols;
  }, [sortField, sortDir, filter]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <span className="opacity-0 group-hover:opacity-50">↕</span>;
    return <span className="text-primary">{sortDir === "asc" ? "↑" : "↓"}</span>;
  };

  const stats = useMemo(() => {
    const cols = mockColumnData;
    const totalNulls = cols.reduce((sum, c) => sum + c.nullRate, 0) / cols.length;
    const warningCols = cols.filter(c => c.quality === "warning").length;
    const dangerCols = cols.filter(c => c.quality === "danger").length;
    const avgUnique = cols.reduce((sum, c) => sum + parseFloat(c.unique), 0) / cols.length;
    return { totalNulls, warningCols, dangerCols, avgUnique };
  }, []);

  return (
    <section className="space-y-6">
      <PageHeader
        title="Data Studio"
        subtitle="Connect, profile, and prepare data with enterprise-grade quality monitoring."
        actions={
          <div className="flex items-center gap-3">
            <div className="relative">
              <select
                value={selectedDatasetId}
                onChange={e => setSelectedDatasetId(e.target.value)}
                className="input -appearance-none pr-10"
              >
                {datasets.map(d => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
              <svg className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <button className="btn-primary">
              <span className="flex items-center gap-2">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Auto-Clean
              </span>
            </button>
          </div>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card card-hover">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Health Score</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-text tabular-nums">{selectedDataset.qualityScore}</p>
            </div>
            <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${selectedDataset.qualityScore >= 85 ? "bg-success-muted" : selectedDataset.qualityScore >= 70 ? "bg-warning-muted" : "bg-danger-muted"}`}>
              <svg className={`h-5 w-5 ${selectedDataset.qualityScore >= 85 ? "text-success" : selectedDataset.qualityScore >= 70 ? "text-warning" : "text-danger"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <div className="mt-4">
            <DistributionBar percentage={selectedDataset.qualityScore} color="bg-success" />
          </div>
          <p className="mt-2 text-xs text-text-tertiary">{selectedDataset.freshness === "daily" ? "Updated daily" : "Updated weekly"}</p>
        </div>

        <div className="card card-hover">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Total Rows</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-text tabular-nums">{selectedDataset.rows.toLocaleString()}</p>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary-muted">
              <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
              </svg>
            </div>
          </div>
          <div className="mt-4 flex items-center gap-2 text-xs text-text-tertiary">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-primary" />
              {selectedDataset.columns} columns
            </span>
            <span>•</span>
            <span>~{Math.round(selectedDataset.rows / 1000)}K</span>
          </div>
        </div>

        <div className="card card-hover">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Quality Issues</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-text tabular-nums">{selectedDataset.issues.length}</p>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-warning-muted">
              <svg className="h-5 w-5 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
          </div>
          <div className="mt-4 flex gap-2">
            {stats.dangerCols > 0 && (
              <StatusPill label={`${stats.dangerCols} critical`} tone="danger" size="xs" />
            )}
            <StatusPill label={`${stats.warningCols} warning`} tone="warning" size="xs" />
          </div>
        </div>

        <div className="card card-hover">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Data Completeness</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-text tabular-nums">{(100 - stats.totalNulls).toFixed(1)}%</p>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-info-muted">
              <svg className="h-5 w-5 text-info" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
          </div>
          <div className="mt-4">
            <DistributionBar percentage={100 - stats.totalNulls} color="bg-info" />
          </div>
          <p className="mt-2 text-xs text-text-tertiary">{stats.totalNulls.toFixed(1)}% null rate</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="card overflow-hidden">
            <div className="flex flex-col gap-4 border-b border-border-subtle bg-surface-2/20 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <h2 className="text-base font-semibold text-text">Schema Profiler</h2>
                <StatusPill label={`${mockColumnData.length} columns`} tone="info" />
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Search columns..."
                    value={filter}
                    onChange={e => setFilter(e.target.value)}
                    className="input h-8 w-40 pl-8"
                  />
                  <svg className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
                <ColumnVisibilityMenu columns={mockColumnData} visible={visibleColumns} onToggle={toggleColumn} />
                <button className="btn-ghost flex items-center gap-2">
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Export
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-surface-2/50 text-xs font-semibold uppercase tracking-wider text-text-tertiary">
                  <tr>
                    {[
                      { key: "name", label: "Column" },
                      { key: "type", label: "Type" },
                      { key: "nulls", label: "Nulls" },
                      { key: "unique", label: "Unique" },
                      { key: "quality", label: "Quality" },
                    ].map(({ key, label }) => (
                      <th
                        key={key}
                        onClick={() => handleSort(key as SortField)}
                        className={`cursor-pointer px-4 py-3 transition-colors hover:bg-surface-2 ${key === "quality" ? "text-center" : ""}`}
                      >
                        <span className="flex items-center gap-1.5 group">
                          {label}
                          <SortIcon field={key as SortField} />
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {sortedColumns.map((col, idx) => (
                    <tr
                      key={col.name}
                      onClick={() => setSelectedColumn(col)}
                      onMouseEnter={() => setHoveredRow(col.name)}
                      onMouseLeave={() => setHoveredRow(null)}
                      className={`cursor-pointer transition-all table-row ${hoveredRow === col.name ? "bg-surface-2/50" : ""} ${selectedColumn?.name === col.name ? "ring-1 ring-inset ring-primary/30 bg-primary-muted" : ""}`}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <span className={`flex h-6 w-6 items-center justify-center rounded-md text-xs font-bold ${col.name === "customer_id" ? "bg-primary-muted text-primary" : "bg-surface-2 text-text-tertiary"}`}>
                            {idx + 1}
                          </span>
                          <span className="font-medium text-text">{col.name}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center rounded-lg bg-surface-2 px-2 py-1 text-xs font-medium text-text-tertiary">
                          {col.type}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className={`font-medium tabular-nums ${col.nullRate > 0 ? "text-danger" : "text-text"}`}>
                            {col.nullRate.toFixed(1)}%
                          </span>
                          {col.nullRate > 0 && (
                            <div className="h-1.5 w-8 overflow-hidden rounded-full bg-surface-2">
                              <div className="h-full rounded-full bg-danger" style={{ width: `${Math.min(col.nullRate * 5, 100)}%` }} />
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="text-text tabular-nums">{col.unique}</span>
                          <div className="h-1.5 w-8 overflow-hidden rounded-full bg-surface-2">
                            <div className="h-full rounded-full bg-info" style={{ width: col.unique }} />
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <QualityBadge quality={col.quality} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="card overflow-hidden">
            <div className="border-b border-border-subtle bg-surface-2/20 px-4 py-3">
              <h2 className="text-base font-semibold text-text">Column Inspector</h2>
            </div>
            {selectedColumn ? (
              <div className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-lg font-semibold text-text">{selectedColumn.name}</p>
                    <p className="text-sm text-text-tertiary">{selectedColumn.type}</p>
                  </div>
                  <QualityBadge quality={selectedColumn.quality} />
                </div>
                <div className="mt-4 space-y-1">
                  <StatisticBar label="Missing" value={`${selectedColumn.nullRate.toFixed(1)}%`} color="bg-danger" />
                  <StatisticBar label="Unique Values" value={selectedColumn.unique} color="bg-info" />
                  {selectedColumn.mean !== undefined && (
                    <>
                      <StatisticBar label="Mean" value={selectedColumn.mean.toString()} subValue={`σ=${selectedColumn.std}`} color="bg-primary" />
                      <StatisticBar label="Range" value={`${selectedColumn.min} - ${selectedColumn.max}`} color="bg-success" />
                    </>
                  )}
                </div>
                <div className="mt-4 rounded-lg bg-surface-2/50 p-3">
                  <p className="text-xs font-semibold text-text-tertiary">Distribution</p>
                  <p className="mt-1 text-sm text-text">{selectedColumn.distribution}</p>
                </div>
                <button className="btn-primary mt-4 w-full">
                  Create Transform
                </button>
              </div>
            ) : (
              <div className="flex h-48 items-center justify-center p-8 text-center">
                <div>
                  <svg className="mx-auto h-10 w-10 text-text-tertiary/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                  </svg>
                  <p className="mt-3 text-sm font-medium text-text">Select a column</p>
                  <p className="mt-1 text-xs text-text-tertiary">Click any row to inspect</p>
                </div>
              </div>
            )}
          </div>

          <div className="card overflow-hidden">
            <div className="flex items-center justify-between border-b border-border-subtle bg-surface-2/20 px-4 py-3">
              <h2 className="text-base font-semibold text-text">Quality Issues</h2>
              <StatusPill label={`${selectedDataset.issues.length} open`} tone="warning" />
            </div>
            <div className="divide-y divide-border-subtle">
              {selectedDataset.issues.length > 0 ? (
                selectedDataset.issues.map((issue, idx) => (
                  <div key={idx} className="flex items-start gap-3 p-4">
                    <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-danger-muted">
                      <span className="h-1.5 w-1.5 rounded-full bg-danger" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text">{issue}</p>
                      <button className="mt-1 text-xs font-medium text-primary hover:underline">
                        Fix automatically →
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex items-center justify-center p-8">
                  <div className="flex items-center gap-2 text-success">
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-sm font-medium">All issues resolved</span>
                  </div>
                </div>
              )}
            </div>
            {selectedDataset.issues.length > 0 && (
              <div className="border-t border-border-subtle p-3">
                <button className="btn-secondary w-full">
                  View All Issues
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}