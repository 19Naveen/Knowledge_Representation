import { useState, useEffect, useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { PageHeader } from "../../components/shared/PageHeader";
import { useDatasets, useQueryApi, API_BASE, type DatasetPreview } from "../../lib/hooks/useDatasets";
import { useAuthContext } from "../../lib/context/AuthContext";
import { errMessage, readError } from "../../lib/http";

// ── Backend TransformStep shapes (send exactly these keys) ────────────────────
type CastType = "string" | "integer" | "decimal" | "boolean" | "timestamp";
type FilterOp = "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "isnull" | "notnull";
type FillStrategy = "value" | "mean" | "median" | "mode";

type TransformStep =
  | { type: "drop"; column: string }
  | { type: "rename"; column: string; to: string }
  | { type: "cast"; column: string; to_type: CastType }
  | { type: "filter"; column: string; op: FilterOp; value?: string }
  | { type: "fillna"; column: string; strategy: FillStrategy; value?: string };

// ── Staged-preview contract ───────────────────────────────────────────────────
interface SchemaDiff {
  dataset_id: string;
  has_diff: boolean;
  added_columns: string[];
  missing_columns: string[];
  type_changes: { column: string; old_type: string; new_type: string }[];
  suggested_mappings: { column: string; suggested_target: string }[];
}

interface StagedPreview {
  columns: string[];
  dataset_schema: Record<string, string>;
  sample_rows: unknown[][];
  previous_schema: Record<string, string> | null;
  diff: SchemaDiff | null;
}

interface JobResponse {
  id: string;
  dataset_id: string;
  status: string;
  source_type: string;
  error_message?: string;
}

interface ImportState {
  jobId?: string;
  datasetId?: string;
  datasetName?: string;
}

const OP_KIND = ["drop", "rename", "cast", "filter", "fillna"] as const;
type OpKind = (typeof OP_KIND)[number];

const OP_LABEL: Record<OpKind, string> = {
  drop: "Drop Column",
  rename: "Rename Column",
  cast: "Cast Type",
  filter: "Filter Rows",
  fillna: "Fill Missing",
};

const CAST_TYPES: CastType[] = ["string", "integer", "decimal", "boolean", "timestamp"];
const FILTER_OPS: FilterOp[] = ["eq", "ne", "lt", "le", "gt", "ge", "isnull", "notnull"];
const FILL_STRATEGIES: FillStrategy[] = ["value", "mean", "median", "mode"];

function stepLabel(s: TransformStep): string {
  switch (s.type) {
    case "drop": return `drop ${s.column}`;
    case "rename": return `rename ${s.column} → ${s.to}`;
    case "cast": return `cast ${s.column} → ${s.to_type}`;
    case "filter": return `filter ${s.column} ${s.op}${s.value !== undefined ? ` ${s.value}` : ""}`;
    case "fillna": return `fillna ${s.column} (${s.strategy}${s.value !== undefined ? `=${s.value}` : ""})`;
  }
}

export function DataTransformPage() {
  const location = useLocation();
  const importState = (location.state ?? {}) as ImportState;
  const jobId = importState.jobId;
  const importMode = Boolean(jobId);

  if (importMode) {
    return <ImportReview jobId={jobId!} datasetName={importState.datasetName} />;
  }
  return <PreviewMode />;
}

// ══════════════════════════════════════════════════════════════════════════════
// Read-only preview mode (UNCHANGED behavior when there is no jobId)
// ══════════════════════════════════════════════════════════════════════════════
interface UIStep {
  id: string;
  title: string;
  code: string;
}

const operationsDict = [
  { title: "Drop Nulls", desc: "Remove rows containing NaN/Null values." },
  { title: "Filter Rows", desc: "Retain only rows matching a condition." },
  { title: "Fill Missing (Impute)", desc: "Replace missing data with statistical markers." },
  { title: "Cast Type", desc: "Convert a column to another data type." },
  { title: "Z-Score Normalize", desc: "Standardize a numeric distribution." },
];

function PreviewMode() {
  const { datasets, loading: datasetsLoading } = useDatasets();
  const { preview } = useQueryApi();

  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [data, setData] = useState<DatasetPreview | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);

  const [pipeline, setPipeline] = useState<UIStep[]>([{ id: "source", title: "Source Data", code: "SELECT * FROM dataset" }]);
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [selectedOperation, setSelectedOperation] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    if (!selectedDatasetId && datasets.length > 0) setSelectedDatasetId(datasets[0].id);
  }, [datasets, selectedDatasetId]);

  useEffect(() => {
    if (!selectedDatasetId) { setData(null); return; }
    let cancelled = false;
    setPreviewLoading(true);
    setPreviewError("");
    preview(selectedDatasetId, 50)
      .then((p) => { if (!cancelled) setData(p); })
      .catch((e) => { if (!cancelled) { setData(null); setPreviewError(e.message); } })
      .finally(() => { if (!cancelled) setPreviewLoading(false); });
    return () => { cancelled = true; };
  }, [selectedDatasetId, preview]);

  const columns = data ? data.columns : [];

  const openDrawer = useCallback((op: string | null = null, col = "") => {
    setSelectedOperation(op);
    setSelectedColumn(col);
    setDrawerOpen(true);
  }, []);

  const closeDrawer = () => {
    setDrawerOpen(false);
    setTimeout(() => { setSelectedOperation(null); setSearchQuery(""); }, 200);
  };

  const applyOperation = () => {
    if (!selectedOperation) return;
    const codeMap: Record<string, string> = {
      "Drop Nulls": selectedColumn ? `df.dropna(subset=['${selectedColumn}'])` : "df.dropna()",
      "Filter Rows": `df[df['${selectedColumn || columns[0] || "col"}'].notna()]`,
      "Cast Type": `df['${selectedColumn}'] = df['${selectedColumn}'].astype(...)`,
      "Z-Score Normalize": `df['${selectedColumn}'] = zscore(df['${selectedColumn}'])`,
    };
    setPipeline((prev) => [...prev, {
      id: `step-${Date.now()}`,
      title: selectedOperation,
      code: codeMap[selectedOperation] || `df['${selectedColumn || "col"}'].transform(...)`,
    }]);
    setActiveStepIndex(pipeline.length);
    closeDrawer();
  };

  const activeStepCode = pipeline[activeStepIndex]?.code || "";

  return (
    <div className="flex h-[calc(100vh-5rem)] flex-col overflow-hidden bg-[#fafafa]">
      <WfStyles />

      <PageHeader
        title="Data Transformation"
        subtitle="Build a transformation pipeline visually. Preview runs on the latest version of the selected dataset."
        actions={
          <div className="relative flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-2 border border-subtle text-xs">
            <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79 8-4" />
            </svg>
            <select
              className="bg-transparent border-none outline-none appearance-none pr-5 cursor-pointer text font-medium"
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              disabled={datasetsLoading || datasets.length === 0}
            >
              {datasets.length === 0 && <option value="">No datasets</option>}
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
        }
      />

      <div className="wf-container m-4 mb-0 flex-1">
        <div className="wf-topbar">
          <div className="font-semibold text-sm border-r border-[#eaeaea] pr-4 mr-4 flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="text-[#666] hover:text-black transition-colors" title="Toggle Sidebar">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="9" y1="3" x2="9" y2="21"></line>
              </svg>
            </button>
            DataForge
          </div>
          <div className="ml-auto flex gap-3 items-center text-xs text-[#666]">
            <div>Preview rows: <strong className="text-black font-mono">{data?.rows.length ?? 0}</strong></div>
            <div>Cols: <strong className="text-black font-mono">{columns.length}</strong></div>
            <button className="bg-black text-white px-3 py-1.5 rounded font-medium ml-2 opacity-50 cursor-not-allowed" title="Transform execution backend coming soon" disabled>Save &amp; Apply</button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden relative">
          <div className={`wf-sidebar ${sidebarOpen ? "" : "collapsed"}`}>
            <div className="text-xs font-semibold p-4">Execution Graph</div>
            <div className="flex-1 overflow-y-auto px-3">
              {pipeline.map((step, idx) => (
                <div key={step.id} className={`wf-step-item shadow-sm ${idx === activeStepIndex ? "active" : ""}`} onClick={() => setActiveStepIndex(idx)}>
                  <div className="wf-step-icon">{idx + 1}</div>
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] font-medium mb-0.5">{step.title}</div>
                    <div className="text-[11px] text-[#666] truncate font-mono">{step.code}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="p-4 border-t border-[#eaeaea]">
              <button onClick={() => openDrawer(null)} disabled={columns.length === 0}
                className="w-full flex justify-between items-center text-[13px] border border-[#eaeaea] bg-white px-4 py-2 rounded hover:bg-[#fafafa] transition-colors shadow-sm disabled:opacity-40">
                <span className="font-medium">Add Step</span>
                <span className="text-[#666] text-lg leading-none">+</span>
              </button>
            </div>
          </div>

          <div className="flex-1 flex flex-col overflow-hidden bg-white">
            <div className="flex items-center p-2 px-4 border-b border-[#eaeaea] gap-3 bg-[#fafafa]">
              <span className="font-mono text-xs font-semibold text-[#666]">fx</span>
              <input className="flex-1 border border-[#eaeaea] rounded px-3 py-2 font-mono text-xs bg-white text-[#666]" readOnly value={activeStepCode} />
            </div>

            <div className="wf-table-container flex-1 overflow-auto">
              {previewLoading ? (
                <div className="p-8 text-sm text-[#666]">Loading preview…</div>
              ) : previewError ? (
                <div className="p-8 text-sm text-danger">{previewError}</div>
              ) : !data || columns.length === 0 ? (
                <div className="p-8 text-sm text-[#666]">Select a dataset to preview its latest version.</div>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th style={{ width: 50 }}>#</th>
                      {columns.map((col) => (
                        <th key={col} onClick={() => openDrawer(null, col)}>
                          <div className="flex justify-between items-center gap-3">
                            <span className="text-xs font-semibold">{col}</span>
                            <span className="text-[10px] font-mono text-[#666] bg-[#fafafa] px-1 rounded">{data.dataset_schema[col]}</span>
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((row, idx) => (
                      <tr key={idx} className="hover:bg-[#fafafa]">
                        <td className="font-mono text-right text-[#888]">{idx + 1}</td>
                        {row.map((cell, j) => (
                          <td key={j} className={`font-mono ${cell === null || cell === undefined ? "text-[#888] italic" : ""}`}>
                            {cell === null || cell === undefined ? "null" : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          <div className={`wf-drawer ${drawerOpen ? "open" : ""}`}>
            <div className="flex items-center justify-between p-4 border-b border-[#eaeaea]">
              <div className="font-semibold text-[13px]">{selectedOperation ? "Configure Operation" : "Select Operation"}</div>
              <button className="text-[#666] hover:text-[#000]" onClick={closeDrawer}>✕</button>
            </div>

            <div className="flex-1 overflow-y-auto bg-[#fafafa]">
              {!selectedOperation && (
                <div className="p-4 flex flex-col gap-2">
                  {selectedColumn && (
                    <div className="border border-[#eaeaea] rounded p-3 mb-2 bg-white">
                      <div className="text-[10px] uppercase tracking-wider text-[#666] font-semibold mb-1">Target Column</div>
                      <span className="font-mono font-semibold text-sm">{selectedColumn}</span>
                    </div>
                  )}
                  <input type="text" placeholder="Search operations..."
                    className="w-full px-3 py-2 border border-[#eaeaea] rounded text-[13px] outline-none focus:border-black shadow-sm mb-1"
                    value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
                  {operationsDict.filter(op => op.title.toLowerCase().includes(searchQuery.toLowerCase())).map((op) => (
                    <div key={op.title} className="wf-op-card shadow-sm bg-white" onClick={() => setSelectedOperation(op.title)}>
                      <div className="font-semibold text-[13px] mb-1">{op.title}</div>
                      <div className="text-xs text-[#666] leading-tight">{op.desc}</div>
                    </div>
                  ))}
                </div>
              )}

              {selectedOperation && (
                <div className="p-5 flex flex-col gap-4 h-full bg-white">
                  <div className="text-[15px] font-semibold mb-2">{selectedOperation}</div>
                  <div>
                    <label className="block text-xs font-semibold text-[#666] mb-1.5">Target Column</label>
                    <select className="w-full px-3 py-2 border border-[#eaeaea] rounded text-[13px] outline-none mb-3 bg-[#fafafa]"
                      value={selectedColumn} onChange={e => setSelectedColumn(e.target.value)}>
                      <option value="">-- Apply to entire dataframe --</option>
                      {columns.map((c) => <option key={c} value={c}>{c} ({data?.dataset_schema[c]})</option>)}
                    </select>
                  </div>
                  <div className="flex-1" />
                  <div className="flex gap-3 pt-4 border-t border-[#eaeaea]">
                    <button className="flex-1 py-2 text-sm border border-[#eaeaea] rounded font-medium hover:bg-[#fafafa]" onClick={() => setSelectedOperation(null)}>Back</button>
                    <button className="flex-1 py-2 text-sm bg-black text-white rounded font-medium hover:bg-[#333]" onClick={applyOperation}>Add Step</button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// Import-review mode (triggered by a staged jobId handed off from Data Import)
// ══════════════════════════════════════════════════════════════════════════════
function ImportReview({ jobId, datasetName }: { jobId: string; datasetName?: string }) {
  const navigate = useNavigate();
  const { session } = useAuthContext();

  const authHeaders = useCallback(
    (): Record<string, string> => ({ Authorization: `Bearer ${session?.accessToken}` }),
    [session],
  );

  const [preview, setPreview] = useState<StagedPreview | null>(null);
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(true);

  const [steps, setSteps] = useState<TransformStep[]>([]);
  const [committing, setCommitting] = useState(false);
  const [progress, setProgress] = useState<string>("");
  const [commitError, setCommitError] = useState("");

  // Step builder draft
  const [draftOp, setDraftOp] = useState<OpKind>("drop");
  const [draftCol, setDraftCol] = useState("");
  const [draftTo, setDraftTo] = useState("");
  const [draftCast, setDraftCast] = useState<CastType>("string");
  const [draftFilterOp, setDraftFilterOp] = useState<FilterOp>("eq");
  const [draftValue, setDraftValue] = useState("");
  const [draftStrategy, setDraftStrategy] = useState<FillStrategy>("value");

  // ── Fetch staged preview ────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError("");
    fetch(`${API_BASE}/data-ingest/jobs/${jobId}/staged-preview?limit=50`, { headers: authHeaders() })
      .then(async (res) => {
        if (!res.ok) throw new Error(await readError(res, "Failed to load staged preview"));
        return res.json() as Promise<StagedPreview>;
      })
      .then((p) => { if (!cancelled) setPreview(p); })
      .catch((e) => { if (!cancelled) setLoadError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [jobId, authHeaders]);

  const columns = preview?.columns ?? [];
  const diff = preview?.diff && preview.diff.has_diff ? preview.diff : null;

  useEffect(() => {
    if (!draftCol && columns.length > 0) setDraftCol(columns[0]);
  }, [columns, draftCol]);

  const filterNeedsValue = !["isnull", "notnull"].includes(draftFilterOp);
  const fillNeedsValue = draftStrategy === "value";

  function addStep() {
    if (!draftCol) return;
    let step: TransformStep | null = null;
    switch (draftOp) {
      case "drop":
        step = { type: "drop", column: draftCol };
        break;
      case "rename":
        if (!draftTo.trim()) return;
        step = { type: "rename", column: draftCol, to: draftTo.trim() };
        break;
      case "cast":
        step = { type: "cast", column: draftCol, to_type: draftCast };
        break;
      case "filter":
        step = filterNeedsValue
          ? { type: "filter", column: draftCol, op: draftFilterOp, value: draftValue }
          : { type: "filter", column: draftCol, op: draftFilterOp };
        break;
      case "fillna":
        step = fillNeedsValue
          ? { type: "fillna", column: draftCol, strategy: draftStrategy, value: draftValue }
          : { type: "fillna", column: draftCol, strategy: draftStrategy };
        break;
    }
    if (step) {
      setSteps((prev) => [...prev, step!]);
      setDraftTo("");
      setDraftValue("");
    }
  }

  function addMappingStep(suggestedTarget: string, previousName: string) {
    // Rename the incoming column to the previous-schema name.
    setSteps((prev) => [...prev, { type: "rename", column: suggestedTarget, to: previousName }]);
  }

  function removeStep(idx: number) {
    setSteps((prev) => prev.filter((_, i) => i !== idx));
  }

  // ── Commit + poll ───────────────────────────────────────────────────────────
  async function saveAndApply() {
    setCommitting(true);
    setCommitError("");
    setProgress("Dispatching pipeline…");
    try {
      const res = await fetch(`${API_BASE}/data-ingest/jobs/${jobId}/commit`, {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ transforms: steps }),
      });
      if (!res.ok) throw new Error(await readError(res, "Commit failed"));

      setProgress("Running pipeline…");
      await new Promise<void>((resolve, reject) => {
        const poll = setInterval(async () => {
          try {
            const r = await fetch(`${API_BASE}/data-ingest/jobs/${jobId}`, { headers: authHeaders() });
            if (!r.ok) return;
            const job: JobResponse = await r.json();
            if (job.status === "SUCCESS") {
              clearInterval(poll);
              resolve();
            } else if (job.status === "FAILED") {
              clearInterval(poll);
              reject(new Error(errMessage(job.error_message, "Pipeline failed")));
            }
          } catch { /* transient, keep polling */ }
        }, 2000);
      });

      navigate("/app/data-import");
    } catch (e: any) {
      setCommitError(e.message);
      setCommitting(false);
      setProgress("");
    }
  }

  return (
    <div className="flex h-[calc(100vh-5rem)] flex-col overflow-hidden bg-[#fafafa]">
      <WfStyles />

      <PageHeader
        title="Review &amp; Transform"
        subtitle={datasetName ? `Reviewing staged data for “${datasetName}” before committing a new version.` : "Review staged data before committing a new version."}
        actions={
          <span className="text-xs text-text-tertiary font-mono px-3 py-1.5 rounded-lg bg-surface-2 border border-subtle">
            staged job · {jobId.slice(0, 8)}…
          </span>
        }
      />

      <div className="wf-container m-4 mb-0 flex-1">
        <div className="wf-topbar">
          <div className="font-semibold text-sm border-r border-[#eaeaea] pr-4 mr-4">DataForge · Import Review</div>
          <div className="ml-auto flex gap-3 items-center text-xs text-[#666]">
            <div>Rows: <strong className="text-black font-mono">{preview?.sample_rows.length ?? 0}</strong></div>
            <div>Cols: <strong className="text-black font-mono">{columns.length}</strong></div>
            <div>Steps: <strong className="text-black font-mono">{steps.length}</strong></div>
            <button
              onClick={saveAndApply}
              disabled={committing || loading || !!loadError}
              className="bg-black text-white px-3 py-1.5 rounded font-medium ml-2 hover:bg-[#333] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {committing ? "Applying…" : "Save & Apply"}
            </button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden relative">
          {/* Sidebar: diff + transform plan builder */}
          <div className="wf-sidebar" style={{ width: 300, minWidth: 300 }}>
            <div className="flex-1 overflow-y-auto p-3 space-y-4">
              {diff && (
                <div className="border border-[#eaeaea] rounded bg-white p-3 space-y-3">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-[#666]">Schema changed</div>
                  {diff.added_columns.length > 0 && (
                    <div>
                      <div className="text-[10px] font-semibold text-green-700 mb-1">Added</div>
                      <div className="flex flex-wrap gap-1">
                        {diff.added_columns.map((c) => <span key={c} className="text-[10px] font-mono bg-green-50 text-green-700 border border-green-200 rounded px-1.5 py-0.5">{c}</span>)}
                      </div>
                    </div>
                  )}
                  {diff.missing_columns.length > 0 && (
                    <div>
                      <div className="text-[10px] font-semibold text-red-700 mb-1">Missing</div>
                      <div className="flex flex-wrap gap-1">
                        {diff.missing_columns.map((c) => <span key={c} className="text-[10px] font-mono bg-red-50 text-red-700 border border-red-200 rounded px-1.5 py-0.5">{c}</span>)}
                      </div>
                    </div>
                  )}
                  {diff.type_changes.length > 0 && (
                    <div>
                      <div className="text-[10px] font-semibold text-amber-700 mb-1">Type changes</div>
                      {diff.type_changes.map((c) => (
                        <div key={c.column} className="text-[10px] font-mono"><b>{c.column}</b> {c.old_type} → {c.new_type}</div>
                      ))}
                    </div>
                  )}
                  {diff.suggested_mappings.length > 0 && (
                    <div>
                      <div className="text-[10px] font-semibold text-[#666] mb-1">Suggested mappings</div>
                      <div className="space-y-1.5">
                        {diff.suggested_mappings.map((m) => (
                          <button
                            key={`${m.column}-${m.suggested_target}`}
                            onClick={() => addMappingStep(m.suggested_target, m.column)}
                            className="w-full text-left text-[11px] font-mono border border-[#eaeaea] rounded px-2 py-1.5 hover:border-black hover:bg-[#fafafa] transition-colors"
                            title="Add a rename step mapping the incoming column to the previous name"
                          >
                            <span className="text-[#888]">rename</span> {m.suggested_target} → {m.column}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Transform plan */}
              <div>
                <div className="text-[11px] font-bold uppercase tracking-wider text-[#666] mb-2">Transform plan</div>
                {steps.length === 0 ? (
                  <div className="text-[11px] text-[#888] italic px-1">No steps. Data commits as-is.</div>
                ) : (
                  <div className="space-y-1.5">
                    {steps.map((s, idx) => (
                      <div key={idx} className="flex items-center gap-2 border border-[#eaeaea] rounded bg-white px-2 py-1.5">
                        <span className="wf-step-icon">{idx + 1}</span>
                        <span className="flex-1 min-w-0 text-[11px] font-mono truncate">{stepLabel(s)}</span>
                        <button onClick={() => removeStep(idx)} className="text-[#888] hover:text-red-600 text-xs leading-none">✕</button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Step builder */}
            <div className="p-3 border-t border-[#eaeaea] bg-white space-y-2">
              <div className="text-[11px] font-bold uppercase tracking-wider text-[#666]">Add step</div>
              <select
                className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none"
                value={draftOp}
                onChange={(e) => setDraftOp(e.target.value as OpKind)}
              >
                {OP_KIND.map((k) => <option key={k} value={k}>{OP_LABEL[k]}</option>)}
              </select>

              <select
                className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none font-mono"
                value={draftCol}
                onChange={(e) => setDraftCol(e.target.value)}
              >
                {columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>

              {draftOp === "rename" && (
                <input
                  className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none font-mono"
                  placeholder="new name"
                  value={draftTo}
                  onChange={(e) => setDraftTo(e.target.value)}
                />
              )}

              {draftOp === "cast" && (
                <select
                  className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none"
                  value={draftCast}
                  onChange={(e) => setDraftCast(e.target.value as CastType)}
                >
                  {CAST_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              )}

              {draftOp === "filter" && (
                <>
                  <select
                    className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none"
                    value={draftFilterOp}
                    onChange={(e) => setDraftFilterOp(e.target.value as FilterOp)}
                  >
                    {FILTER_OPS.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                  {filterNeedsValue && (
                    <input
                      className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none font-mono"
                      placeholder="value"
                      value={draftValue}
                      onChange={(e) => setDraftValue(e.target.value)}
                    />
                  )}
                </>
              )}

              {draftOp === "fillna" && (
                <>
                  <select
                    className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none"
                    value={draftStrategy}
                    onChange={(e) => setDraftStrategy(e.target.value as FillStrategy)}
                  >
                    {FILL_STRATEGIES.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                  {fillNeedsValue && (
                    <input
                      className="w-full px-2 py-1.5 border border-[#eaeaea] rounded text-[12px] outline-none font-mono"
                      placeholder="fill value"
                      value={draftValue}
                      onChange={(e) => setDraftValue(e.target.value)}
                    />
                  )}
                </>
              )}

              <button
                onClick={addStep}
                disabled={columns.length === 0}
                className="w-full text-[12px] bg-black text-white rounded px-3 py-2 font-medium hover:bg-[#333] disabled:opacity-40"
              >
                Add to plan
              </button>
            </div>
          </div>

          {/* Preview table */}
          <div className="flex-1 flex flex-col overflow-hidden bg-white">
            {commitError && (
              <div className="px-4 py-2 text-[12px] text-red-700 bg-red-50 border-b border-red-200 font-mono">{commitError}</div>
            )}
            {committing && (
              <div className="px-4 py-2 text-[12px] text-[#666] bg-[#fafafa] border-b border-[#eaeaea]">{progress}</div>
            )}
            <div className="wf-table-container flex-1 overflow-auto">
              {loading ? (
                <div className="p-8 text-sm text-[#666]">Loading staged preview…</div>
              ) : loadError ? (
                <div className="p-8 text-sm text-danger">{loadError}</div>
              ) : !preview || columns.length === 0 ? (
                <div className="p-8 text-sm text-[#666]">No staged data to preview.</div>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th style={{ width: 50 }}>#</th>
                      {columns.map((col) => (
                        <th key={col}>
                          <div className="flex justify-between items-center gap-3">
                            <span className="text-xs font-semibold">{col}</span>
                            <span className="text-[10px] font-mono text-[#666] bg-[#fafafa] px-1 rounded">{preview.dataset_schema[col]}</span>
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.sample_rows.map((row, idx) => (
                      <tr key={idx} className="hover:bg-[#fafafa]">
                        <td className="font-mono text-right text-[#888]">{idx + 1}</td>
                        {row.map((cell, j) => (
                          <td key={j} className={`font-mono ${cell === null || cell === undefined ? "text-[#888] italic" : ""}`}>
                            {cell === null || cell === undefined ? "null" : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Shared DataForge styling (extracted so both modes render identically).
function WfStyles() {
  return (
    <style dangerouslySetInnerHTML={{ __html: `
      .wf-container { height: calc(100vh - 120px); border: 1px solid #eaeaea; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; position: relative; background: #fff; }
      .wf-topbar { display: flex; align-items: center; padding: 0 16px; height: 52px; border-bottom: 1px solid #eaeaea; background: #fff; flex-shrink: 0; }
      .wf-sidebar { flex-shrink: 0; border-right: 1px solid #eaeaea; display: flex; flex-direction: column; background: #fafafa; z-index: 2; transition: width .3s ease, min-width .3s ease; width: 220px; min-width: 220px; }
      .wf-sidebar.collapsed { width: 0; min-width: 0; border-right: none; overflow: hidden; opacity: 0; pointer-events: none; }
      .wf-step-item { display: flex; align-items: flex-start; gap: 12px; padding: 12px 16px; border-bottom: 1px solid #eaeaea; cursor: pointer; transition: background .2s; }
      .wf-step-item:hover { background: #fff; }
      .wf-step-item.active { background: #fff; box-shadow: inset 2px 0 0 #000; }
      .wf-step-icon { width: 20px; height: 20px; border-radius: 4px; background: #eaeaea; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; flex-shrink: 0; }
      .wf-step-item.active .wf-step-icon { background: #000; color: #fff; }
      .wf-table-container { overflow: auto; background: #fff; }
      .wf-table-container table { width: 100%; border-collapse: separate; border-spacing: 0; text-align: left; }
      .wf-table-container th { position: sticky; top: 0; background: #fff; box-shadow: 0 1px 0 #eaeaea; z-index: 2; padding: 12px 16px; border-right: 1px solid #eaeaea; cursor: pointer; white-space: nowrap; }
      .wf-table-container th:hover { background: #fafafa; }
      .wf-table-container td { padding: 10px 16px; font-size: 13px; border-right: 1px solid #eaeaea; border-bottom: 1px solid #eaeaea; white-space: nowrap; }
      .wf-op-card { padding: 12px; border: 1px solid #eaeaea; border-radius: 6px; cursor: pointer; margin-bottom: 8px; }
      .wf-op-card:hover { border-color: #000; }
      .wf-drawer { position: absolute; top: 52px; right: -320px; width: 320px; bottom: 0; background: #fff; border-left: 1px solid #eaeaea; transition: right .3s ease; display: flex; flex-direction: column; z-index: 20; box-shadow: -4px 0 12px rgba(0,0,0,0.05); }
      .wf-drawer.open { right: 0; }
    `}} />
  );
}
