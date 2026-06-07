import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { datasets, workspace } from "../../lib/mocks/data";

interface Widget {
  id: string;
  type: "line" | "bar" | "pie" | "table" | "kpi";
  title: string;
  xAxis: string;
  yAxis: string;
  aggregation: "sum" | "avg" | "count" | "min" | "max";
  data: { label: string; value: number }[];
}

const mockColumns = ["date", "revenue", "users", "orders", "category", "region"];

function generateMockData(_type: Widget["type"], _xAxis: string, _yAxis: string, aggregation: Widget["aggregation"]): { label: string; value: number }[] {
  const labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"];
  const values = [42000, 48000, 51000, 47000, 54000, 62000];
  
  if (aggregation === "count") {
    return labels.map((label) => ({ label, value: Math.floor(Math.random() * 10) + 1 }));
  }
  if (aggregation === "min") {
    return labels.map((label) => ({ label, value: Math.min(...values) }));
  }
  if (aggregation === "max") {
    return labels.map((label) => ({ label, value: Math.max(...values) }));
  }
  
  return labels.map((label, idx) => ({
    label,
    value: values[idx]
  }));
}

function MiniLineChart({ data, color = "#0ea5e9" }: { data: { label: string; value: number }[]; color?: string }) {
  const max = Math.max(...data.map(d => d.value));
  const min = Math.min(...data.map(d => d.value));
  const range = max - min || 1;
  const width = 100;
  const height = 40;
  const padding = 4;
  
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * (width - padding * 2) + padding;
    const y = height - padding - ((d.value - min) / range) * (height - padding * 2);
    return `${x},${y}`;
  }).join(" ");
  
  const areaPoints = `${padding},${height - padding} ${points} ${width - padding},${height - padding}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
      <defs>
        <linearGradient id={`grad-${color}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline points={areaPoints} fill={`url(#grad-${color})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MiniBarChart({ data }: { data: { label: string; value: number }[] }) {
  const max = Math.max(...data.map(d => d.value));
  
  return (
    <div className="flex h-full items-end gap-1 pb-1">
      {data.map((d, i) => (
        <div key={i} className="flex-1 rounded-t-sm bg-indigo-500/80 hover:bg-indigo-500 transition-colors" style={{ height: `${(d.value / max) * 100}%` }} title={`${d.label}: ${d.value}`}></div>
      ))}
    </div>
  );
}

function KPICard({ data, title: _title }: { data: { label: string; value: number }[]; title: string }) {
  const latest = data[data.length - 1].value;
  const previous = data[data.length - 2].value;
  const percentChange = ((latest - previous) / previous) * 100;
  const isPositive = percentChange >= 0;

  return (
    <div className="flex flex-col justify-center h-full">
      <div className="text-3xl font-light tracking-tight mb-2">{latest.toLocaleString()}</div>
      <div className={`flex items-center text-sm font-medium ${isPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
        <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isPositive ? "M5 10l7-7m0 0l7 7m-7-7v18" : "M19 14l-7 7m0 0l-7-7m7 7V3"} />
        </svg>
        {Math.abs(percentChange).toFixed(1)}% vs last period
      </div>
    </div>
  );
}

export function EDADashboardsPage() {
  const [selectedDatasetId, setSelectedDatasetId] = useState(workspace.activeDatasetId || (datasets && datasets[0]?.id));
  
  const [widgets, setWidgets] = useState<Widget[]>([
    { id: "w1", type: "kpi", title: "Total Users", xAxis: "date", yAxis: "users", aggregation: "sum", data: generateMockData("kpi", "date", "users", "sum") },
    { id: "w2", type: "line", title: "Revenue Trend", xAxis: "date", yAxis: "revenue", aggregation: "sum", data: generateMockData("line", "date", "revenue", "sum") },
    { id: "w4", type: "bar", title: "Orders by Category", xAxis: "category", yAxis: "orders", aggregation: "count", data: generateMockData("bar", "category", "orders", "count") },
  ]);

  const [isAdding, setIsAdding] = useState(false);
  const [newWidget, setNewWidget] = useState<Partial<Widget>>({ type: "line", aggregation: "sum" });

  const handleAddWidget = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWidget.title || !newWidget.xAxis || !newWidget.yAxis) return;

    const widget: Widget = {
      id: `w-${Date.now()}`,
      type: newWidget.type as Widget["type"],
      title: newWidget.title,
      xAxis: newWidget.xAxis,
      yAxis: newWidget.yAxis,
      aggregation: newWidget.aggregation as Widget["aggregation"],
      data: generateMockData(newWidget.type as Widget["type"], newWidget.xAxis, newWidget.yAxis, newWidget.aggregation as Widget["aggregation"])
    };

    setWidgets([...widgets, widget]);
    setIsAdding(false);
    setNewWidget({ type: "line", aggregation: "sum" });
  };

  const removeWidget = (id: string) => {
    setWidgets(widgets.filter(w => w.id !== id));
  };

  return (
    <div className="flex flex-col min-h-[calc(100vh-5rem)]">
      <PageHeader
        title="Dashboards & Reports"
        subtitle="Explore your data visually with interactive ad-hoc dashboards."
        actions={
          <div className="flex items-center gap-3">
            <div className="relative flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-2 border border-subtle hover:border-primary/30 transition-colors text-xs group cursor-pointer">
              <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79 8-4" />
              </svg>
              <select
                className="bg-transparent border-none outline-none appearance-none pr-5 cursor-pointer text font-medium w-full"
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id} className="bg-surface text">{d.name}</option>
                ))}
              </select>
              <svg className="w-3 h-3 text-secondary absolute right-3 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <button onClick={() => setIsAdding(true)} className="btn btn-primary text-sm">
              <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4"/></svg>
              Add Chart
            </button>
          </div>
        }
      />

      <div className="flex-1 p-6 bg-surface-2/20">
        
        {isAdding && (
          <div className="mb-8 bg-white border border-subtle rounded-xl p-5 shadow-sm animate-in fade-in slide-in-from-top-2">
            <div className="flex justify-between items-center mb-4 border-b border-subtle pb-3">
              <h3 className="font-semibold">Configure New Chart</h3>
              <button onClick={() => setIsAdding(false)} className="text-secondary hover:text-black">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12"/></svg>
              </button>
            </div>
            <form onSubmit={handleAddWidget} className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div>
                <label className="block text-xs font-medium text-secondary mb-1">Title</label>
                <input type="text" required value={newWidget.title || ''} onChange={e => setNewWidget({...newWidget, title: e.target.value})} className="input w-full" placeholder="e.g. Daily Active Users" />
              </div>
              <div>
                <label className="block text-xs font-medium text-secondary mb-1">Chart Type</label>
                <select className="input w-full" value={newWidget.type} onChange={e => setNewWidget({...newWidget, type: e.target.value as Widget["type"]})}>
                  <option value="line">Line Chart</option>
                  <option value="bar">Bar Chart</option>
                  <option value="kpi">KPI Metric</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-secondary mb-1">X-Axis (Dimension)</label>
                <select className="input w-full" required value={newWidget.xAxis || ''} onChange={e => setNewWidget({...newWidget, xAxis: e.target.value})}>
                  <option value="">Select column...</option>
                  {mockColumns.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-secondary mb-1">Y-Axis (Measure)</label>
                <select className="input w-full" required value={newWidget.yAxis || ''} onChange={e => setNewWidget({...newWidget, yAxis: e.target.value})}>
                  <option value="">Select column...</option>
                  {mockColumns.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="flex items-end gap-2">
                <div className="flex-1">
                  <label className="block text-xs font-medium text-secondary mb-1">Aggregation</label>
                  <select className="input w-full" value={newWidget.aggregation} onChange={e => setNewWidget({...newWidget, aggregation: e.target.value as Widget["aggregation"]})}>
                    <option value="sum">Sum</option>
                    <option value="avg">Average</option>
                    <option value="count">Count</option>
                    <option value="max">Max</option>
                    <option value="min">Min</option>
                  </select>
                </div>
                <button type="submit" className="btn btn-primary px-4 h-[38px]">Add</button>
              </div>
            </form>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {widgets.map(widget => (
            <div key={widget.id} className="bg-white border border-subtle rounded-xl shadow-sm hover:shadow-md transition-shadow group flex flex-col">
              <div className="flex justify-between items-center p-4 border-b border-subtle">
                <div>
                  <h3 className="font-semibold text-sm">{widget.title}</h3>
                  <p className="text-[11px] text-secondary lowercase">{widget.aggregation}({widget.yAxis}) by {widget.xAxis}</p>
                </div>
                <button onClick={() => removeWidget(widget.id)} className="btn btn-ghost p-1 opacity-0 group-hover:opacity-100 text-secondary hover:text-red-600 transition-all">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                </button>
              </div>
              <div className="flex-1 p-5 min-h-[160px]">
                {widget.type === 'line' && <MiniLineChart data={widget.data} />}
                {widget.type === 'bar' && <MiniBarChart data={widget.data} />}
                {widget.type === 'kpi' && <KPICard data={widget.data} title={widget.title} />}
              </div>
            </div>
          ))}
        </div>

        {widgets.length === 0 && (
          <div className="text-center py-20">
            <div className="w-16 h-16 bg-surface-2 rounded-full flex items-center justify-center mx-auto mb-4 border border-subtle">
              <svg className="w-8 h-8 text-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
            </div>
            <h3 className="text-lg font-medium mb-2">No dashboards yet</h3>
            <p className="text-secondary max-w-sm mx-auto mb-6">Add charts to explore the current dataset visually and extract insights instantly.</p>
            <button onClick={() => setIsAdding(true)} className="btn btn-primary">Create Your First Chart</button>
          </div>
        )}

      </div>
    </div>
  );
}
