import { useState, useCallback } from "react";
import { PageHeader } from "../../components/shared/PageHeader";

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
    value: aggregation === "sum" || aggregation === "avg" ? values[idx] : values[idx]
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
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={areaPoints} fill={`url(#gradient-${color})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function MiniBarChart({ data }: { data: { label: string; value: number }[] }) {
  const max = Math.max(...data.map(d => d.value));
  const width = 100;
  const height = 40;
  const barWidth = (width / data.length) - 4;
  const colors = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"];
  
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="none">
      {data.map((d, i) => {
        const barHeight = (d.value / max) * (height - 8);
        const x = (i * (width / data.length)) + 2;
        const y = height - 4 - barHeight;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barWidth}
            height={barHeight}
            rx={2}
            fill={colors[i % colors.length]}
            className="opacity-90 hover:opacity-100 transition-opacity"
          />
        );
      })}
    </svg>
  );
}

function MiniPieChart({ data }: { data: { label: string; value: number }[] }) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  const colors = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"];
  let cumulative = 0;
  
  const slices = data.map((d, i) => {
    const startAngle = (cumulative / total) * 360;
    const endAngle = ((cumulative + d.value) / total) * 360;
    cumulative += d.value;
    
    const startRad = (startAngle - 90) * (Math.PI / 180);
    const endRad = (endAngle - 90) * (Math.PI / 180);
    
    const x1 = 50 + 35 * Math.cos(startRad);
    const y1 = 50 + 35 * Math.sin(startRad);
    const x2 = 50 + 35 * Math.cos(endRad);
    const y2 = 50 + 35 * Math.sin(endRad);
    
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    
    return (
      <path
        key={i}
        d={`M 50 50 L ${x1} ${y1} A 35 35 0 ${largeArc} 1 ${x2} ${y2} Z`}
        fill={colors[i % colors.length]}
        className="hover:opacity-80 transition-opacity"
      />
    );
  });
  
  return (
    <svg viewBox="0 0 100 100" className="w-full h-full">
      {slices}
    </svg>
  );
}

function KPICard({ data, title }: { data: { label: string; value: number }[]; title: string }) {
  const latest = data[data.length - 1]?.value || 0;
  const change = data.length > 1 ? ((latest - data[0].value) / data[0].value * 100).toFixed(1) : "0";
  
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <span className="text-3xl font-bold text">{latest.toLocaleString()}</span>
      <span className="text-xs text-tertiary mt-1">{title}</span>
      <span className={`text-xs font-medium mt-2 ${Number(change) >= 0 ? 'text-success' : 'text-danger'}`}>
        {Number(change) >= 0 ? '+' : ''}{change}% from period
      </span>
    </div>
  );
}

function DataTable({ data }: { data: { label: string; value: number }[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-subtle">
          <th className="text-left py-2 text-tertiary font-medium">Label</th>
          <th className="text-right py-2 text-tertiary font-medium">Value</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i} className="border-b border-subtle/50 hover:bg-surface-2 transition-colors">
            <td className="py-2 text">{row.label}</td>
            <td className="py-2 text-right text">{row.value.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function WidgetRenderer({ widget }: { widget: Widget }) {
  const renderContent = () => {
    switch (widget.type) {
      case "line":
        return <MiniLineChart data={widget.data} color="#0ea5e9" />;
      case "bar":
        return <MiniBarChart data={widget.data} />;
      case "pie":
        return <MiniPieChart data={widget.data} />;
      case "table":
        return <DataTable data={widget.data} />;
      case "kpi":
        return <KPICard data={widget.data} title={widget.yAxis} />;
      default:
        return null;
    }
  };
  
  const heightClass = widget.type === "table" ? "h-40" : "h-32";
  
  return (
    <div className="card p-0 overflow-hidden">
      <div className="px-4 py-3 border-b border-subtle flex items-center justify-between">
        <h3 className="text-sm font-semibold text">{widget.title}</h3>
        <span className="text-[10px] text-tertiary uppercase">{widget.type}</span>
      </div>
      <div className={`p-4 ${heightClass}`}>
        {renderContent()}
      </div>
    </div>
  );
}

function WidgetConfigPanel({
  onAddWidget,
  onClose
}: {
  onAddWidget: (widget: Widget) => void;
  onClose: () => void;
}) {
  const [widgetType, setWidgetType] = useState<Widget["type"]>("line");
  const [title, setTitle] = useState("");
  const [xAxis, setXAxis] = useState("date");
  const [yAxis, setYAxis] = useState("revenue");
  const [aggregation, setAggregation] = useState<Widget["aggregation"]>("sum");
  
  const handleAdd = () => {
    if (!title) return;
    const widget: Widget = {
      id: `widget-${Date.now()}`,
      type: widgetType,
      title,
      xAxis,
      yAxis,
      aggregation,
      data: generateMockData(widgetType, xAxis, yAxis, aggregation)
    };
    onAddWidget(widget);
    onClose();
  };
  
  const widgetTypes = [
    { id: "line", label: "Line Chart", icon: "📈" },
    { id: "bar", label: "Bar Chart", icon: "📊" },
    { id: "pie", label: "Pie Chart", icon: "🥧" },
    { id: "table", label: "Table", icon: "📋" },
    { id: "kpi", label: "KPI Card", icon: "🎯" }
  ] as const;
  
  return (
    <div className="card p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text">Add Widget</h3>
        <button onClick={onClose} className="text-tertiary hover:text">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      <div>
        <label className="text-xs font-medium text-tertiary block mb-1.5">Widget Type</label>
        <div className="grid grid-cols-5 gap-2">
          {widgetTypes.map((wt) => (
            <button
              key={wt.id}
              onClick={() => setWidgetType(wt.id)}
              className={`p-2 rounded-lg border text-center transition-all ${widgetType === wt.id ? 'border-accent bg-accent/10' : 'border-subtle hover:border-tertiary'}`}
            >
              <span className="block text-lg mb-1">{wt.icon}</span>
              <span className="text-[10px] text-tertiary">{wt.label}</span>
            </button>
          ))}
        </div>
      </div>
      
      <div>
        <label className="text-xs font-medium text-tertiary block mb-1.5">Title</label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Enter widget title"
          className="input w-full"
        />
      </div>
      
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs font-medium text-tertiary block mb-1.5">X-Axis</label>
          <select value={xAxis} onChange={(e) => setXAxis(e.target.value)} className="input w-full">
            {mockColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs font-medium text-tertiary block mb-1.5">Y-Axis</label>
          <select value={yAxis} onChange={(e) => setYAxis(e.target.value)} className="input w-full">
            {mockColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      </div>
      
      <div>
        <label className="text-xs font-medium text-tertiary block mb-1.5">Aggregation</label>
        <select value={aggregation} onChange={(e) => setAggregation(e.target.value as Widget["aggregation"])} className="input w-full">
          <option value="sum">Sum</option>
          <option value="avg">Average</option>
          <option value="count">Count</option>
          <option value="min">Minimum</option>
          <option value="max">Maximum</option>
        </select>
      </div>
      
      <button onClick={handleAdd} className="btn btn-primary w-full">
        Add Widget
      </button>
    </div>
  );
}

function DashboardBuilder({
  widgets,
  onAddWidget,
  onRemoveWidget,
  onSaveDashboard,
  onShare
}: {
  widgets: Widget[];
  onAddWidget: (widget: Widget) => void;
  onRemoveWidget: (id: string) => void;
  onSaveDashboard: (name: string) => void;
  onShare: () => void;
}) {
  const [showConfig, setShowConfig] = useState(false);
  const [dashboardName, setDashboardName] = useState("");
  const [showSaveModal, setShowSaveModal] = useState(false);
  
  const handleSave = () => {
    if (dashboardName) {
      onSaveDashboard(dashboardName);
      setDashboardName("");
      setShowSaveModal(false);
    }
  };
  
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="btn btn-primary flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Widget
          </button>
          {widgets.length > 0 && (
            <>
              <button onClick={() => setShowSaveModal(true)} className="btn btn-secondary flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                </svg>
                Save Dashboard
              </button>
              <button onClick={onShare} className="btn btn-ghost flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
                Share
              </button>
            </>
          )}
        </div>
      </div>
      
      {showConfig && (
        <WidgetConfigPanel
          onAddWidget={(widget) => {
            onAddWidget(widget);
            setShowConfig(false);
          }}
          onClose={() => setShowConfig(false)}
        />
      )}
      
      {showSaveModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="card p-6 w-96 space-y-4">
            <h3 className="text-sm font-semibold text">Save Dashboard</h3>
            <input
              type="text"
              value={dashboardName}
              onChange={(e) => setDashboardName(e.target.value)}
              placeholder="Enter dashboard name"
              className="input w-full"
              autoFocus
            />
            <div className="flex gap-2">
              <button onClick={() => setShowSaveModal(false)} className="btn btn-secondary flex-1">
                Cancel
              </button>
              <button onClick={handleSave} className="btn btn-primary flex-1">
                Save
              </button>
            </div>
          </div>
        </div>
      )}
      
      {widgets.length === 0 ? (
        <div className="card p-12 text-center">
          <div className="text-4xl mb-4">📊</div>
          <h3 className="text-sm font-semibold text mb-2">No widgets yet</h3>
          <p className="text-xs text-tertiary mb-4">Click "Add Widget" to create your first visualization</p>
          <button onClick={() => setShowConfig(true)} className="btn btn-primary">
            Add Your First Widget
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          {widgets.map((widget) => (
            <div key={widget.id} className="relative group">
              <WidgetRenderer widget={widget} />
              <button
                onClick={() => onRemoveWidget(widget.id)}
                className="absolute top-2 right-2 p-1.5 rounded-md bg-danger/10 hover:bg-danger/20 text-danger opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function EDADashboardsPage() {
  const [widgets, setWidgets] = useState<Widget[]>([]);
  
  const handleAddWidget = useCallback((widget: Widget) => {
    setWidgets(prev => [...prev, widget]);
  }, []);
  
  const handleRemoveWidget = useCallback((id: string) => {
    setWidgets(prev => prev.filter(w => w.id !== id));
  }, []);
  
  const handleSaveDashboard = useCallback((name: string) => {
    console.log("Saving dashboard:", name, widgets);
    alert(`Dashboard "${name}" saved successfully!`);
  }, [widgets]);
  
  const handleShare = useCallback(() => {
    alert("Share link copied to clipboard!");
  }, []);
  
  return (
    <section className="space-y-6">
      <PageHeader
        title="EDA Dashboards"
        subtitle="Build custom dashboards with widgets"
      />
      
      <DashboardBuilder
        widgets={widgets}
        onAddWidget={handleAddWidget}
        onRemoveWidget={handleRemoveWidget}
        onSaveDashboard={handleSaveDashboard}
        onShare={handleShare}
      />
    </section>
  );
}