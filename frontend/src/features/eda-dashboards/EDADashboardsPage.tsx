import { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { PageHeader } from "../../components/shared/PageHeader";
import { insights, charts } from "../../lib/mocks/data";

interface ChartDataPoint {
  label: string;
  value: number;
  value2?: number;
}

const mockChartData: Record<string, ChartDataPoint[]> = {
  line: [
    { label: "Jan", value: 42000 },
    { label: "Feb", value: 48000 },
    { label: "Mar", value: 51000 },
    { label: "Apr", value: 47000 },
    { label: "May", value: 54000 },
    { label: "Jun", value: 62000 },
    { label: "Jul", value: 68000 },
    { label: "Aug", value: 72000 },
    { label: "Sep", value: 69000 },
    { label: "Oct", value: 75000 },
    { label: "Nov", value: 82000 },
    { label: "Dec", value: 91000 }
  ],
  bar: [
    { label: "18-25", value: 42 },
    { label: "26-35", value: 28 },
    { label: "36-45", value: 18 },
    { label: "46-55", value: 12 },
    { label: "55+", value: 8 }
  ],
  scatter: [
    { label: "A", value: 1200, value2: 2 },
    { label: "B", value: 3400, value2: 5 },
    { label: "C", value: 2800, value2: 3 },
    { label: "D", value: 5600, value2: 8 },
    { label: "E", value: 1900, value2: 1 },
    { label: "F", value: 4200, value2: 6 },
    { label: "G", value: 3800, value2: 4 },
    { label: "H", value: 6100, value2: 9 },
    { label: "I", value: 2200, value2: 2 },
    { label: "J", value: 4700, value2: 7 }
  ]
};

function MiniLineChart({ data, color = "#0ea5e9" }: { data: ChartDataPoint[]; color?: string }) {
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

function MiniBarChart({ data }: { data: ChartDataPoint[] }) {
  const max = Math.max(...data.map(d => d.value));
  const width = 100;
  const height = 40;
  const barWidth = (width / data.length) - 4;
  
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="none">
      {data.map((d, i) => {
        const barHeight = (d.value / max) * (height - 8);
        const x = (i * (width / data.length)) + 2;
        const y = height - 4 - barHeight;
        const color = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"][i % 5];
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barWidth}
            height={barHeight}
            rx={2}
            fill={color}
            className="opacity-90 hover:opacity-100 transition-opacity"
          />
        );
      })}
    </svg>
  );
}

function MiniScatterChart({ data }: { data: ChartDataPoint[] }) {
  const maxX = Math.max(...data.map(d => d.value));
  const maxY = Math.max(...data.map(d => d.value2 || 0));
  const width = 100;
  const height = 40;
  const padding = 6;
  
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
      <defs>
        <radialGradient id="scatterGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.2" />
        </radialGradient>
      </defs>
      {data.map((d, i) => {
        const x = padding + ((d.value / maxX) * (width - padding * 2));
        const y = height - padding - ((d.value2 || 0) / maxY) * (height - padding * 2);
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r={3}
            fill="url(#scatterGrad)"
            stroke="#0ea5e9"
            strokeWidth={0.5}
            className="hover:r-4 transition-all"
          />
        );
      })}
    </svg>
  );
}

function TrendIndicator({ value }: { value: number }) {
  const isPositive = value > 0;
  const isNeutral = value === 0;
  return (
    <span className={`inline-flex items-center gap-0.5 text-xs font-medium ${isPositive ? 'text-success' : isNeutral ? 'text-tertiary' : 'text-danger'}`}>
      <svg className={`w-3 h-3 ${isPositive ? '' : 'rotate-180'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
      </svg>
      {Math.abs(value)}%
    </span>
  );
}

function InsightCard({ insight }: { insight: typeof insights[0] }) {
  const [expanded, setExpanded] = useState(false);
  
  const severityConfig = {
    warning: {
      icon: (
        <svg className="w-4 h-4 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      ),
      bg: "bg-warning/10",
      border: "border-warning/20",
      label: "Warning"
    },
    info: {
      icon: (
        <svg className="w-4 h-4 text-info" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      bg: "bg-info/10",
      border: "border-info/20",
      label: "Insight"
    },
    success: {
      icon: (
        <svg className="w-4 h-4 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      bg: "bg-success/10",
      border: "border-success/20",
      label: "Success"
    }
  };
  
  const config = severityConfig[insight.severity];
  
  return (
    <div 
      className={`group p-4 rounded-lg border ${config.bg} ${config.border} cursor-pointer transition-all hover:shadow-md`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 p-1.5 rounded-md ${config.bg}`}>
          {config.icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-semibold text truncate">{insight.title}</p>
            <span className={`shrink-0 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider rounded-full ${config.bg} ${config.border} border`}>
              {config.label}
            </span>
          </div>
          <p className="mt-1.5 text-xs text-tertiary leading-relaxed">{insight.detail}</p>
          
          {expanded && (
            <div className="mt-3 pt-3 border-t border-subtle animate-in fade-in slide-in-from-top-1 duration-200">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2 rounded surface">
                  <span className="text-tertiary block mb-0.5">Confidence</span>
                  <span className="font-semibold text">94.2%</span>
                </div>
                <div className="p-2 rounded surface">
                  <span className="text-tertiary block mb-0.5">Impact</span>
                  <span className="font-semibold text">High</span>
                </div>
              </div>
              <button className="btn btn-ghost mt-2 w-full py-1.5 text-xs font-medium text-accent">
                View Details →
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ChartCard({ chart, index }: { chart: typeof charts[0]; index: number }) {
  const [showControls, setShowControls] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [timeRange, setTimeRange] = useState("12M");
  
  const chartData = mockChartData[chart.type] || mockChartData.line;
  
  const chartColors = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b"];
  
  return (
    <div 
      className={`relative group rounded-xl border surface shadow-sm transition-all hover:shadow-lg ${isFullscreen ? 'fixed inset-4 z-50 shadow-2xl' : ''}`}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      <div className="absolute top-3 right-3 z-10 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button 
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-1.5 rounded-md surface-2 hover:surface border shadow-sm transition-colors"
          title="Fullscreen"
        >
          <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
        <button className="p-1.5 rounded-md surface-2 hover:surface border shadow-sm transition-colors" title="Export">
          <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </button>
        <button className="p-1.5 rounded-md surface-2 hover:surface border shadow-sm transition-colors" title="Share">
          <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
          </svg>
        </button>
      </div>
      
      <div className="p-4 border-b border-subtle">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text">{chart.title}</h3>
            <span className="rounded-md border surface-2 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-tertiary">
              {chart.type}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {chart.type === "line" && (
              <div className="flex surface-2 rounded-md p-0.5">
                {["1M", "3M", "6M", "12M"].map((range) => (
                  <button
                    key={range}
                    onClick={() => setTimeRange(range)}
                    className={`px-2 py-0.5 text-[10px] font-medium rounded transition-colors ${timeRange === range ? 'surface shadow-sm text' : 'text-tertiary hover:text'}`}
                  >
                    {range}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className={`p-4 ${isFullscreen ? 'h-[calc(100vh-120px)]' : 'h-48'}`}>
        {chart.type === "line" && <MiniLineChart data={chartData} color={chartColors[index % chartColors.length]} />}
        {chart.type === "bar" && <MiniBarChart data={chartData} />}
        {chart.type === "scatter" && <MiniScatterChart data={chartData} />}
      </div>
      
      <div className="px-4 pb-4 flex items-center justify-between">
        <p className="text-xs text-tertiary leading-relaxed">{chart.note}</p>
        <div className="flex items-center gap-2">
          <span className="text-xs text-tertiary">vs prev period</span>
          <TrendIndicator value={[12, -8, 23][index % 3]} />
        </div>
      </div>
      
      {showControls && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1 px-2 py-1.5 surface-2/90 backdrop-blur-sm rounded-full border shadow-lg animate-in fade-in zoom-in-95 duration-200">
          <button className="p-1 hover:surface rounded transition-colors" title="Zoom In">
            <svg className="w-3.5 h-3.5 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
          </button>
          <button className="p-1 hover:surface rounded transition-colors" title="Zoom Out">
            <svg className="w-3.5 h-3.5 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
            </svg>
          </button>
          <div className="w-px h-4 border" />
          <button className="p-1 hover:surface rounded transition-colors" title="Reset View">
            <svg className="w-3.5 h-3.5 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}

function DashboardStats() {
  const stats = [
    { label: "Total Revenue", value: "$847K", change: "+12.4%", trend: "up" },
    { label: "Active Users", value: "12,847", change: "+8.2%", trend: "up" },
    { label: "Churn Rate", value: "3.2%", change: "-0.8%", trend: "down" },
    { label: "Avg. Session", value: "4m 32s", change: "+5.1%", trend: "up" }
  ];
  
  return (
    <div className="grid grid-cols-4 gap-4">
      {stats.map((stat, i) => (
        <div key={i} className="card p-4">
          <p className="text-xs font-medium text-tertiary uppercase tracking-wider">{stat.label}</p>
          <div className="mt-2 flex items-end justify-between">
            <span className="text-2xl font-bold text">{stat.value}</span>
            <span className={`text-xs font-medium ${stat.trend === 'up' ? 'text-success' : stat.change.startsWith('-') ? 'text-success' : 'text-danger'}`}>
              {stat.change}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function QuickActions() {
  const actions = [
    { icon: "📊", label: "New Chart", color: "btn-ghost" },
    { icon: "📈", label: "Run Analysis", color: "btn-ghost" },
    { icon: "🔍", label: "Deep Dive", color: "btn-ghost" },
    { icon: "📤", label: "Export Report", color: "btn-ghost" }
  ];
  
  return (
    <div className="flex items-center gap-2">
      {actions.map((action, i) => (
        <button
          key={i}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-all hover:shadow-md ${action.color}`}
        >
          <span>{action.icon}</span>
          <span className="text">{action.label}</span>
        </button>
      ))}
    </div>
  );
}

function TimeRangeSelector() {
  const ranges = ["Today", "7D", "30D", "90D", "YTD", "All"];
  const [selected, setSelected] = useState("30D");
  
  return (
    <div className="flex items-center gap-1 p-1 surface-2 rounded-lg border">
      {ranges.map((range) => (
        <button
          key={range}
          onClick={() => setSelected(range)}
          className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${selected === range ? 'surface shadow-sm text' : 'text-tertiary hover:text'}`}
        >
          {range}
        </button>
      ))}
    </div>
  );
}

function RefreshButton() {
  const [refreshing, setRefreshing] = useState(false);
  
  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1500);
  }, []);
  
  return (
    <button
      onClick={handleRefresh}
      className="btn border flex items-center gap-2 px-3 py-2"
    >
      <svg className={`w-4 h-4 text-tertiary ${refreshing ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
      <span className="text-xs font-medium text">{refreshing ? 'Refreshing...' : 'Refresh'}</span>
    </button>
  );
}

export function EDADashboardsPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "charts" | "insights">("overview");
  
  const tabs = [
    { id: "overview", label: "Overview", icon: "📊" },
    { id: "charts", label: "Charts", icon: "📈" },
    { id: "insights", label: "Insights", icon: "💡" }
  ] as const;
  
  return (
    <section className="space-y-6">
      <PageHeader
        title="Analytics Dashboard"
        subtitle="Real-time metrics, automated insights, and shareable reports"
        actions={
          <div className="flex items-center gap-3">
            <TimeRangeSelector />
            <RefreshButton />
            <button className="btn btn-secondary flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              Invite
            </button>
            <button className="btn btn-primary flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Chart
            </button>
          </div>
        }
      />
      
      <div className="flex items-center gap-4 border-b border-subtle">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`relative px-4 py-3 text-sm font-medium transition-colors flex items-center gap-2 ${activeTab === tab.id ? 'text-accent' : 'text-tertiary hover:text'}`}
          >
            <span>{tab.icon}</span>
            {tab.label}
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent rounded-full" />
            )}
          </button>
        ))}
      </div>
      
      {activeTab === "overview" && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <DashboardStats />
          
          <div className="grid gap-6 lg:grid-cols-3">
            <article className="card flex flex-col lg:col-span-1">
              <div className="border-b px-5 py-4 flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text">Automated Insights</h2>
                  <p className="mt-0.5 text-[11px] uppercase tracking-wider text-tertiary">AI-powered analysis</p>
                </div>
                <span className="px-2 py-1 text-[10px] font-medium bg-accent/10 text-accent rounded-full">
                  {insights.length} new
                </span>
              </div>
              <div className="flex-1 divide-y p-3 space-y-2">
                {insights.map((insight) => (
                  <InsightCard key={insight.id} insight={insight} />
                ))}
              </div>
              <div className="p-3 border-t">
                <button className="w-full py-2 text-xs font-medium text-tertiary hover:text-accent transition-colors">
                  View all insights →
                </button>
              </div>
            </article>
            
            <div className="flex flex-col gap-4 lg:col-span-2">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text">Pinned Charts</h2>
                <div className="flex items-center gap-2">
                  <button className="p-1.5 rounded border surface hover:surface-2 transition-colors">
                    <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                    </svg>
                  </button>
                  <button className="p-1.5 rounded border surface hover:surface-2 transition-colors">
                    <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                    </svg>
                  </button>
                </div>
              </div>
              
              <section className="grid gap-4 sm:grid-cols-2">
                {charts.map((chart, i) => (
                  <ChartCard key={chart.id} chart={chart} index={i} />
                ))}
              </section>
              
              <article className="card border-accent/20 bg-gradient-to-r from-accent/5 to-transparent p-5 flex items-center justify-between">
                <div className="flex items-start gap-4">
                  <div className="p-2 rounded-lg bg-accent/10">
                    <svg className="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text">Need deeper analysis?</h3>
                    <p className="mt-1 text-xs text-tertiary max-w-md">Use the AI Query Studio to generate custom charts, run SQL queries, and get natural language insights from your data.</p>
                  </div>
                </div>
                <Link to="/query-studio" className="btn btn-primary shrink-0">
                  Open Query Studio →
                </Link>
              </article>
            </div>
          </div>
        </div>
      )}
      
      {activeTab === "charts" && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="flex items-center justify-between">
            <QuickActions />
            <div className="flex items-center gap-2">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search charts..."
                  className="input w-64 pl-9 pr-4 py-2"
                />
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <select className="input px-3 py-2 text-xs">
                <option>All Types</option>
                <option>Line</option>
                <option>Bar</option>
                <option>Scatter</option>
              </select>
            </div>
          </div>
          
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {charts.map((chart, i) => (
              <ChartCard key={chart.id} chart={chart} index={i} />
            ))}
            {[...charts, ...charts].slice(0, 3).map((chart, i) => (
              <ChartCard key={`extra-${i}`} chart={{ ...chart, id: `extra-${i}` }} index={i} />
            ))}
          </section>
        </div>
      )}
      
      {activeTab === "insights" && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-tertiary">Filter by:</span>
                <select className="input px-3 py-1.5 text-xs">
                  <option>All Severity</option>
                  <option>Warning</option>
                  <option>Info</option>
                  <option>Success</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-tertiary">Sort by:</span>
                <select className="input px-3 py-1.5 text-xs">
                  <option>Newest First</option>
                  <option>Highest Impact</option>
                  <option>Oldest First</option>
                </select>
              </div>
            </div>
            <button className="btn btn-ghost flex items-center gap-2 px-3 py-2 text-xs font-medium text-accent border border-accent/30">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
              </svg>
              More Filters
            </button>
          </div>
          
          <div className="grid gap-4 sm:grid-cols-2">
            {[...insights, ...insights].map((insight, i) => (
              <InsightCard key={`insight-${i}`} insight={insight} />
            ))}
          </div>
          
          <div className="flex items-center justify-center gap-2 py-8">
            <button className="btn btn-secondary px-4 py-2 text-xs">Previous</button>
            <span className="px-4 py-2 text-xs text-tertiary">Page 1 of 3</span>
            <button className="btn btn-secondary px-4 py-2 text-xs">Next</button>
          </div>
        </div>
      )}
    </section>
  );
}