import { useState, useMemo } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { datasets, workspace } from "../../lib/mocks/data";

import { Server, Activity, ArrowRight, Play, CheckCircle2, Copy, Eye, Clock, GitBranch } from "lucide-react";

const rawData = [
  { id: 1, region: 'South', dept: 'Engineering', salary: 72000 },
  { id: 2, region: 'North', dept: 'Marketing', salary: 48500 },
  { id: 3, region: 'South', dept: 'Engineering', salary: 95400 },
  { id: 4, region: 'South', dept: 'Sales', salary: null },
  { id: 5, region: 'East',  dept: 'Product', salary: 88200 },
  { id: 6, region: 'North', dept: 'Sales', salary: null }
];

const columnStats: Record<string, any> = {
  'salary': { 
    dtype: 'float64', missing: '20%', unique: 4, min: '48.5k', max: '95.4k',
    dist: [10, 45, 80, 60, 100, 30, 15, 5] 
  },
  'region': { dtype: 'object', missing: '0%', unique: 3, top: 'South (66%)', empty: '0' },
  'department': { dtype: 'object', missing: '0%', unique: 4, top: 'Engineering (33%)', empty: '0' }
};

const operationsDict = [
  { title: 'Drop Nulls', desc: 'Remove rows containing NaN/Null values.', types: ['float64', 'object'] },
  { title: 'Filter Rows', desc: 'Retains only rows matching a condition.', types: ['float64', 'object'] },
  { title: 'Fill Missing (Impute)', desc: 'Replace missing data with statistical markers.', types: ['float64'] },
  { title: 'Extract Regex', desc: 'Pull substring using pattern matching.', types: ['object'] },
  { title: 'Z-Score Normalize', desc: 'Standardize numeric distribution.', types: ['float64'] }
];

export function DataTransformPage() {
  const [selectedDatasetId, setSelectedDatasetId] = useState(workspace.activeDatasetId || (datasets && datasets[0]?.id));
  
  const [pipeline, setPipeline] = useState([{ id: 'source', title: 'Source Data', code: "pd.read_sql('SELECT * FROM employees')" }]);
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [view, setView] = useState<'select' | 'config' | 'versions' | 'run'>('select');
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  const [selectedOperation, setSelectedOperation] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Version Control & Execution State
  const [activeVersion, setActiveVersion] = useState('v2');
  const [runMode, setRunMode] = useState<'incremental' | 'full'>('incremental');
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);

  const partitionStatus = [
    { id: 'date=2026-05-01', status: 'up-to-date', icon: <CheckCircle2 size={14} className="text-success" /> },
    { id: 'date=2026-05-02', status: 'changed', icon: <span className="text-[10px] w-[14px] flex justify-center">⚠️</span> },
    { id: 'date=2026-05-03', status: 'new', icon: <span className="text-[10px] w-[14px] flex justify-center">➕</span> }
  ];

  const currentData = useMemo(() => {
    let data = [...rawData];
    for (let i = 1; i <= activeStepIndex; i++) {
      if (pipeline[i].title === 'Drop Nulls') data = data.filter(row => row.salary !== null);
      if (pipeline[i].title === 'Filter Rows') data = data.filter(row => row.region === 'South');
    }
    return data;
  }, [pipeline, activeStepIndex]);

  const activeStepCode = pipeline[activeStepIndex]?.code || '';

  const openDrawer = (v: 'select' | 'config' | 'versions' | 'run', col: string | null = null, op: string | null = null) => {
    setView(v);
    if (col) setSelectedColumn(col);
    if (op) setSelectedOperation(op);
    setDrawerOpen(true);
  };

  const closeDrawer = () => {
    setDrawerOpen(false);
    setTimeout(() => {
      setSelectedColumn(null);
      setSelectedOperation(null);
      setView('select');
      setSearchQuery('');
    }, 200);
  };

  const applyOperation = () => {
    if (!selectedOperation) return;
    const codeMap: Record<string, string> = {
      'Drop Nulls': `${selectedColumn ? `df.dropna(subset=['${selectedColumn}'])` : 'df.dropna()'}`,
      'Filter Rows': `df[df['${selectedColumn || 'region'}'] == 'South']`,
    };
    
    const newPipeline = [...pipeline.slice(0, activeStepIndex + 1), {
      id: `step-${Date.now()}`,
      title: selectedOperation,
      code: codeMap[selectedOperation] || `df['${selectedColumn || 'col'}'].transform(...)`
    }];
    setPipeline(newPipeline);
    setActiveStepIndex(newPipeline.length - 1);
    closeDrawer();
  };

  return (
    <div className="flex h-[calc(100vh-5rem)] flex-col overflow-hidden bg-[#fafafa]">
      <style dangerouslySetInnerHTML={{__html: `
        .wf-root { --geist-foreground: #000; --geist-background: #fff; --accents-1: #fafafa; --accents-2: #eaeaea; --accents-3: #999; }
        .wf-container { height: calc(100vh - 120px); border: 1px solid var(--accents-2); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; position: relative; background: var(--geist-background); }
        .wf-topbar { display: flex; align-items: center; padding: 0 16px; height: 52px; border-bottom: 1px solid var(--accents-2); background: var(--geist-background); flex-shrink: 0; }
        .wf-sidebar { flex-shrink: 0; border-right: 1px solid var(--accents-2); display: flex; flex-direction: column; background: var(--accents-1); z-index: 2; transition: width 0.3s ease, min-width 0.3s ease; width: 220px; min-width: 220px; }
        .wf-sidebar.collapsed { width: 0; min-width: 0; border-right: none; overflow: hidden; opacity: 0; pointer-events: none; }
        .wf-step-item { display: flex; align-items: flex-start; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--accents-2); cursor: pointer; transition: background 0.2s; }
        .wf-step-item:hover { background: var(--geist-background); }
        .wf-step-item.active { background: var(--geist-background); box-shadow: inset 2px 0 0 #000; }
        .wf-step-icon { width: 20px; height: 20px; border-radius: 4px; background: var(--accents-2); display: flex; align-items: center; justify-center; font-size: 10px; font-weight: bold; flex-shrink: 0; padding-left: 6.5px; padding-top: 1.5px;}
        .wf-step-item.active .wf-step-icon { background: #000; color: #fff; }
        .wf-table-container { overflow: auto; background: var(--geist-background); margin: 0; padding: 0; display: block; position: relative; z-index: 0; }
        .wf-table-container table { width: 100%; border-collapse: separate; border-spacing: 0; text-align: left; table-layout: auto; }
        .wf-table-container th { position: sticky; top: 0; background: var(--geist-background); box-shadow: 0 1px 0 var(--accents-2); z-index: 2; padding: 12px 16px; border-right: 1px solid var(--accents-2); cursor: pointer; white-space: normal; }
        .wf-table-container th:hover { background: var(--accents-1); }
        .wf-table-container td { padding: 10px 16px; font-size: 13px; border-right: 1px solid var(--accents-2); border-bottom: 1px solid var(--accents-2); white-space: normal; }
        .wf-op-card { padding: 12px; border: 1px solid var(--accents-2); border-radius: 6px; cursor: pointer; margin-bottom: 8px; }
        .wf-op-card:hover { border-color: var(--geist-foreground); }
        .wf-drawer { position: absolute; top: 52px; right: -320px; width: 320px; bottom: 0; background: var(--geist-background); border-left: 1px solid var(--accents-2); transition: right 0.3s ease; display: flex; flex-direction: column; z-index: 20; box-shadow: -4px 0 12px rgba(0,0,0,0.05); }
        .wf-drawer.open { right: 0; }
      `}} />
      
      <PageHeader 
        title="Data Transformation" 
        subtitle="Clean, shape, and transform your data visually."
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
          </div>
        }
      />

      <div className="wf-root wf-container m-4 mb-0 flex-1">
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
            <div>Rows: <strong className="text-black font-mono">{currentData.length}</strong></div>
            <div>Cols: <strong className="text-black font-mono">4</strong></div>
            <button className="text-[#666] hover:text-black border border-[#eaeaea] px-3 py-1.5 rounded font-medium ml-2 bg-white transition-colors text-xs flex items-center gap-1" onClick={() => openDrawer('versions')}>
              <GitBranch size={14} /> {activeVersion}
            </button>
            <button className="bg-black text-white px-3 py-1.5 rounded font-medium ml-2 hover:bg-[#333] transition-colors" onClick={() => setShowSaveModal(true)}>Save & Apply</button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden relative">
          {/* Sidebar */}
          <div className={`wf-sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
            <div className="text-xs font-semibold p-4">Execution Graph</div>
            <div className="flex-1 overflow-y-auto px-3">
              {pipeline.map((step, idx) => (
                <div key={idx} className={`wf-step-item shadow-sm ${idx === activeStepIndex ? 'active' : ''}`} onClick={() => setActiveStepIndex(idx)}>
                  <div className="wf-step-icon">{idx + 1}</div>
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] font-medium mb-0.5">{step.title}</div>
                    <div className="text-[11px] text-[#666] whitespace-nowrap overflow-hidden text-ellipsis font-mono">{step.code}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="p-4 border-t border-[#eaeaea]">
              <button 
                onClick={() => openDrawer('select', null)}
                className="w-full flex justify-between items-center text-[13px] border border-[#eaeaea] bg-white px-4 py-2 rounded hover:bg-[#fafafa] transition-colors shadow-sm"
              >
                <span className="font-medium">Add Step</span>
                <span className="text-[#666] text-lg leading-none">+</span>
              </button>
            </div>
          </div>

          {/* Main Table Area */}
          <div className="flex-1 flex flex-col overflow-hidden z-10 z-0 bg-white">
            <div className="flex items-center p-2 px-4 border-b border-[#eaeaea] gap-3 bg-[#fafafa]">
              <span className="font-mono text-xs font-semibold text-[#666]">fx</span>
              <input className="flex-1 border border-[#eaeaea] rounded px-3 py-2 font-mono text-xs bg-white text-[#666]" readOnly value={activeStepCode} />
            </div>
            
            <div className="wf-table-container flex-1 overflow-auto">
              <table>
                <thead>
                  <tr>
                    <th style={{width: 50}}>#</th>
                    <th onClick={() => openDrawer('select', 'region')}>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-semibold">region</span>
                        <span className="text-[10px] font-mono text-[#666] bg-[#fafafa] px-1 rounded">object</span>
                      </div>
                      <div className="flex h-1 w-full bg-[#0070f3] mb-1"></div>
                      <div className="flex justify-between text-[10px] text-[#666]"><span>100%</span></div>
                    </th>
                    <th onClick={() => openDrawer('select', 'department')}>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-semibold">department</span>
                        <span className="text-[10px] font-mono text-[#666] bg-[#fafafa] px-1 rounded">object</span>
                      </div>
                      <div className="flex h-1 w-full bg-[#0070f3] mb-1"></div>
                      <div className="flex justify-between text-[10px] text-[#666]"><span>100%</span></div>
                    </th>
                    <th onClick={() => openDrawer('select', 'salary')}>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-semibold">salary</span>
                        <div className="flex items-center gap-2">
                           <button className="text-[9px] text-[#666] hover:text-black border border-[#eaeaea] px-1.5 py-0.5 rounded bg-white transition-colors" title="View Lineage" onClick={(e) => { e.stopPropagation(); openDrawer('config', 'salary'); }}>Lineage</button>
                           <span className="text-[10px] font-mono text-[#666] bg-[#fafafa] px-1 rounded">float64</span>
                        </div>
                      </div>
                      <div className="flex h-1 w-full mb-1">
                        <div style={{width: '80%'}} className="bg-[#0070f3]"></div>
                        <div style={{width: '20%'}} className="bg-[#888888]"></div>
                      </div>
                      <div className="flex justify-between text-[10px] text-[#666]"><span>80%</span><span>20%</span></div>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {currentData.map((row, idx) => (
                    <tr key={idx} className="hover:bg-[#fafafa]">
                      <td className="font-mono text-right text-[#888]">{idx + 1}</td>
                      <td>{row.region}</td>
                      <td>{row.dept}</td>
                      <td className={`font-mono text-right ${row.salary ? '' : 'text-[#888] italic'}`}>
                        {row.salary ? row.salary.toLocaleString() : 'NaN'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right Drawer */}
          <div className={`wf-drawer ${drawerOpen ? 'open' : ''}`}>
            <div className="flex items-center justify-between p-4 border-b border-[#eaeaea]">
              <div className="font-semibold text-[13px]">
                {view === 'select' ? 'Select Operation' : view === 'config' ? 'Configure Operation' : view === 'versions' ? 'Transformation Versions' : 'Execution Mode'}
              </div>
              <button className="text-[#666] hover:text-[#000]" onClick={closeDrawer}>✕</button>
            </div>

            <div className="flex-1 overflow-y-auto bg-[#fafafa]">
              {view === 'versions' && (
                <div className="p-4">
                  <div className="space-y-3">
                    {['v3', 'v2', 'v1'].map(v => (
                      <div key={v} className={`border rounded-lg p-3 cursor-pointer transition-all ${activeVersion === v ? 'border-black bg-white shadow-sm ring-1 ring-black' : 'border-[#eaeaea] bg-white hover:border-[#999]'}`} onClick={() => setActiveVersion(v)}>
                         <div className="flex justify-between items-center mb-1">
                           <span className="font-semibold text-sm flex items-center gap-2"><GitBranch size={14}/> {v} {v === activeVersion && <span className="text-[10px] bg-black text-white px-2 py-0.5 rounded-full">ACTIVE</span>}</span>
                           <span className="text-[10px] text-[#666]">{v === 'v3' ? '2 hrs ago' : v === 'v2' ? '1 day ago' : '5 days ago'}</span>
                         </div>
                         <p className="text-[11px] text-[#666] mb-3">Commit message or auto-generated description for this transform state.</p>
                         <div className="flex gap-2">
                           <button className="text-[10px] uppercase font-bold text-[#666] hover:text-black flex items-center gap-1 border border-[#eaeaea] rounded px-2 py-1"><Eye size={12}/> View DAG</button>
                           <button className="text-[10px] uppercase font-bold text-[#666] hover:text-black flex items-center gap-1 border border-[#eaeaea] rounded px-2 py-1"><Copy size={12}/> Clone</button>
                         </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedColumn && view === 'select' && (
                <div className="p-4 border-b border-[#eaeaea] bg-white">
                  <div className="text-[11px] uppercase tracking-wider text-[#666] font-semibold mb-3">Target Column</div>
                  <div className="border border-[#eaeaea] rounded p-3 mb-4 bg-[#fafafa]">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-mono font-semibold text-sm">{selectedColumn}</span>
                      <span className="text-[10px] text-[#666]">{columnStats[selectedColumn]?.dtype}</span>
                    </div>
                    <div className="text-xs text-[#666]">Missing: {columnStats[selectedColumn]?.missing}</div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 mb-4">
                    {(() => {
                      const stats = columnStats[selectedColumn];
                      if (!stats) return null;
                      return stats.dtype === 'float64' ? (
                        <>
                          <div className="flex flex-col"><span className="text-[10px] text-[#666]">Min</span><span className="text-xs font-mono font-medium">{stats.min}</span></div>
                          <div className="flex flex-col"><span className="text-[10px] text-[#666]">Max</span><span className="text-xs font-mono font-medium">{stats.max}</span></div>
                        </>
                      ) : (
                        <>
                          <div className="flex flex-col"><span className="text-[10px] text-[#666]">Unique</span><span className="text-xs font-mono font-medium">{stats.unique}</span></div>
                          <div className="flex flex-col"><span className="text-[10px] text-[#666]">Top Mode</span><span className="text-xs font-mono font-medium">{stats.top}</span></div>
                        </>
                      );
                    })()}
                  </div>
                  {columnStats[selectedColumn]?.dtype === 'float64' && (
                    <div>
                      <div className="flex items-end gap-[2px] h-[60px] border-b border-[#eaeaea] pb-[2px]">
                        {columnStats[selectedColumn].dist.map((h: number, i: number) => (
                          <div key={i} className="flex-1 bg-[#999] hover:bg-black transition-colors rounded-t-[2px]" style={{height: `${h}%`}}></div>
                        ))}
                      </div>
                      <div className="flex justify-between text-[9px] font-mono text-[#666] mt-1">
                        <span>{columnStats[selectedColumn].min}</span><span>{columnStats[selectedColumn].max}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {view === 'select' && (
                <div className="p-4 flex flex-col gap-2">
                  <div className="pb-3 mb-1 border-b border-[#eaeaea]">
                    <input 
                      type="text" 
                      placeholder="Search operations..." 
                      className="w-full px-3 py-2 border border-[#eaeaea] rounded font-sans text-[13px] outline-none focus:border-black shadow-sm"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                    />
                  </div>
                  {operationsDict.filter(op => 
                    op.title.toLowerCase().includes(searchQuery.toLowerCase()) && 
                    (!selectedColumn || op.types.includes(columnStats[selectedColumn]?.dtype))
                  ).map((op, idx) => (
                    <div key={idx} className="wf-op-card shadow-sm bg-white" onClick={() => openDrawer('config', selectedColumn, op.title)}>
                      <div className="font-semibold text-[13px] mb-1">{op.title}</div>
                      <div className="text-xs text-[#666] leading-tight">{op.desc}</div>
                    </div>
                  ))}
                </div>
              )}

              {view === 'config' && selectedColumn === 'salary' && !selectedOperation && (
                <div className="p-5 bg-white h-full border-b border-[#eaeaea]">
                    <div className="text-[15px] font-semibold mb-4">Column Lineage</div>
                    <div className="border border-[#eaeaea] rounded p-4 bg-[#fafafa]">
                      <div className="text-[11px] uppercase tracking-wider text-[#666] font-semibold mb-2">Column</div>
                      <div className="font-mono font-bold text-sm mb-4">salary_cleaned</div>
                      
                      <div className="text-[11px] uppercase tracking-wider text-[#666] font-semibold mb-2">Derived From</div>
                      <div className="flex items-center gap-2 mb-4">
                        <ArrowRight size={14} className="text-[#999]" />
                        <span className="font-mono text-xs bg-white border border-[#eaeaea] px-2 py-1 rounded">salary_raw</span>
                      </div>

                      <div className="text-[11px] uppercase tracking-wider text-[#666] font-semibold mb-2">Transformations Applied</div>
                      <ul className="space-y-2 relative before:absolute before:left-[11px] before:top-2 before:bottom-2 before:w-[2px] before:bg-[#eaeaea]">
                        <li className="flex items-center gap-3 relative z-10">
                          <div className="w-6 h-6 rounded-full bg-black text-white flex items-center justify-center text-[10px] font-bold">1</div>
                          <span className="text-xs font-medium bg-white px-2 py-1 border border-[#eaeaea] rounded shadow-sm">Drop Nulls</span>
                        </li>
                        <li className="flex items-center gap-3 relative z-10">
                          <div className="w-6 h-6 rounded-full bg-black text-white flex items-center justify-center text-[10px] font-bold">2</div>
                          <span className="text-xs font-medium bg-white px-2 py-1 border border-[#eaeaea] rounded shadow-sm">Z-Score Normalize</span>
                        </li>
                      </ul>
                    </div>
                </div>
              )}

              {view === 'config' && selectedOperation && (
                <div className="p-5 flex flex-col gap-4 h-full bg-white">
                  <div className="text-[15px] font-semibold mb-2">{selectedOperation}</div>
                  
                  <div>
                    <label className="block text-xs font-semibold text-[#666] mb-1.5">Target Column</label>
                    <select 
                      className="w-full px-3 py-2 border border-[#eaeaea] rounded font-sans text-[13px] outline-none mb-3 bg-[#fafafa]"
                      value={selectedColumn || ''}
                      onChange={e => setSelectedColumn(e.target.value)}
                    >
                      <option value="">-- Apply to entire dataframe --</option>
                      <option value="region">region (object)</option>
                      <option value="department">department (object)</option>
                      <option value="salary">salary (float64)</option>
                    </select>
                  </div>
                  
                  <div className="flex-1"></div>
                  
                  <div className="flex gap-3 pt-4 border-t border-[#eaeaea]">
                    <button className="flex-1 py-2 text-sm border border-[#eaeaea] rounded font-medium hover:bg-[#fafafa]" onClick={() => setView('select')}>Back</button>
                    <button className="flex-1 py-2 text-sm bg-black text-white rounded font-medium hover:bg-[#333]" onClick={applyOperation}>Apply</button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Save & Execution Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in">
          <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden border border-border animate-in zoom-in-95 duration-200">
            {!showConfirmModal ? (
              <>
                <div className="p-6 border-b border-border">
                  <h3 className="text-xl font-bold tracking-tight">Save Transformation</h3>
                  <p className="text-sm text-text-tertiary mt-1">Choose how to save and execute this transformation pipeline.</p>
                </div>
                <div className="p-6 space-y-6 bg-surface-2/30">
                  <div className="space-y-3">
                    <h4 className="text-[11px] font-black uppercase tracking-widest text-text-tertiary">Save Strategy</h4>
                    <label className="flex items-center gap-3 p-3 border border-border-subtle rounded-lg cursor-pointer hover:bg-white bg-white shadow-sm ring-1 ring-primary/20">
                      <input type="radio" name="save_strat" defaultChecked className="accent-primary size-4" />
                      <div>
                        <span className="text-sm font-bold block">Save as New Version (Default)</span>
                        <span className="text-[10px] text-text-tertiary">Safe: creates {activeVersion.replace(/v(\d+)/, (m,p1)=>`v${parseInt(p1)+1}`)} and leaves {activeVersion} intact.</span>
                      </div>
                    </label>
                    <label className="flex items-center gap-3 p-3 border border-border-subtle rounded-lg cursor-pointer hover:bg-white opacity-70">
                      <input type="radio" name="save_strat" className="accent-primary size-4" />
                      <div>
                        <span className="text-sm font-bold block">Overwrite Current Version</span>
                        <span className="text-[10px] text-text-tertiary">Dangerous: permanently replaces {activeVersion}.</span>
                      </div>
                    </label>
                  </div>

                  <div className="space-y-3 pt-6 border-t border-border-subtle">
                    <h4 className="text-[11px] font-black uppercase tracking-widest text-text-tertiary">Execution Mode</h4>
                    <div className="flex gap-4">
                       <label className={`flex-1 flex flex-col items-center gap-2 p-4 border rounded-xl cursor-pointer text-center transition-all ${runMode === 'incremental' ? 'border-primary bg-primary/5 ring-1 ring-primary text-primary' : 'border-border-subtle bg-white hover:border-border text-text-tertiary hover:text-text'}`}>
                         <input type="radio" name="run_mode" className="sr-only" checked={runMode==='incremental'} onChange={() => setRunMode('incremental')} />
                         <Clock size={20} />
                         <div>
                          <span className="text-sm font-bold block text-text">Incremental</span>
                          <span className="text-[10px] uppercase font-bold tracking-widest">Recommended</span>
                         </div>
                       </label>
                       <label className={`flex-1 flex flex-col items-center gap-2 p-4 border rounded-xl cursor-pointer text-center transition-all ${runMode === 'full' ? 'border-primary bg-primary/5 ring-1 ring-primary text-primary' : 'border-border-subtle bg-white hover:border-border text-text-tertiary hover:text-text'}`}>
                         <input type="radio" name="run_mode" className="sr-only" checked={runMode==='full'} onChange={() => setRunMode('full')} />
                         <Server size={20} />
                         <div>
                          <span className="text-sm font-bold block text-text">Full Recompute</span>
                          <span className="text-[10px] uppercase font-bold tracking-widest">Resource Intensive</span>
                         </div>
                       </label>
                    </div>
                  </div>
                </div>
                <div className="p-6 border-t border-border flex justify-end gap-3 bg-white">
                  <button onClick={() => setShowSaveModal(false)} className="btn btn-secondary px-6">Cancel</button>
                  <button onClick={() => setShowConfirmModal(true)} className="btn btn-primary px-8">Continue to Execute</button>
                </div>
              </>
            ) : (
              <>
               <div className="p-6 border-b border-border">
                  <h3 className="text-xl font-bold tracking-tight">Execution Preview</h3>
                  <p className="text-sm text-text-tertiary mt-1">Review the partition impact before executing <span className="font-mono text-xs">{runMode.toUpperCase()}</span> run.</p>
                </div>
                <div className="p-6 bg-surface-2/30 space-y-6">
                  
                  <div className="bg-white rounded-xl p-4 border border-border-subtle">
                    <p className="text-[11px] font-black uppercase tracking-widest text-text-tertiary mb-3 flex items-center justify-between">
                      Partitions Detected
                      <span className="text-[9px] bg-surface-2 px-2 py-1 rounded">Target: sales_data</span>
                    </p>
                    <ul className="space-y-2 text-sm font-mono">
                       {partitionStatus.map((p) => (
                         <li key={p.id} className="flex items-center gap-2">
                            {p.icon} {p.id} <span className="text-text-tertiary text-xs ml-auto">({p.status})</span>
                         </li>
                       ))}
                    </ul>
                  </div>

                  <div className="bg-primary/5 border border-primary/20 rounded-xl p-5 text-sm font-medium leading-relaxed">
                    You are about to process:
                    <ul className="list-disc pl-5 mt-2 space-y-1 mb-4 font-normal text-text-secondary">
                      <li><strong className="text-text font-bold">1</strong> new partition</li>
                      <li><strong className="text-text font-bold">1</strong> modified partition</li>
                      {runMode === 'incremental' ? (
                        <li><strong className="text-text font-bold">10</strong> unchanged partitions <span className="text-text-tertiary">(will be skipped)</span></li>
                      ) : (
                        <li className="text-warning-strong"><strong className="font-bold">10</strong> unchanged partitions <span className="underline decoration-warning">will be recomputed</span></li>
                      )}
                    </ul>
                    <div className="flex items-center justify-between pt-3 border-t border-primary/10">
                      <span className="text-xs uppercase tracking-widest font-black text-text-tertiary">Est. Compute Cost</span>
                      <span className={`text-sm font-black tracking-widest uppercase ${runMode === 'incremental' ? 'text-success' : 'text-warning-strong'}`}>
                        {runMode === 'incremental' ? 'LOW' : 'HIGH'}
                      </span>
                    </div>
                  </div>

                </div>
                <div className="p-6 border-t border-border flex justify-end gap-3 bg-white">
                  <button onClick={() => setShowConfirmModal(false)} className="btn btn-secondary px-6">Back</button>
                  <button onClick={() => { setShowSaveModal(false); setShowConfirmModal(false); }} className="btn btn-primary px-8 flex items-center gap-2"><Play size={16} fill="currentColor" /> Execute Pipeline</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
