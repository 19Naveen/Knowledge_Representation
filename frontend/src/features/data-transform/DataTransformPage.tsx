import { useState, useMemo } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { datasets, workspace } from "../../lib/mocks/data";

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
  const [view, setView] = useState<'select' | 'config'>('select');
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  const [selectedOperation, setSelectedOperation] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const currentData = useMemo(() => {
    let data = [...rawData];
    for (let i = 1; i <= activeStepIndex; i++) {
      if (pipeline[i].title === 'Drop Nulls') data = data.filter(row => row.salary !== null);
      if (pipeline[i].title === 'Filter Rows') data = data.filter(row => row.region === 'South');
    }
    return data;
  }, [pipeline, activeStepIndex]);

  const activeStepCode = pipeline[activeStepIndex]?.code || '';

  const openDrawer = (v: 'select' | 'config', col: string | null = null, op: string | null = null) => {
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
        .wf-sidebar { width: 220px; flex-shrink: 0; border-right: 1px solid var(--accents-2); display: flex; flex-direction: column; background: var(--accents-1); z-index: 2; }
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
          <div className="font-semibold text-sm border-r border-[#eaeaea] pr-4 mr-4">DataForge</div>
          <div className="ml-auto flex gap-3 items-center text-xs text-[#666]">
            <div>Rows: <strong className="text-black font-mono">{currentData.length}</strong></div>
            <div>Cols: <strong className="text-black font-mono">4</strong></div>
            <button className="bg-black text-white px-3 py-1.5 rounded font-medium ml-2 hover:bg-[#333] transition-colors">Apply Transformation</button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden relative">
          {/* Sidebar */}
          <div className="wf-sidebar">
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
                        <span className="text-[10px] font-mono text-[#666] bg-[#fafafa] px-1 rounded">float64</span>
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
                {view === 'select' ? 'Select Operation' : 'Configure Operation'}
              </div>
              <button className="text-[#666] hover:text-[#000]" onClick={closeDrawer}>✕</button>
            </div>

            <div className="flex-1 overflow-y-auto bg-[#fafafa]">
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

              {view === 'config' && (
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
    </div>
  );
}
