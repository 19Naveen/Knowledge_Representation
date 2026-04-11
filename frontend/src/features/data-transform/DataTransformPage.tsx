import { useState, useMemo } from "react";
import { PageHeader } from "../../components/shared/PageHeader";

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

  const openDrawer = (targetView: 'select' | 'config', col: string | null = null, op: string | null = null) => {
    setDrawerOpen(true);
    setView(targetView);
    if (col !== null) setSelectedColumn(col);
    if (op !== null) setSelectedOperation(op);
  };

  const closeDrawer = () => {
    setDrawerOpen(false);
    setTimeout(() => {
      setView('select');
      setSelectedColumn(null);
      setSelectedOperation(null);
      setSearchQuery('');
    }, 300);
  };

  const commitTransformation = () => {
    const col = selectedColumn || 'region'; 
    const op = selectedOperation;
    if(!op) return;
    
    let code = '';
    if (op === 'Drop Nulls') code = `df.dropna(subset=['${col}'])`;
    else if (op === 'Filter Rows') code = `df[df['${col}'] == 'South']`;
    else if (op === 'Fill Missing (Impute)') code = `df['${col}'].fillna(0)`;
    else code = `df['${col}'].apply(${op.toLowerCase().replace(' ', '_')})`;

    const newPipeline = pipeline.slice(0, activeStepIndex + 1);
    newPipeline.push({ id: Date.now().toString(), title: op, code: code });
    setPipeline(newPipeline);
    setActiveStepIndex(newPipeline.length - 1);
    closeDrawer();
  };

  const activeStepCode = pipeline[activeStepIndex]?.code || '';

  const filteredOps = operationsDict.filter(op => {
    if (selectedColumn) {
      const targetType = columnStats[selectedColumn].dtype;
      if (!op.types.includes(targetType)) return false;
    }
    if (searchQuery) {
      return op.title.toLowerCase().includes(searchQuery.toLowerCase()) || op.desc.toLowerCase().includes(searchQuery.toLowerCase());
    }
    return true;
  });

  const stats = selectedColumn ? columnStats[selectedColumn] : null;

  return (
    <div className="flex h-full flex-col p-5 animate-in w-full overflow-hidden">
      <style dangerouslySetInnerHTML={{__html: `
        .wf-root {
          --geist-background: #ffffff;
          --geist-foreground: #000000;
          --accents-1: #fafafa;
          --accents-2: #eaeaea;
          --accents-3: #999999;
          --accents-5: #666666;
          --geist-success: #0070f3;
          --geist-error: #e00;
        }
        .wf-container { height: calc(100vh - 120px); border: 1px solid var(--accents-2); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; position: relative; background: var(--geist-background); }
        .wf-topbar { display: flex; align-items: center; padding: 0 16px; height: 52px; border-bottom: 1px solid var(--accents-2); background: var(--geist-background); flex-shrink: 0; }
        .wf-sidebar { width: 210px; flex-shrink: 0; border-right: 1px solid var(--accents-2); display: flex; flex-direction: column; background: var(--accents-1); }
        .wf-step-item { padding: 12px; border-radius: 6px; border: 1px solid transparent; cursor: pointer; display: flex; align-items: flex-start; gap: 10px; background: var(--geist-background); margin-bottom: 6px; }
        .wf-step-item.active { border-color: var(--geist-foreground); box-shadow: inset 0 0 0 1px var(--geist-foreground); }
        .wf-step-icon { width: 20px; height: 20px; border-radius: 4px; background: var(--accents-2); display: flex; align-items: center; justify-content: center; font-size: 10px; font-family: monospace; flex-shrink: 0; }
        .wf-step-item.active .wf-step-icon { background: var(--geist-foreground); color: var(--geist-background); }
        .wf-drawer { position: absolute; top: 0; right: 0; width: 360px; height: 100%; background: var(--geist-background); border-left: 1px solid var(--accents-2); box-shadow: -10px 0 30px rgba(0,0,0,0.05); transform: translateX(100%); transition: transform 0.3s; z-index: 40; display: flex; flex-direction: column; }
        .wf-drawer.open { transform: translateX(0); }
        .wf-table-container table { width: 100%; table-layout: auto; border-collapse: collapse; text-align: left; }
        .wf-table-container th { position: sticky; top: 0; background: var(--geist-background); box-shadow: 0 1px 0 var(--accents-2); z-index: 2; padding: 12px 16px; border-right: 1px solid var(--accents-2); cursor: pointer; white-space: normal; }
        .wf-table-container th:hover { background: var(--accents-1); }
        .wf-table-container td { padding: 10px 16px; font-size: 13px; border-right: 1px solid var(--accents-2); border-bottom: 1px solid var(--accents-2); white-space: normal; }
        .wf-op-card { padding: 12px; border: 1px solid var(--accents-2); border-radius: 6px; cursor: pointer; margin-bottom: 8px; }
        .wf-op-card:hover { border-color: var(--geist-foreground); }
      `}} />
      
      <PageHeader 
        title="Data Transformation" 
        subtitle="Clean, shape, and transform your data visually."
      />

      

      <div className="wf-root wf-container mt-4">
        <div className="wf-topbar">
          <div className="font-semibold text-sm border-r border-[#eaeaea] pr-4 mr-4">DataForge</div>
          <div className="ml-auto flex gap-3 items-center text-xs text-[#666]">
            <div>Rows: <strong className="text-black font-mono">{currentData.length}</strong></div>
            <div>Cols: <strong className="text-black font-mono">4</strong></div>
            <button className="bg-black text-white px-3 py-1.5 rounded font-medium ml-2">Apply Transformation</button>
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
                className="w-full flex justify-between items-center text-[13px] border border-[#eaeaea] bg-white px-4 py-2 rounded hover:bg-[#fafafa]"
              >
                <span className="font-medium">Add Step</span>
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
            <div className="p-4 border-b border-[#eaeaea] flex justify-between items-center bg-white z-10">
              <div className="text-sm font-semibold flex items-center gap-2">
                {view === 'config' && (
                  <button className="text-[#666] hover:text-black" onClick={() => setView('select')}>←</button>
                )}
                {view === 'config' ? `Configure: ${selectedOperation}` : selectedColumn ? `Transform: ${selectedColumn}` : 'Add Transformation'}
              </div>
              <button className="text-xl text-[#666] hover:text-black leading-none" onClick={closeDrawer}>×</button>
            </div>

            <div className="flex-1 overflow-y-auto flex flex-col">
              {selectedColumn && stats && (
                <div className="bg-[#fafafa] border-b border-[#eaeaea] p-4">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-[#666] mb-3">
                    Context: <span className="text-black font-mono normal-case">{selectedColumn}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div className="flex flex-col"><span className="text-[10px] text-[#666]">Dtype</span><span className="text-xs font-mono font-medium">{stats.dtype}</span></div>
                    <div className="flex flex-col"><span className="text-[10px] text-[#666]">Missing</span><span className="text-xs font-mono font-medium">{stats.missing}</span></div>
                    {stats.dtype === 'float64' ? (
                      <>
                        <div className="flex flex-col"><span className="text-[10px] text-[#666]">Min</span><span className="text-xs font-mono font-medium">{stats.min}</span></div>
                        <div className="flex flex-col"><span className="text-[10px] text-[#666]">Max</span><span className="text-xs font-mono font-medium">{stats.max}</span></div>
                      </>
                    ) : (
                      <>
                        <div className="flex flex-col"><span className="text-[10px] text-[#666]">Unique</span><span className="text-xs font-mono font-medium">{stats.unique}</span></div>
                        <div className="flex flex-col"><span className="text-[10px] text-[#666]">Top Mode</span><span className="text-xs font-mono font-medium">{stats.top}</span></div>
                      </>
                    )}
                  </div>
                  {stats.dtype === 'float64' && (
                    <div>
                      <div className="flex items-end gap-[2px] h-[60px] border-b border-[#eaeaea] pb-[2px]">
                        {stats.dist.map((h: number, i: number) => (
                          <div key={i} className="flex-1 bg-[#999] hover:bg-black transition-colors rounded-t-[2px]" style={{height: `${h}%`}}></div>
                        ))}
                      </div>
                      <div className="flex justify-between text-[9px] font-mono text-[#666] mt-1">
                        <span>{stats.min}</span><span>{stats.max}</span>
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
                      className="w-full px-3 py-2 border border-[#eaeaea] rounded font-sans text-[13px] outline-none focus:border-black"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                    />
                  </div>
                  <div>
                    {filteredOps.map((op, i) => (
                      <div key={i} className="wf-op-card bg-white" onClick={() => openDrawer('config', selectedColumn, op.title)}>
                        <div className="text-[13px] font-semibold text-black mb-1">{op.title}</div>
                        <div className="text-[11px] text-[#666] leading-snug">{op.desc}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {view === 'config' && (
                <div className="p-4 flex flex-col gap-4">
                  {!selectedColumn && (
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-medium text-black">Target Column</label>
                      <select 
                        className="px-3 py-2 border border-[#eaeaea] rounded text-[13px] outline-none"
                        value={selectedColumn || ''}
                        onChange={e => setSelectedColumn(e.target.value)}
                      >
                        <option value="">-- Select Column --</option>
                        <option value="region">region</option>
                        <option value="department">department</option>
                        <option value="salary">salary</option>
                      </select>
                    </div>
                  )}
                  
                  {selectedOperation === 'Filter Rows' ? (
                    <>
                      <div className="flex flex-col gap-1.5 mt-2">
                        <label className="text-xs font-medium text-black">Condition</label>
                        <select className="px-3 py-2 border border-[#eaeaea] rounded text-[13px] outline-none">
                          <option>Equals (==)</option>
                          <option>Contains</option>
                        </select>
                      </div>
                      <div className="flex flex-col gap-1.5 mt-2">
                        <label className="text-xs font-medium text-black">Value</label>
                        <input type="text" className="px-3 py-2 border border-[#eaeaea] rounded text-[13px] outline-none" defaultValue="South" />
                      </div>
                    </>
                  ) : (
                    <div className="mt-2 text-[11px] text-[#666]">
                      Operation utilizes default parameters for structural integrity.
                    </div>
                  )}
                </div>
              )}
            </div>

            {view === 'config' && (
              <div className="p-4 border-t border-[#eaeaea] bg-white flex gap-2">
                <button className="flex-1 py-2 text-[13px] font-medium border border-[#eaeaea] rounded bg-white hover:bg-[#fafafa]" onClick={closeDrawer}>Cancel</button>
                <button className="flex-1 py-2 text-[13px] font-medium border border-black bg-black text-white rounded hover:bg-[#333]" onClick={commitTransformation}>Apply Step</button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
