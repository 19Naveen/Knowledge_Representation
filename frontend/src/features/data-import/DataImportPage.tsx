import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { Database, FileSpreadsheet, Activity, Clock, Upload, Plus, Table, CheckCircle2, ArrowRight, Server, Key, User } from "lucide-react";
import { cn } from "../../lib/cn";

type Step = 'list' | 'db_creds' | 'preview' | 'success';

export function DataImportPage() {
  const [activeTab, setActiveTab] = useState<'scheduled' | 'adhoc'>('scheduled');
  const [step, setStep] = useState<Step>('list');

  // Config state for preview
  const [sourceName, setSourceName] = useState<string>('');
  const [targetTable, setTargetTable] = useState<string>('new_table');
  const [newTableName, setNewTableName] = useState<string>('');
  const [scheduleFreq, setScheduleFreq] = useState<string>('daily');

  const handleConnectDb = (e: React.FormEvent) => {
    e.preventDefault();
    setSourceName('PostgreSQL - public.users');
    setNewTableName('sync_users');
    setTargetTable('new_table');
    setStep('preview');
  };

  const handleFileUpload = () => {
    setSourceName('uploaded_data.csv');
    setNewTableName('adhoc_import_data');
    setTargetTable('new_table');
    setStep('preview');
  };

  const handleImport = () => {
    setStep('success');
    setTimeout(() => {
      setStep('list');
    }, 3000);
  };

  const switchTab = (tab: 'scheduled' | 'adhoc') => {
    setActiveTab(tab);
    setStep('list');
  };

  return (
    <div className="flex h-full flex-col animate-in fade-in duration-500">
      <PageHeader
        title="Data Ingestion"
        subtitle="Orchestrate automated database synchronization or execute rapid ad-hoc object uploads."
      />

      <div className="flex-1 p-8 overflow-y-auto bg-surface-2/20">
        {/* Tabs - Reverted to Original Feature/Structure */}
        <div className="flex gap-8 border-b border-border mb-8">
          <button
            onClick={() => switchTab('scheduled')}
            className={cn(
              "pb-4 text-xs font-black uppercase tracking-widest transition-all relative",
              activeTab === 'scheduled' ? "text-primary" : "text-text-tertiary hover:text-text"
            )}
          >
            Scheduled Ingestion
            {activeTab === 'scheduled' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />}
          </button>
          <button
            onClick={() => switchTab('adhoc')}
            className={cn(
              "pb-4 text-xs font-black uppercase tracking-widest transition-all relative",
              activeTab === 'adhoc' ? "text-primary" : "text-text-tertiary hover:text-text"
            )}
          >
            Ad-Hoc Protocols
            {activeTab === 'adhoc' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />}
          </button>
        </div>

        {/* --- STEP 1: Main Lists --- */}
        {step === 'list' && activeTab === 'scheduled' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[11px] font-black uppercase tracking-widest text-text-tertiary">Active Synchronizations</h3>
              <button
                onClick={() => setStep('db_creds')}
                className="btn btn-primary text-xs flex items-center gap-2">
                <Plus size={14} /> Initialize Connection
              </button>
            </div>

            <div className="grid gap-6">
              <div className="card p-6 flex flex-col md:flex-row items-start gap-6 group hover:border-primary/50 transition-all">
                <div className="size-12 rounded-2xl bg-primary/5 border border-primary/10 flex items-center justify-center text-primary flex-shrink-0 group-hover:bg-primary group-hover:text-white transition-all">
                  <Database size={24} />
                </div>
                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-bold text-lg tracking-tight">PostgreSQL - Production Entities</h4>
                      <p className="text-xs text-text-tertiary font-mono mt-1 opacity-60">jdbc:postgresql://compute-cluster.internal:5432/main</p>
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1 bg-success/10 text-success text-[10px] font-black rounded uppercase tracking-widest border border-success/20">
                      <Activity size={12} className="animate-pulse" /> Live & Healthy
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 mt-6 pt-6 border-t border-border-subtle">
                    <div>
                      <p className="text-[9px] font-bold text-text-tertiary uppercase tracking-widest mb-1">Interval</p>
                      <p className="text-sm font-bold flex items-center gap-2">
                        <Clock size={14} className="text-text-tertiary" /> Every 12H Cycle
                      </p>
                    </div>
                    <div>
                      <p className="text-[9px] font-bold text-text-tertiary uppercase tracking-widest mb-1">Last Convergence</p>
                      <p className="text-sm font-bold">14:32 UTC</p>
                    </div>
                    <div>
                      <p className="text-[9px] font-bold text-text-tertiary uppercase tracking-widest mb-1">Entity Count</p>
                      <p className="text-sm font-black font-mono tracking-tighter">1,248,902</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 'list' && activeTab === 'adhoc' && (
          <div className="max-w-4xl mx-auto space-y-12">
            <div className="card p-12 text-center border-dashed border-2 border-border-subtle hover:border-primary transition-all group bg-white shadow-2xl shadow-primary/5">
              <div className="size-16 bg-surface-2 rounded-3xl flex items-center justify-center mx-auto mb-6 border border-border group-hover:bg-primary group-hover:text-white transition-all">
                <FileSpreadsheet className="text-text-tertiary group-hover:text-white" size={32} />
              </div>
              <h3 className="text-xl font-bold tracking-tight mb-2">Ingest Vector Sets</h3>
              <p className="text-sm text-text-tertiary mb-8 max-w-sm mx-auto">Drop your CSV, Parquet, or Excel files into the neural buffer for immediate ingestion.</p>

              <button
                onClick={handleFileUpload}
                className="btn btn-primary px-8 py-3 text-sm">
                <Upload size={18} /> Select Source Files
              </button>
            </div>

            <div className="space-y-6">
              <h3 className="text-[11px] font-black uppercase tracking-widest text-text-tertiary">Recent Buffers</h3>
              <div className="card divide-y divide-border-subtle overflow-hidden">
                {[
                  { name: 'q3_transaction_vectors.csv', size: '14.2 MB', time: '2H ago' },
                  { name: 'user_behavior_dump.parquet', size: '3.1 MB', time: 'Yesterday' }
                ].map((file, i) => (
                  <div key={i} className="flex items-center justify-between p-5 hover:bg-surface-2 transition-all group">
                    <div className="flex items-center gap-4">
                      <div className="size-10 rounded-xl bg-surface-2 flex items-center justify-center border border-border group-hover:bg-white transition-all">
                        <FileSpreadsheet size={18} className="text-text-tertiary" />
                      </div>
                      <div>
                        <p className="text-sm font-bold tracking-tight">{file.name}</p>
                        <p className="text-[10px] text-text-tertiary font-bold uppercase tracking-widest mt-1">{file.size} • {file.time}</p>
                      </div>
                    </div>
                    <span className="px-3 py-1 bg-success/5 text-success text-[10px] font-black rounded uppercase tracking-widest border border-success/20">Finalized</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* --- STEP 1.5: Database Credentials --- */}
        {step === 'db_creds' && (
          <div className="animate-in slide-in-from-right duration-500 max-w-2xl mx-auto bg-white border border-border rounded-3xl shadow-2xl shadow-primary/10 overflow-hidden">
            <div className="p-8 border-b border-border bg-surface-2/50">
              <h3 className="text-xl font-bold tracking-tight flex items-center gap-3">
                <Database size={24} className="text-primary" /> Operational Gate
              </h3>
              <p className="text-xs text-text-tertiary mt-1 uppercase font-bold tracking-widest">Provide cluster credentials for secure object replication.</p>
            </div>

            <form onSubmit={handleConnectDb} className="p-8 space-y-8">
              <div className="grid grid-cols-2 gap-6">
                <div className="col-span-2 space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Provider Architecture</label>
                  <select className="input text-sm h-12">
                    <option value="postgres">PostgreSQL / TimescaleDB</option>
                    <option value="mysql">MySQL Cluster</option>
                    <option value="snowflake">Snowflake Warehouse</option>
                  </select>
                </div>

                <div className="col-span-2 md:col-span-1 space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Endpoint</label>
                  <input type="text" required placeholder="db.infra.internal" className="input text-sm h-12" />
                </div>

                <div className="col-span-2 md:col-span-1 space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Port</label>
                  <input type="text" required defaultValue="5432" className="input text-sm h-12" />
                </div>

                <div className="col-span-2 space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Internal Schema</label>
                  <input type="text" required placeholder="production_v1" className="input text-sm h-12" />
                </div>

                <div className="col-span-2 md:col-span-1 space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Identity</label>
                  <input type="text" required placeholder="svc_account" className="input text-sm h-12" />
                </div>

                <div className="col-span-2 md:col-span-1 space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Key Protocol</label>
                  <input type="password" required placeholder="••••••••" className="input text-sm h-12" />
                </div>
              </div>

              <div className="pt-6 border-t border-border flex justify-end gap-3">
                <button type="button" onClick={() => setStep('list')} className="btn btn-secondary px-8">Esc</button>
                <button type="submit" className="btn btn-primary px-8 flex items-center gap-2">
                  Verify & Connect <ArrowRight size={16} />
                </button>
              </div>
            </form>
          </div>
        )}

        {/* --- STEP 2: Configure & Preview --- */}
        {step === 'preview' && (
          <div className="animate-in slide-in-from-right duration-500 max-w-6xl mx-auto space-y-8">

            <div className="card overflow-hidden">
              <div className="p-8 border-b border-border bg-white flex justify-between items-center">
                <div>
                  <div className="flex items-center gap-3 text-primary font-black tracking-tighter text-2xl uppercase">
                    {activeTab === 'scheduled' ? <Database size={24} /> : <FileSpreadsheet size={24} />}
                    {sourceName}
                  </div>
                  <p className="text-xs text-text-tertiary font-bold tracking-widest uppercase mt-1">Staging Layer Configuration</p>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => activeTab === 'scheduled' ? setStep('db_creds') : setStep('list')}
                    className="btn btn-secondary px-6">
                    Revise
                  </button>
                  <button
                    onClick={handleImport}
                    className="btn btn-primary px-8 flex items-center gap-2 shadow-xl shadow-primary/10">
                    {activeTab === 'scheduled' ? 'Establish Sync' : 'Finalize Import'} <ArrowRight size={16} />
                  </button>
                </div>
              </div>

              <div className="p-8 grid grid-cols-2 gap-16">
                <div className="space-y-6">
                  <h4 className="text-[11px] font-black uppercase tracking-widest text-text-tertiary border-b border-border-subtle pb-3">Destination Geometry</h4>
                  <div className="flex gap-8">
                    <label className="flex items-center gap-3 text-xs font-bold cursor-pointer group">
                      <input
                        type="radio"
                        name="target"
                        value="new_table"
                        checked={targetTable === 'new_table'}
                        onChange={(e) => setTargetTable(e.target.value)}
                        className="accent-primary size-4"
                      />
                      Generate New Target
                    </label>
                    <label className="flex items-center gap-3 text-xs font-bold cursor-pointer group">
                      <input
                        type="radio"
                        name="target"
                        value="existing"
                        checked={targetTable === 'existing'}
                        onChange={(e) => setTargetTable(e.target.value)}
                        className="accent-primary size-4"
                      />
                      Append to Cluster
                    </label>
                  </div>

                  {targetTable === 'new_table' ? (
                    <div className="space-y-2">
                      <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Object Identifier</label>
                      <input
                        type="text"
                        value={newTableName}
                        onChange={(e) => setNewTableName(e.target.value)}
                        className="input text-sm h-12"
                        placeholder="namespace.table_name"
                      />
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Target Entity</label>
                      <select className="input text-sm h-12">
                        <option>production.main_vectors</option>
                        <option>historical.archive_set</option>
                      </select>
                    </div>
                  )}
                </div>

                {activeTab === 'scheduled' && (
                  <div className="space-y-6">
                    <h4 className="text-[11px] font-black uppercase tracking-widest text-text-tertiary border-b border-border-subtle pb-3">Temporal Sync Protocol</h4>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <label className="text-[10px] font-black uppercase tracking-widest text-text-tertiary">Cadence</label>
                        <select
                          value={scheduleFreq}
                          onChange={e => setScheduleFreq(e.target.value)}
                          className="input text-sm h-12">
                          <option value="hourly">Real-time (Hourly)</option>
                          <option value="daily">Nightly Batch (00:00)</option>
                          <option value="weekly">Weekly Audit (Sun)</option>
                        </select>
                      </div>

                      <label className="flex items-center gap-3 text-xs font-bold cursor-pointer mt-4">
                        <input type="checkbox" defaultChecked className="accent-primary size-4 rounded" />
                        Execute initial convergence immediate
                      </label>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Data Preview Panel */}
            <div className="card overflow-hidden">
              <div className="px-8 py-5 border-b border-border flex items-center justify-between bg-surface-2/30">
                <h4 className="text-[11px] font-black flex items-center gap-3 uppercase tracking-widest text-text-tertiary">
                  <Table size={16} /> Data Pre-Visualization
                </h4>
                <span className="text-[9px] font-bold text-primary bg-primary/5 px-3 py-1 rounded-full uppercase tracking-widest border border-primary/20">Head Buffer (5 Rows)</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-border bg-white">
                      {['ID', 'ENTITY_NAME', 'METADATA_LOC', 'DOMAIN', 'QUANT_VAL'].map(h => (
                        <th key={h} className="px-6 py-4 font-black text-text-tertiary tracking-widest uppercase border-r border-border-subtle last:border-0">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="font-mono divide-y divide-border-subtle">
                    {[1, 2, 3, 4, 5].map(i => (
                      <tr key={i} className="hover:bg-surface-2 transition-all group">
                        <td className="px-6 py-3 border-r border-border-subtle group-last:border-r">{i}</td>
                        <td className="px-6 py-3 border-r border-border-subtle font-bold text-text truncate">Entity_Alpha_{i * 92}</td>
                        <td className="px-6 py-3 border-r border-border-subtle opacity-50 truncate">meta.internal/node_{i}</td>
                        <td className="px-6 py-3 border-r border-border-subtle font-bold">{i % 2 === 0 ? 'Engineering' : 'Global Ops'}</td>
                        <td className="px-6 py-3 font-black text-primary">{(Math.random() * 100000).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* --- STEP 3: Success --- */}
        {step === 'success' && (
          <div className="max-w-xl mx-auto mt-24 card p-12 text-center animate-in zoom-in duration-500 shadow-2xl shadow-primary/20 bg-white">
            <div className="size-20 bg-success/10 text-success rounded-full flex items-center justify-center mx-auto mb-8 border border-success/20 animate-bounce">
              <CheckCircle2 size={40} />
            </div>
            <h3 className="text-3xl font-black tracking-tighter mb-4">
              {activeTab === 'scheduled' ? 'Convergence Verified' : 'Buffer Finalized'}
            </h3>
            <p className="text-sm text-text-tertiary max-w-sm mx-auto leading-relaxed">
              Object stream successfully mapped to <strong className="text-primary">{targetTable === 'new_table' ? newTableName : 'existing cluster'}</strong>. Pipeline initialized.
            </p>
            <div className="mt-12">
              <button onClick={() => setStep('list')} className="btn btn-primary px-12 py-3 text-xs">Return to Workspace</button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
