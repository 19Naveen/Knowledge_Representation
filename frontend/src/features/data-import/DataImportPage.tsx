import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { Database, FileSpreadsheet, Activity, Clock, Upload, Plus, Table, CheckCircle2, ArrowRight, Server, Key, User } from "lucide-react";

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
    <div className="flex h-full flex-col">
      <PageHeader 
        title="Data Import" 
        subtitle="Manage regular database connections or perform ad-hoc file uploads."
      />

      <div className="flex-1 p-6 overflow-y-auto bg-[#fafafa]">
        {/* Tabs */}
        <div className="flex gap-1 border-b border-[#eaeaea] mb-6">
          <button 
            onClick={() => switchTab('scheduled')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'scheduled' 
                ? 'border-black text-black' 
                : 'border-transparent text-[#666] hover:text-black'
            }`}
          >
            Scheduled Imports
          </button>
          <button 
            onClick={() => switchTab('adhoc')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'adhoc' 
                ? 'border-black text-black' 
                : 'border-transparent text-[#666] hover:text-black'
            }`}
          >
            Ad-Hoc Uploads
          </button>
        </div>

        {/* --- STEP 1: Main Lists --- */}
        {step === 'list' && activeTab === 'scheduled' && (
          <div className="animate-in fade-in slide-in-from-bottom-2">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-semibold">Active Database Connections</h3>
              <button 
                onClick={() => setStep('db_creds')}
                className="flex items-center gap-2 px-3 py-1.5 bg-black text-white text-xs font-medium rounded hover:bg-[#333]">
                <Plus size={14} /> New Connection
              </button>
            </div>
            
            <div className="grid gap-4">
              <div className="bg-white border border-[#eaeaea] rounded-lg p-5 flex items-start gap-4">
                <div className="w-10 h-10 rounded bg-[#0070f3]/10 flex items-center justify-center text-[#0070f3] flex-shrink-0">
                  <Database size={20} />
                </div>
                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-semibold text-sm">PostgreSQL - Production Users</h4>
                      <p className="text-xs text-[#666] mt-0.5">jdbc:postgresql://prod-db.internal:5432/main</p>
                    </div>
                    <div className="flex items-center gap-1.5 px-2 py-1 bg-[#ccfbf1] text-[#00701a] text-[10px] font-bold rounded uppercase tracking-wider">
                      <Activity size={12} /> Healthy
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-6 mt-4 pt-4 border-t border-[#eaeaea]">
                    <div>
                      <p className="text-[10px] text-[#666] uppercase mb-1">Schedule</p>
                      <p className="text-xs font-medium flex items-center gap-1.5">
                        <Clock size={12} className="text-[#666]" /> Every 12 Hours
                      </p>
                    </div>
                    <div>
                      <p className="text-[10px] text-[#666] uppercase mb-1">Last Sync</p>
                      <p className="text-xs font-medium">Today, 04:30 AM</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-[#666] uppercase mb-1">Rows Imported</p>
                      <p className="text-xs font-medium font-mono">1.2M</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 'list' && activeTab === 'adhoc' && (
          <div className="animate-in fade-in slide-in-from-bottom-2 max-w-3xl mx-auto mt-8 text-center border-b border-[#eaeaea] pb-12">
            <div className="bg-white border text-center border-dashed border-[#ccc] rounded-xl p-12 max-w-2xl mx-auto">
              <div className="w-12 h-12 bg-[#fafafa] rounded-full flex items-center justify-center mx-auto mb-4 border border-[#eaeaea]">
                <FileSpreadsheet className="text-[#666]" size={24} />
              </div>
              <h3 className="text-base font-semibold mb-1">Upload CSV or Excel</h3>
              <p className="text-sm text-[#666] mb-6">Drag and drop your file here, or click to browse.</p>
              
              <button 
                onClick={handleFileUpload}
                className="flex items-center justify-center gap-2 px-4 py-2 bg-black text-white text-sm font-medium rounded hover:bg-[#333] mx-auto">
                <Upload size={16} /> Choose File
              </button>
              
              <p className="text-xs text-[#999] mt-4">Max file size: 50MB. Supported formats: .csv, .xlsx</p>
            </div>

            <div className="mt-12 text-left max-w-2xl mx-auto">
               <h3 className="text-sm font-semibold mb-4">Recent Ad-Hoc Uploads</h3>
               <div className="bg-white border border-[#eaeaea] rounded-lg flex flex-col overflow-hidden">
                 <div className="flex items-center justify-between p-4 border-b border-[#eaeaea] hover:bg-[#fafafa]">
                   <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded bg-[#fafafa] flex items-center justify-center border border-[#eaeaea]"><FileSpreadsheet size={16} className="text-[#666]"/></div>
                      <div>
                        <p className="text-sm font-medium">sales_q3_report.csv</p>
                        <p className="text-xs text-[#666] mt-0.5">14.2 MB • Uploaded 2 hours ago</p>
                      </div>
                   </div>
                   <span className="px-2 py-1 bg-[#ccfbf1] text-[#00701a] text-[10px] font-bold rounded uppercase tracking-wider">Imported</span>
                 </div>
                 <div className="flex items-center justify-between p-4 border-b border-[#eaeaea] hover:bg-[#fafafa]">
                   <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded bg-[#fafafa] flex items-center justify-center border border-[#eaeaea]"><FileSpreadsheet size={16} className="text-[#666]"/></div>
                      <div>
                        <p className="text-sm font-medium">employee_feedback_dump.xlsx</p>
                        <p className="text-xs text-[#666] mt-0.5">3.1 MB • Uploaded yesterday</p>
                      </div>
                   </div>
                   <span className="px-2 py-1 bg-[#ccfbf1] text-[#00701a] text-[10px] font-bold rounded uppercase tracking-wider">Imported</span>
                 </div>
               </div>
            </div>
          </div>
        )}

        {/* --- STEP 1.5: Database Credentials (SCHEDULED ONLY) --- */}
        {step === 'db_creds' && (
          <div className="animate-in fade-in slide-in-from-right-4 max-w-2xl mx-auto mt-6 bg-white border border-[#eaeaea] rounded-xl shadow-sm overflow-hidden">
            <div className="p-6 border-b border-[#eaeaea] bg-[#fafafa]">
              <h3 className="text-base font-semibold flex items-center gap-2">
                <Database size={18} className="text-[#0070f3]"/> Connect to Database
              </h3>
              <p className="text-sm text-[#666] mt-1.5">Provide credentials to securely connect to your database source.</p>
            </div>
            
            <form onSubmit={handleConnectDb} className="p-6 flex flex-col gap-6">
              <div className="grid grid-cols-2 gap-5">
                <div className="col-span-2">
                  <label className="block text-xs font-semibold text-[#666] mb-1.5 flex items-center gap-1.5">
                    <Database size={14} className="text-[#999]"/> Database Type
                  </label>
                  <select className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black bg-white">
                    <option value="postgres">PostgreSQL</option>
                    <option value="mysql">MySQL</option>
                    <option value="snowflake">Snowflake</option>
                  </select>
                </div>
                
                <div className="col-span-2 md:col-span-1">
                  <label className="block text-xs font-semibold text-[#666] mb-1.5 flex items-center gap-1.5">
                    <Server size={14} className="text-[#999]"/> Host / Endpoint
                  </label>
                  <input type="text" required placeholder="db.example.com" className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black transition-colors" />
                </div>
                
                <div className="col-span-2 md:col-span-1">
                  <label className="block text-xs font-semibold text-[#666] mb-1.5 flex items-center gap-1.5">
                    <Server size={14} className="text-[#999]"/> Port
                  </label>
                  <input type="text" required defaultValue="5432" className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black transition-colors" />
                </div>

                <div className="col-span-2">
                  <label className="block text-xs font-semibold text-[#666] mb-1.5 flex items-center gap-1.5">
                    <Database size={14} className="text-[#999]"/> Database Name
                  </label>
                  <input type="text" required placeholder="production_main" className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black transition-colors" />
                </div>

                <div className="col-span-2 md:col-span-1">
                  <label className="block text-xs font-semibold text-[#666] mb-1.5 flex items-center gap-1.5">
                    <User size={14} className="text-[#999]"/> Username
                  </label>
                  <input type="text" required placeholder="postgres" className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black transition-colors" />
                </div>

                <div className="col-span-2 md:col-span-1">
                  <label className="block text-xs font-semibold text-[#666] mb-1.5 flex items-center gap-1.5">
                    <Key size={14} className="text-[#999]"/> Password
                  </label>
                  <input type="password" required placeholder="••••••••" className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black transition-colors" />
                </div>
              </div>

              <div className="pt-5 mt-2 border-t border-[#eaeaea] flex justify-end gap-3">
                <button type="button" onClick={() => setStep('list')} className="px-5 py-2 border border-[#eaeaea] text-[#333] text-sm font-medium rounded hover:bg-[#fafafa]">
                  Cancel
                </button>
                <button type="submit" className="px-5 py-2 bg-black text-white text-sm font-medium rounded hover:bg-[#333] flex items-center gap-2">
                  Test & Connect <ArrowRight size={16} />
                </button>
              </div>
            </form>
          </div>
        )}

        {/* --- STEP 2: Configure & Preview (SHARED) --- */}
        {step === 'preview' && (
          <div className="animate-in fade-in slide-in-from-right-4 max-w-5xl mx-auto flex flex-col gap-6">
            
            {/* Configuration Panel */}
            <div className="bg-white border border-[#eaeaea] rounded-lg w-full flex flex-col">
              <div className="p-6 border-b border-[#eaeaea] flex justify-between items-center">
                <div>
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    {activeTab === 'scheduled' ? <Database className="text-[#0070f3]" size={20} /> : <FileSpreadsheet className="text-[#0070f3]" size={20} />}
                    {sourceName}
                  </h3>
                  <p className="text-sm text-[#666] mt-1">Configure {activeTab === 'scheduled' ? 'sync destination' : 'table destination'} and preview data.</p>
                </div>
                <div className="flex gap-3">
                  <button 
                    onClick={() => activeTab === 'scheduled' ? setStep('db_creds') : setStep('list')}
                    className="px-5 py-2 border border-[#eaeaea] text-[#333] text-sm font-medium rounded hover:bg-[#fafafa]">
                    Back
                  </button>
                  <button 
                    onClick={handleImport}
                    className="px-5 py-2 bg-black text-white text-sm font-medium rounded hover:bg-[#333] flex items-center gap-2">
                    {activeTab === 'scheduled' ? 'Create Sync' : 'Start Import'} <ArrowRight size={16} />
                  </button>
                </div>
              </div>

              <div className="p-6 grid grid-cols-2 gap-12">
                {/* Target Configuration */}
                <div>
                  <h4 className="text-sm font-semibold mb-4 border-b border-[#eaeaea] pb-2">Destination Target</h4>
                  <div className="flex flex-col gap-1.5 mb-4">
                    <div className="flex gap-4">
                      <label className="flex items-center gap-2 text-sm cursor-pointer">
                        <input 
                          type="radio" 
                          name="target" 
                          value="new_table"
                          checked={targetTable === 'new_table'}
                          onChange={(e) => setTargetTable(e.target.value)}
                          className="accent-black"
                        />
                        Create New Table
                      </label>
                      <label className="flex items-center gap-2 text-sm cursor-pointer">
                        <input 
                          type="radio" 
                          name="target" 
                          value="existing"
                          checked={targetTable === 'existing'}
                          onChange={(e) => setTargetTable(e.target.value)}
                          className="accent-black"
                        />
                        Append to Existing
                      </label>
                    </div>
                  </div>

                  {targetTable === 'new_table' ? (
                    <div>
                      <label className="block text-xs font-semibold text-[#666] mb-1.5">New Table Name</label>
                      <input 
                        type="text" 
                        value={newTableName}
                        onChange={(e) => setNewTableName(e.target.value)}
                        className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black"
                        placeholder="Enter table name..."
                      />
                    </div>
                  ) : (
                    <div>
                      <label className="block text-xs font-semibold text-[#666] mb-1.5">Select Existing Table</label>
                      <select className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black bg-white">
                        <option>public.employees</option>
                        <option>public.sales_data</option>
                        <option>public.churn_metrics</option>
                      </select>
                    </div>
                  )}
                </div>

                {/* Additional Config (Schedule for DB) */}
                {activeTab === 'scheduled' && (
                  <div>
                    <h4 className="text-sm font-semibold mb-4 border-b border-[#eaeaea] pb-2">Sync Schedule</h4>
                    <div>
                      <label className="block text-xs font-semibold text-[#666] mb-1.5">Frequency</label>
                      <select 
                        value={scheduleFreq}
                        onChange={e => setScheduleFreq(e.target.value)}
                        className="w-full px-3 py-2 border border-[#eaeaea] rounded text-sm focus:outline-none focus:border-black bg-white mb-4">
                        <option value="hourly">Every Hour</option>
                        <option value="daily">Daily (Midnight)</option>
                        <option value="weekly">Weekly (Sunday)</option>
                        <option value="custom">Custom Cron...</option>
                      </select>
                      
                      <label className="flex items-center gap-2 text-sm cursor-pointer">
                        <input type="checkbox" defaultChecked className="accent-black rounded" />
                        Run initial sync immediately
                      </label>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Data Preview Panel (SHARED) */}
            <div className="bg-white border border-[#eaeaea] rounded-lg flex flex-col overflow-hidden">
              <div className="px-6 py-4 border-b border-[#eaeaea] flex items-center justify-between bg-[#fafafa]">
                <h4 className="text-sm font-semibold flex items-center gap-2">
                  <Table size={16} className="text-[#666]" /> Sample Data Preview
                </h4>
                <span className="text-xs text-[#666] font-mono">Previewing first 5 rows</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-[#eaeaea] bg-white">
                      <th className="px-4 py-3 font-semibold text-[#333] border-r border-[#eaeaea]">id <span className="text-[#999] font-normal text-xs ml-2">int</span></th>
                      <th className="px-4 py-3 font-semibold text-[#333] border-r border-[#eaeaea]">name <span className="text-[#999] font-normal text-xs ml-2">string</span></th>
                      <th className="px-4 py-3 font-semibold text-[#333] border-r border-[#eaeaea]">email <span className="text-[#999] font-normal text-xs ml-2">string</span></th>
                      <th className="px-4 py-3 font-semibold text-[#333] border-r border-[#eaeaea]">department <span className="text-[#999] font-normal text-xs ml-2">string</span></th>
                      <th className="px-4 py-3 font-semibold text-[#333]">salary <span className="text-[#999] font-normal text-xs ml-2">float</span></th>
                    </tr>
                  </thead>
                  <tbody className="font-mono text-xs">
                    <tr className="border-b border-[#eaeaea] hover:bg-[#fafafa]">
                      <td className="px-4 py-2 border-r border-[#eaeaea]">1</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">Alice Smith</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">alice@acme.inc</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">Engineering</td>
                      <td className="px-4 py-2">95000.00</td>
                    </tr>
                    <tr className="border-b border-[#eaeaea] hover:bg-[#fafafa]">
                      <td className="px-4 py-2 border-r border-[#eaeaea]">2</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">Bob Jones</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">bob@acme.inc</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">Marketing</td>
                      <td className="px-4 py-2">72500.00</td>
                    </tr>
                    <tr className="border-b border-[#eaeaea] hover:bg-[#fafafa]">
                      <td className="px-4 py-2 border-r border-[#eaeaea]">3</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">Charlie Brown</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">charlie@acme.inc</td>
                      <td className="px-4 py-2 border-r border-[#eaeaea]">Sales</td>
                      <td className="px-4 py-2">88000.00</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* --- STEP 3: Success (SHARED) --- */}
        {step === 'success' && (
          <div className="max-w-md mx-auto mt-12 bg-white border border-[#eaeaea] rounded-xl p-8 text-center animate-in zoom-in-95 shadow-sm">
            <CheckCircle2 size={48} className="mx-auto text-[#0070f3] mb-4" />
            <h3 className="text-xl font-semibold mb-2">
              {activeTab === 'scheduled' ? 'Connection Established' : 'Import Successful'}
            </h3>
            <p className="text-sm text-[#666]">
              Data {activeTab === 'scheduled' ? 'sync' : 'import'} has been successfully configured for <strong className="text-black">{targetTable === 'new_table' ? newTableName : 'existing table'}</strong>. 
            </p>
          </div>
        )}

      </div>
    </div>
  );
}
