import { useState, useEffect, useRef } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { cn } from "../../lib/cn";

type TrainingMode = "auto" | "finetune";
type ProblemType = "Classification" | "Regression";
type Algorithm = "XGBoost" | "Random Forest" | "Logistic Regression" | "Neural Network";
type TrainingStatus = "idle" | "training" | "completed" | "error";

interface TrainingConfig {
  problemType: ProblemType;
  algorithm: Algorithm;
  learningRate: number;
  epochs: number;
  trainSplit: number;
  crossValidation: number;
  earlyStopping: boolean;
}

function ProgressBar({ progress, status }: { progress: number; status: TrainingStatus }) {
  const colors = {
    idle: "bg-surface-2",
    training: "bg-accent",
    completed: "bg-success",
    error: "bg-danger",
  };
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
      <div
        className={cn("h-full rounded-full transition-all duration-500 ease-out", colors[status])}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}

function MetricGraph({ data, label, color }: { data: number[]; label: string; color: string }) {
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-text-tertiary">{label}</span>
        <span className={cn("font-mono font-medium", color)}>{data[data.length - 1]?.toFixed(4) ?? "—"}</span>
      </div>
      <div className="h-12 flex items-end gap-0.5">
        {data.slice(-20).map((v, i) => (
          <div
            key={i}
            className={cn("flex-1 rounded-t-sm transition-all", color.replace("text-", "bg-"))}
            style={{ height: `${((v - min) / range) * 100}%`, minHeight: "2px" }}
          />
        ))}
      </div>
    </div>
  );
}

function TrainingPanel({
  mode,
  config,
  setConfig,
  status,
  progress,
  epoch,
  maxEpochs,
  lossData,
  accuracyData,
  logs,
  onStartTraining,
  onStopTraining,
  onModeChange,
}: {
  mode: TrainingMode;
  config: TrainingConfig;
  setConfig: (c: TrainingConfig) => void;
  onModeChange: (m: TrainingMode) => void;
  status: TrainingStatus;
  progress: number;
  epoch: number;
  maxEpochs: number;
  lossData: number[];
  accuracyData: number[];
  logs: string[];
  onStartTraining: () => void;
  onStopTraining: () => void;
}) {
  return (
    <div className="card flex flex-col overflow-hidden">
      <div className="border-b border-border-subtle px-5 py-4">
        <h2 className="text-base font-semibold text">Training</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-5 space-y-6">
        <div className="space-y-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Training Mode</label>
          <div className="flex rounded-lg border border-border p-1">
            <button
              onClick={() => onModeChange("auto")}
              className={cn(
                "flex-1 rounded-md px-4 py-2 text-sm font-medium transition-all",
                mode === "auto" ? "bg-accent text-white shadow-sm" : "text-text-secondary hover:bg-surface-2"
              )}
            >
              Auto
            </button>
            <button
              onClick={() => onModeChange("finetune")}
              className={cn(
                "flex-1 rounded-md px-4 py-2 text-sm font-medium transition-all",
                mode === "finetune" ? "bg-accent text-white shadow-sm" : "text-text-secondary hover:bg-surface-2"
              )}
            >
              Fine-tune
            </button>
          </div>
        </div>

        {mode === "finetune" && (
          <div className="space-y-4 rounded-lg border border-border-subtle bg-surface-2/30 p-4">
            <h3 className="text-sm font-semibold text">Parameters</h3>
            
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-text-secondary">Problem Type</label>
                <select
                  value={config.problemType}
                  onChange={(e) => setConfig({ ...config, problemType: e.target.value as ProblemType })}
                  className="input w-full text-sm"
                >
                  <option value="Classification">Classification</option>
                  <option value="Regression">Regression</option>
                </select>
              </div>
              
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-text-secondary">Algorithm</label>
                <select
                  value={config.algorithm}
                  onChange={(e) => setConfig({ ...config, algorithm: e.target.value as Algorithm })}
                  className="input w-full text-sm"
                >
                  <option value="XGBoost">XGBoost</option>
                  <option value="Random Forest">Random Forest</option>
                  <option value="Logistic Regression">Logistic Regression</option>
                  <option value="Neural Network">Neural Network</option>
                </select>
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-text-secondary">Learning Rate</label>
                <span className="text-xs font-mono text-text-tertiary">{config.learningRate.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min={0.001}
                max={1}
                step={0.001}
                value={config.learningRate}
                onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
                className="w-full accent-accent"
                disabled={status === "training"}
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-text-secondary">Epochs / Iterations</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 10 })}
                  className="input w-20 text-center text-sm"
                  min={1}
                  max={1000}
                  disabled={status === "training"}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-text-secondary">Train/Val Split</label>
                <span className="text-xs font-mono text-text-tertiary">{config.trainSplit}%</span>
              </div>
              <input
                type="range"
                min={70}
                max={95}
                step={5}
                value={config.trainSplit}
                onChange={(e) => setConfig({ ...config, trainSplit: parseInt(e.target.value) })}
                className="w-full accent-accent"
                disabled={status === "training"}
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-text-secondary">Cross-Validation</label>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setConfig({ ...config, crossValidation: Math.max(3, config.crossValidation - 1) })}
                    className="btn btn-secondary px-2 py-1"
                    disabled={status === "training"}
                  >
                    −
                  </button>
                  <span className="flex-1 text-center text-sm font-medium">{config.crossValidation}</span>
                  <button
                    onClick={() => setConfig({ ...config, crossValidation: Math.min(10, config.crossValidation + 1) })}
                    className="btn btn-secondary px-2 py-1"
                    disabled={status === "training"}
                  >
                    +
                  </button>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-text-secondary">Early Stopping</label>
                <button
                  onClick={() => setConfig({ ...config, earlyStopping: !config.earlyStopping })}
                  className={cn(
                    "relative w-12 h-6 rounded-full transition-colors",
                    config.earlyStopping ? "bg-accent" : "bg-surface-3"
                  )}
                  disabled={status === "training"}
                >
                  <span
                    className={cn(
                      "absolute top-1 w-4 h-4 rounded-full bg-white shadow transition-transform",
                      config.earlyStopping ? "left-7" : "left-1"
                    )}
                  />
                </button>
              </div>
            </div>
          </div>
        )}

        {status === "training" && (
          <div className="space-y-4 rounded-lg border border-border-subtle bg-surface-2/30 p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text">Training Progress</h3>
              <span className="text-xs text-text-tertiary">Epoch {epoch}/{maxEpochs}</span>
            </div>
            
            <ProgressBar progress={progress} status={status} />
            
            <div className="grid gap-4 sm:grid-cols-2">
              <MetricGraph data={lossData} label="Loss" color="text-danger" />
              <MetricGraph data={accuracyData} label="Accuracy" color="text-success" />
            </div>

            <button
              onClick={onStopTraining}
              className="btn btn-secondary w-full"
            >
              Stop Training
            </button>
          </div>
        )}

        <div className="space-y-2">
          <h3 className="text-sm font-semibold text">Training Logs</h3>
          <div className="h-40 overflow-y-auto rounded-lg border border-border-subtle bg-surface-2 p-3 font-mono text-xs text-text-secondary">
            {logs.length === 0 ? (
              <span className="text-text-tertiary">No logs yet. Start training to see logs.</span>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="mb-1">{log}</div>
              ))
            )}
          </div>
        </div>

        {status !== "training" && (
          <button
            onClick={onStartTraining}
            className="btn btn-primary w-full"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {mode === "auto" ? "Start Auto Training" : "Start Training"}
          </button>
        )}
      </div>
    </div>
  );
}

export function MLTrainingPage() {
  const [mode, setMode] = useState<TrainingMode>("auto");
  const [config, setConfig] = useState<TrainingConfig>({
    problemType: "Classification",
    algorithm: "XGBoost",
    learningRate: 0.1,
    epochs: 50,
    trainSplit: 80,
    crossValidation: 5,
    earlyStopping: true,
  });
  const [status, setStatus] = useState<TrainingStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [lossData, setLossData] = useState<number[]>([]);
  const [accuracyData, setAccuracyData] = useState<number[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const trainingInterval = useRef<number | null>(null);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  };

  const startTraining = () => {
    if (status === "training") return;
    
    setStatus("training");
    setProgress(0);
    setEpoch(0);
    setLossData([]);
    setAccuracyData([]);
    setLogs([]);
    
    addLog(`Starting ${mode === "auto" ? "Auto" : "Fine-tune"} training...`);
    addLog(`Problem type: ${config.problemType}`);
    addLog(`Algorithm: ${mode === "auto" ? "Auto (All algorithms)" : config.algorithm}`);
    if (mode === "finetune") {
      addLog(`Learning rate: ${config.learningRate}`);
      addLog(`Epochs: ${config.epochs}`);
      addLog(`Train/Val split: ${config.trainSplit}%`);
      addLog(`Cross-validation: ${config.crossValidation}`);
      addLog(`Early stopping: ${config.earlyStopping ? "enabled" : "disabled"}`);
    }
    addLog("Loading dataset...");
    addLog("Dataset loaded: 10000 samples, 24 features");
    addLog("Preprocessing data...");
    addLog("Starting training loop...");

    let currentEpoch = 0;
    const maxEpochs = mode === "finetune" ? config.epochs : 100;
    
    trainingInterval.current = window.setInterval(() => {
      currentEpoch += 1;
      const newProgress = (currentEpoch / maxEpochs) * 100;
      
      const newLoss = 1.5 * Math.exp(-currentEpoch / 30) + 0.1 + Math.random() * 0.05;
      const newAccuracy = 0.5 + 0.45 * (1 - Math.exp(-currentEpoch / 20)) + Math.random() * 0.02;
      
      setProgress(newProgress);
      setEpoch(currentEpoch);
      setLossData(prev => [...prev.slice(-19), newLoss]);
      setAccuracyData(prev => [...prev.slice(-19), newAccuracy]);
      
      if (currentEpoch % 5 === 0) {
        addLog(`Epoch ${currentEpoch}/${maxEpochs} - Loss: ${newLoss.toFixed(4)}, Accuracy: ${newAccuracy.toFixed(4)}`);
      }
      
      if (currentEpoch >= maxEpochs) {
        stopTraining(true);
      }
    }, 200);
  };

  const stopTraining = (completed = false) => {
    if (trainingInterval.current) {
      clearInterval(trainingInterval.current);
      trainingInterval.current = null;
    }
    
    if (completed) {
      setStatus("completed");
      setProgress(100);
      addLog("Training completed successfully!");
      addLog("Validation accuracy: 94.8%");
      addLog("Model saved to registry");
    } else {
      setStatus("idle");
      addLog("Training stopped by user");
    }
  };

  useEffect(() => {
    return () => {
      if (trainingInterval.current) {
        clearInterval(trainingInterval.current);
      }
    };
  }, []);

  return (
    <section className="animate-in fade-in duration-300">
      <PageHeader
        title="ML Training"
        subtitle="Train machine learning models with real-time progress tracking."
        actions={
          <div className="flex items-center gap-2">
            <button className="btn btn-secondary">
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Import Data
            </button>
            <button className="btn btn-primary">
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Model
            </button>
          </div>
        }
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <TrainingPanel
            mode={mode}
            config={config}
            setConfig={setConfig}
            onModeChange={setMode}
            status={status}
            progress={progress}
            epoch={epoch}
            maxEpochs={mode === "finetune" ? config.epochs : 100}
            lossData={lossData}
            accuracyData={accuracyData}
            logs={logs}
            onStartTraining={startTraining}
            onStopTraining={() => stopTraining(false)}
          />
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="card p-5">
            <h3 className="text-sm font-semibold text mb-4">Training Overview</h3>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg border border-border-subtle bg-surface-2/30 p-4">
                <p className="text-xs font-medium uppercase text-text-tertiary">Status</p>
                <p className="mt-1 text-lg font-semibold text">{status === "idle" ? "Ready" : status === "training" ? "Training" : status === "completed" ? "Completed" : "Error"}</p>
              </div>
              <div className="rounded-lg border border-border-subtle bg-surface-2/30 p-4">
                <p className="text-xs font-medium uppercase text-text-tertiary">Current Epoch</p>
                <p className="mt-1 text-lg font-semibold text">{epoch} / {mode === "finetune" ? config.epochs : 100}</p>
              </div>
              <div className="rounded-lg border border-border-subtle bg-surface-2/30 p-4">
                <p className="text-xs font-medium uppercase text-text-tertiary">Progress</p>
                <p className="mt-1 text-lg font-semibold text">{progress.toFixed(1)}%</p>
              </div>
            </div>
          </div>

          {status === "completed" && (
            <div className="card p-5">
              <h3 className="text-sm font-semibold text mb-4">Training Results</h3>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border border-success/30 bg-success-muted/30 p-4">
                  <p className="text-xs font-medium uppercase text-text-tertiary">Validation Accuracy</p>
                  <p className="mt-1 text-2xl font-bold text-success">94.8%</p>
                </div>
                <div className="rounded-lg border border-border-subtle bg-surface-2/30 p-4">
                  <p className="text-xs font-medium uppercase text-text-tertiary">Training Time</p>
                  <p className="mt-1 text-2xl font-bold text">4m 32s</p>
                </div>
              </div>
              <button className="btn btn-primary mt-4 w-full">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Model
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}