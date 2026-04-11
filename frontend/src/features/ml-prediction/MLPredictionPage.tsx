import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeader } from "../../components/shared/PageHeader";
import { cn } from "../../lib/cn";

interface ModelInfo {
  id: string;
  name: string;
  algorithm: string;
  accuracy: number;
  features: string[];
  createdAt: string;
}

const mockModels: ModelInfo[] = [
  { id: "m-1", name: "XGBoost v1", algorithm: "XGBoost", accuracy: 0.948, features: ["age", "income", "education", "employment", "balance", "transactions"], createdAt: "2024-01-15" },
  { id: "m-2", name: "Random Forest", algorithm: "Random Forest", accuracy: 0.923, features: ["age", "income", "education", "employment", "balance", "transactions"], createdAt: "2024-01-14" },
  { id: "m-3", name: "Logistic Regression", algorithm: "Logistic Regression", accuracy: 0.861, features: ["age", "income", "balance", "transactions"], createdAt: "2024-01-10" },
];

const featureDefinitions = [
  { name: "age", label: "Age", type: "number", placeholder: "e.g., 35" },
  { name: "income", label: "Annual Income", type: "number", placeholder: "e.g., 50000" },
  { name: "education", label: "Education Level", type: "select", options: ["High School", "Bachelor", "Master", "PhD"] },
  { name: "employment", label: "Employment Status", type: "select", options: ["Employed", "Self-employed", "Unemployed"] },
  { name: "balance", label: "Account Balance", type: "number", placeholder: "e.g., 10000" },
  { name: "transactions", label: "Monthly Transactions", type: "number", placeholder: "e.g., 15" },
];

interface PredictionResult {
  value: string;
  confidence: number;
}

function ModelSelector({
  models,
  selectedId,
  onSelect,
}: {
  models: ModelInfo[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="card p-5 space-y-4">
      <h2 className="text-base font-semibold text">Model Selector</h2>
      <select
        value={selectedId || ""}
        onChange={(e) => onSelect(e.target.value)}
        className="input w-full text-sm"
      >
        <option value="">Select a model...</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name} ({model.algorithm}) - {(model.accuracy * 100).toFixed(1)}%
          </option>
        ))}
      </select>
      {selectedId && (
        <div className="text-xs text-text-tertiary">
          {models.find((m) => m.id === selectedId)?.features.length} features • Created{" "}
          {models.find((m) => m.id === selectedId)?.createdAt}
        </div>
      )}
    </div>
  );
}

function SinglePredictionPanel({
  model,
  onPredict,
  isPredicting,
  result,
}: {
  model: ModelInfo | null;
  onPredict: (inputs: Record<string, string>) => void;
  isPredicting: boolean;
  result: PredictionResult | null;
}) {
  const [inputs, setInputs] = useState<Record<string, string>>({});

  const features = featureDefinitions.filter((f) =>
    model?.features.includes(f.name)
  );

  const handleSubmit = () => {
    onPredict(inputs);
  };

  const isValid = features.every((f) => inputs[f.name] && inputs[f.name] !== "");

  return (
    <div className="card flex flex-col overflow-hidden">
      <div className="border-b border-border-subtle px-5 py-4">
        <h2 className="text-base font-semibold text">Single Prediction</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {!model ? (
          <p className="text-sm text-text-tertiary">Select a model to make predictions.</p>
        ) : (
          <>
            <p className="text-xs text-text-tertiary">
              Model: <span className="font-medium text">{model.name}</span>
            </p>

            <div className="space-y-3">
              {features.map((feature) => (
                <div key={feature.name} className="space-y-1">
                  <label className="text-xs font-medium text-text-secondary">
                    {feature.label}
                  </label>
                  {feature.type === "select" ? (
                    <select
                      value={inputs[feature.name] || ""}
                      onChange={(e) =>
                        setInputs({ ...inputs, [feature.name]: e.target.value })
                      }
                      className="input w-full text-sm"
                    >
                      <option value="">Select...</option>
                      {feature.options?.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      placeholder={feature.placeholder}
                      value={inputs[feature.name] || ""}
                      onChange={(e) =>
                        setInputs({ ...inputs, [feature.name]: e.target.value })
                      }
                      className="input w-full text-sm"
                    />
                  )}
                </div>
              ))}
            </div>

            <button
              onClick={handleSubmit}
              disabled={isPredicting || !isValid}
              className="btn btn-primary w-full"
            >
              {isPredicting ? "Predicting..." : "Predict"}
            </button>

            {result && (
              <div className="rounded-lg border border-success/30 bg-success-muted/30 p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-medium uppercase text-text-tertiary">
                      Result
                    </p>
                    <p className="text-lg font-bold text-success">
                      {result.value}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-medium uppercase text-text-tertiary">
                      Confidence
                    </p>
                    <p className="text-lg font-bold text">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function BatchPredictionPanel({
  model,
  isProcessing,
  hasResult,
}: {
  model: ModelInfo | null;
  isProcessing: boolean;
  hasResult: boolean;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
    }
  };

  const handleDownload = () => {
    const csvContent = "data:text/csv;charset=utf-8," + "id,prediction,confidence\n1,Positive,0.89\n2,Negative,0.72\n3,Positive,0.91";
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "predictions.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card flex flex-col overflow-hidden">
      <div className="border-b border-border-subtle px-5 py-4">
        <h2 className="text-base font-semibold text">Batch Prediction</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {!model ? (
          <p className="text-sm text-text-tertiary">Select a model to make predictions.</p>
        ) : (
          <>
            <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-border p-8 text-center">
              <svg
                className="h-10 w-10 text-text-tertiary"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="mt-2 text-sm text-text-secondary">
                Upload CSV for batch predictions
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="btn btn-secondary mt-3 text-sm"
                disabled={isProcessing}
              >
                Choose File
              </button>
            </div>

            {fileName && (
              <div className="flex items-center justify-between rounded-lg border border-border-subtle bg-surface-2/30 p-3">
                <div className="flex items-center gap-3">
                  <svg
                    className="h-5 w-5 text-success"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                  <span className="text-sm text-text-secondary">{fileName}</span>
                </div>
                <button
                  onClick={handleDownload}
                  disabled={!hasResult || isProcessing}
                  className="btn btn-ghost text-sm"
                >
                  {isProcessing ? "Processing..." : "Download"}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function EndpointPanel({ modelId }: { modelId: string | null }) {
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);
  const endpoint = `https://api.example.com/v1/predict/${modelId || "model-id"}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(endpoint);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="card p-5 space-y-4">
      <h3 className="text-sm font-semibold text">Endpoint</h3>

      <div className="flex items-center gap-2 rounded-lg border border-border-subtle bg-surface-2/30 p-3">
        <code className="flex-1 truncate text-xs font-mono text-text-secondary">
          {endpoint}
        </code>
        <button onClick={handleCopy} className="btn btn-ghost p-1.5">
          {copied ? (
            <svg
              className="h-4 w-4 text-success"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          ) : (
            <svg
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          )}
        </button>
      </div>

      <button
        onClick={() => navigate("/automl-lab")}
        className="btn btn-secondary w-full text-sm"
      >
        <svg
          className="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.001 0 01-15.357-2m15.357 2H15"
          />
        </svg>
        Retrain
      </button>
    </div>
  );
}

export function MLPredictionPage() {
  const [selectedModelId, setSelectedModelId] = useState<string | null>("m-1");
  const [activeTab, setActiveTab] = useState<"single" | "batch">("single");
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

  const selectedModel = mockModels.find((m) => m.id === selectedModelId);

  const handlePredict = (_inputs: Record<string, string>) => {
    setIsPredicting(true);
    setPredictionResult(null);
    setTimeout(() => {
      setPredictionResult({
        value: Math.random() > 0.5 ? "Positive" : "Negative",
        confidence: 0.75 + Math.random() * 0.2,
      });
      setIsPredicting(false);
    }, 1000);
  };

  return (
    <section className="animate-in fade-in duration-300">
      <PageHeader
        title="ML Prediction"
        subtitle="Make predictions with your trained models."
        actions={
          <div className="flex items-center gap-2">
            <button className="btn btn-secondary">
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              Export
            </button>
          </div>
        }
      />

      <div className="grid gap-6 lg:grid-cols-12">
        <div className="space-y-6 lg:col-span-4">
          <ModelSelector
            models={mockModels}
            selectedId={selectedModelId}
            onSelect={setSelectedModelId}
          />

          {selectedModel && <EndpointPanel modelId={selectedModelId} />}
        </div>

        <div className="space-y-6 lg:col-span-8">
          <div className="card flex flex-col overflow-hidden">
            <div className="border-b border-border-subtle px-5 py-4">
              <div className="flex rounded-lg border border-border p-1">
                <button
                  onClick={() => setActiveTab("single")}
                  className={cn(
                    "flex-1 rounded-md px-4 py-2 text-sm font-medium transition-all",
                    activeTab === "single"
                      ? "bg-accent text-white shadow-sm"
                      : "text-text-secondary hover:bg-surface-2"
                  )}
                >
                  Single
                </button>
                <button
                  onClick={() => setActiveTab("batch")}
                  className={cn(
                    "flex-1 rounded-md px-4 py-2 text-sm font-medium transition-all",
                    activeTab === "batch"
                      ? "bg-accent text-white shadow-sm"
                      : "text-text-secondary hover:bg-surface-2"
                  )}
                >
                  Batch
                </button>
              </div>
            </div>

            <div className="p-5">
              {activeTab === "single" ? (
                <SinglePredictionPanel
                  model={selectedModel || null}
                  onPredict={handlePredict}
                  isPredicting={isPredicting}
                  result={predictionResult}
                />
              ) : (
                <BatchPredictionPanel
                  model={selectedModel || null}
                  isProcessing={false}
                  hasResult={false}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}