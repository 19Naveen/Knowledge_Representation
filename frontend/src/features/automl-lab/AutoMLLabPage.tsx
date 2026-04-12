import { useState, useCallback } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useAppContext } from "../../lib/context/AppContext";

export function AutoMLLabPage() {
    const { activeDataset } = useAppContext();
    const [isScanning, setIsScanning] = useState(false);
    const [results, setResults] = useState<any[] | null>(null);

    const startScan = useCallback(() => {
        setIsScanning(true);
        setTimeout(() => {
            setResults([
                { id: 1, name: "Gradient Booster (Optimal)", score: 0.942, latency: "12ms", complexity: "High" },
                { id: 2, name: "Decision Forest", score: 0.915, latency: "8ms", complexity: "Medium" },
                { id: 3, name: "Linear Ensemble", score: 0.884, latency: "2ms", complexity: "Low" },
            ]);
            setIsScanning(false);
        }, 2500);
    }, []);

    return (
        <div className="flex flex-col gap-6 animate-in fade-in duration-500">
            <PageHeader
                title="AutoML Lab"
                subtitle="Autonomous model discovery using neural architecture search and hyperparameter optimization."
            />

            <div className="p-6 pt-0">
                {!results && !isScanning && (
                    <div className="card p-12 text-center flex flex-col items-center gap-6 max-w-2xl mx-auto mt-12 bg-primary text-white shadow-2xl">
                        <div className="size-20 rounded-3xl bg-white/10 flex items-center justify-center border border-white/20">
                            <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.675.337a4 4 0 01-2.574.345l-2.313-.463c-.574-.115-1.155.032-1.536.413l-1.119 1.119a2 2 0 01-2.828 0l-3.536-3.536a2 2 0 010-2.828l1.119-1.119a2 2 0 00.413-1.536L4.057 6.42a4 4 0 01.345-2.574l.338-.675a6 6 0 00.517-3.861L4.78 1.056a2 2 0 00-.547-1.022L2.73 2.73a2 2 0 000 2.828l3.536 3.536a2 2 0 002.828 0L9.11 9.11a2 2 0 011.536-.413l2.313.463a4 4 0 002.574-.345l.675-.338a6 6 0 013.861-.517l2.387.477a2 2 0 011.022.547l1.503 1.503z" /></svg>
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-3xl font-black tracking-tighter">Ready to Automate?</h3>
                            <p className="opacity-70 text-sm max-w-sm mx-auto">We will automatically analyze <span className="font-bold">"{activeDataset?.name}"</span> to find the highest-performing model architecture.</p>
                        </div>
                        <button onClick={startScan} className="bg-white text-primary font-bold px-12 py-4 rounded-2xl hover:scale-105 active:scale-95 transition-all shadow-xl">
                            Scan Global Architectures
                        </button>
                    </div>
                )}

                {isScanning && (
                    <div className="flex flex-col items-center justify-center py-20 animate-in fade-in zoom-in duration-500 text-center">
                        <div className="w-24 h-24 rounded-full border-4 border-primary/10 border-t-primary animate-spin" />
                        <h3 className="mt-8 text-2xl font-bold tracking-tight">Neural Search Active</h3>
                        <p className="mt-2 text-text-secondary">Testing 4,200+ model variations against your feature vectors...</p>
                    </div>
                )}

                {results && (
                    <div className="max-w-4xl mx-auto space-y-8 animate-in slide-up duration-500">
                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="text-2xl font-black tracking-tight text">Discovery Results</h3>
                                <p className="text-sm text-text-secondary">Optimal architectures for predictive accuracy.</p>
                            </div>
                            <button onClick={() => setResults(null)} className="btn btn-secondary text-xs">Reset Search</button>
                        </div>

                        <div className="grid grid-cols-1 gap-4">
                            {results.map(r => (
                                <div key={r.id} className="card p-6 flex flex-col md:flex-row items-center justify-between gap-6 hover:border-primary/50 hover:shadow-2xl hover:shadow-primary/5 transition-all">
                                    <div className="flex items-center gap-6">
                                        <div className="size-14 rounded-2xl bg-surface-2 border border-border flex items-center justify-center font-black text-primary">
                                            #{r.id}
                                        </div>
                                        <div>
                                            <p className="text-lg font-bold tracking-tight">{r.name}</p>
                                            <p className="text-xs text-text-tertiary">Complexity: <span className="text-text font-bold">{r.complexity}</span> | Latency: <span className="text-text font-bold">{r.latency}</span></p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-8">
                                        <div className="text-right">
                                            <p className="text-[10px] font-bold uppercase tracking-widest text-text-tertiary">Performance Score</p>
                                            <p className="text-3xl font-black tracking-tighter text-primary">{r.score}</p>
                                        </div>
                                        <button className="btn btn-primary px-6">Deploy Model</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
