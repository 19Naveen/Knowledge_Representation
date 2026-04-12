import { useState, useRef, useEffect, useCallback } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useAppContext } from "../../lib/context/AppContext";
import { cn } from "../../lib/cn";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  tokens?: number;
  dataPoint?: {
    label: string;
    value: string;
    trend?: 'up' | 'down';
  }[];
  code?: string;
}

const INITIAL_SUGGESTIONS = [
  "Compute regional churn delta for Q3",
  "Identify top 5 high-value cohorts at risk",
  "Summarize anomaly vectors in balance distribution",
  "Generate SQL for active account segmentation"
];

export function QueryStudioPage() {
  const { activeDataset } = useAppContext();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: `System online. I have analyzed **${activeDataset?.name || "the active dataset"}** and am ready for deep analytical queries. How can I assist with your data exploration today?`,
      timestamp: "10:00 AM"
    }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSend = useCallback((text: string) => {
    if (!text.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    // Simulated LLM logic
    setTimeout(() => {
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `I've analyzed the statistical distribution for **${activeDataset?.name || "your data"}**. Based on the query, I detected a significant correlation between current balance and recent transaction frequency. Here is the technical breakdown:`,
        dataPoint: [
          { label: "Correlation Coefficient", value: "0.84", trend: 'up' },
          { label: "Anomalous Clusters", value: "12", trend: 'down' },
          { label: "Confidence Interval", value: "95.2%", trend: 'up' }
        ],
        code: `SELECT region, AVG(balance) as avg_bal 
FROM data_source 
GROUP BY region 
HAVING AVG(balance) > 10000 
ORDER BY avg_bal DESC;`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        tokens: 142
      };
      setMessages(prev => [...prev, assistantMsg]);
      setIsTyping(false);
    }, 1500);
  }, [activeDataset]);

  return (
    <div className="flex flex-col h-[calc(100vh-var(--header-h))] overflow-hidden animate-in fade-in duration-500">
      <PageHeader
        title="AI Query Studio"
        subtitle="Natural language interface to your analytical pipelines. Query, analyze, and extract insights using LLM-backed intelligence."
      />

      <div className="flex-1 flex flex-col min-h-0 bg-surface-2/20">

        {/* Messages Space */}
        <div className="flex-1 overflow-y-auto p-8 space-y-8">
          <div className="max-w-4xl mx-auto space-y-8 pb-32">
            {messages.map((msg) => (
              <div key={msg.id} className={cn(
                "flex flex-col gap-4 animate-in slide-up duration-500",
                msg.role === 'user' ? "items-end" : "items-start"
              )}>
                <div className={cn(
                  "max-w-[85%] rounded-2xl p-5 text-sm leading-relaxed shadow-sm",
                  msg.role === 'user'
                    ? "bg-primary text-white font-medium"
                    : "bg-surface border border-border text"
                )}>
                  {/* Content Header for Assistant */}
                  {msg.role === 'assistant' && (
                    <div className="flex items-center gap-2 mb-3 border-b border-border pb-2">
                      <div className="size-5 rounded bg-primary text-white flex items-center justify-center text-[10px] font-bold">AI</div>
                      <span className="text-[11px] font-bold uppercase tracking-widest text-text-tertiary">Intelligence Node</span>
                    </div>
                  )}

                  <div className="whitespace-pre-wrap">{msg.content}</div>

                  {/* Assistant Extra Features (Data Points / Code) */}
                  {msg.dataPoint && (
                    <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {msg.dataPoint.map((dp, i) => (
                        <div key={i} className="p-3 rounded-xl bg-surface-2 border border-border-subtle">
                          <p className="text-[10px] font-bold uppercase tracking-widest text-text-tertiary">{dp.label}</p>
                          <div className="flex items-center justify-between mt-1">
                            <span className="text-base font-black tracking-tight tabular-nums">{dp.value}</span>
                            {dp.trend && (
                              <svg className={cn("size-3.5", dp.trend === 'up' ? "text-success" : "text-danger")} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d={dp.trend === 'up' ? "M5 11l7-7m0 0l7 7m-7-7v18" : "M19 13l-7 7m0 0l-7-7m7 7V6"} /></svg>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {msg.code && (
                    <div className="mt-6 rounded-xl bg-primary-muted border border-border overflow-hidden">
                      <div className="px-4 py-2 border-b border-border flex items-center justify-between">
                        <span className="text-[10px] font-bold uppercase tracking-widest text-text-tertiary">Generated SQL</span>
                        <button className="text-[10px] font-bold text-primary hover:underline">Copy Code</button>
                      </div>
                      <pre className="p-4 overflow-x-auto">
                        <code className="text-[13px] font-mono text-primary leading-relaxed">{msg.code}</code>
                      </pre>
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-3 px-2">
                  <span className="text-[10px] font-bold text-text-tertiary uppercase tracking-widest">{msg.timestamp}</span>
                  {msg.tokens && <span className="text-[10px] font-bold text-text-tertiary opacity-40 uppercase tracking-widest">{msg.tokens} tokens</span>}
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex items-center gap-4 animate-in fade-in duration-300">
                <div className="size-8 rounded-full bg-surface-2 border border-border flex items-center justify-center">
                  <div className="flex gap-1">
                    <div className="size-1 bg-text-tertiary rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <div className="size-1 bg-text-tertiary rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <div className="size-1 bg-text-tertiary rounded-full animate-bounce" />
                  </div>
                </div>
                <span className="text-xs font-bold uppercase tracking-widest text-text-tertiary">Agent analyzing datasets...</span>
              </div>
            )}
            <div ref={scrollRef} />
          </div>
        </div>

        {/* Input Bar - Premium Float Design */}
        <div className="absolute bottom-8 inset-x-0 mx-auto max-w-3xl px-6">
          <div className="flex flex-col gap-4">
            {/* Suggestions */}
            {messages.length < 3 && !isTyping && (
              <div className="flex flex-wrap gap-2 justify-center animate-in slide-up duration-500 [animation-delay:200ms]">
                {INITIAL_SUGGESTIONS.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => handleSend(s)}
                    className="px-4 py-2 rounded-full border border-border bg-white/60 backdrop-blur-md text-xs font-semibold hover:border-primary hover:text-primary transition-all shadow-sm"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            <div className="card shadow-2xl shadow-primary/10 border-primary/10 overflow-hidden bg-white/80 backdrop-blur-xl">
              <div className="flex items-center px-6 py-1">
                <input
                  type="text"
                  autoFocus
                  placeholder={activeDataset ? `Ask anything about ${activeDataset.name}...` : "Choose a dataset to begin query..."}
                  className="flex-1 bg-transparent border-none outline-none text-base py-5 placeholder:text-text-tertiary"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSend(input)}
                  disabled={isTyping}
                />
                <button
                  onClick={() => handleSend(input)}
                  disabled={!input.trim() || isTyping}
                  className={cn(
                    "size-10 rounded-xl flex items-center justify-center transition-all",
                    input.trim() ? "bg-primary text-white shadow-lg shadow-primary/20 scale-105" : "bg-surface-2 text-text-tertiary"
                  )}
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" /></svg>
                </button>
              </div>
              <div className="px-6 py-2 bg-surface-2/30 border-t border-border-subtle flex items-center justify-between">
                <div className="flex gap-4">
                  <button className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-text-tertiary hover:text-text transition-colors">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" /></svg>
                    Attach File
                  </button>
                  <button className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-text-tertiary hover:text-text transition-colors">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                    Table Context
                  </button>
                </div>
                <div className="text-[10px] font-bold uppercase tracking-widest text-text-tertiary opacity-40">
                  Proprietary Knowledge Engine v4.2
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}