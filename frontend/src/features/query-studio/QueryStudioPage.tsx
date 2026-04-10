import { useState, useRef, useEffect } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { StatusPill } from "../../components/shared/StatusPill";
import { actionAdvices, askDataConversation, datasets, workspace } from "../../lib/mocks/data";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: string;
  codeBlock?: string;
  citations?: string[];
}

interface QueryTemplate {
  id: string;
  title: string;
  query: string;
  category: string;
}

interface HistoryItem {
  id: string;
  query: string;
  timestamp: string;
  starred: boolean;
}

const queryTemplates: QueryTemplate[] = [
  { id: "t1", title: "Churn Analysis", query: "Why did churn increase in the last quarter?", category: "analysis" },
  { id: "t2", title: "Trend Detection", query: "Show me the top 5 declining metrics this month", category: "analysis" },
  { id: "t3", title: "Correlation Find", query: "What factors correlate most with customer satisfaction?", category: "discovery" },
  { id: "t4", title: "Anomaly Detection", query: "Are there any unusual patterns in the data?", category: "anomaly" },
  { id: "t5", title: "Segment Analysis", query: "Compare performance across customer segments", category: "comparison" },
  { id: "t6", title: "Prediction Ask", query: "What will our revenue be next month?", category: "prediction" },
];

const queryHistory: HistoryItem[] = [
  { id: "h1", query: "Why did churn increase in Region A last quarter?", timestamp: "10:02 AM", starred: true },
  { id: "h2", query: "What should we do first to reduce risk this month?", timestamp: "10:03 AM", starred: false },
  { id: "h3", query: "Show me the top performing segments", timestamp: "Yesterday", starred: true },
  { id: "h4", query: "Compare Q3 vs Q4 revenue trends", timestamp: "Yesterday", starred: false },
];

export function QueryStudioPage() {
  const activeDataset = datasets.find((item) => item.id === workspace.activeDatasetId) ?? datasets[0];
  const [messages, setMessages] = useState<Message[]>(askDataConversation as Message[]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [activeTab, setActiveTab] = useState<"chat" | "templates" | "history">("chat");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!inputValue.trim()) return;
    
    const newUserMessage: Message = {
      id: `msg-${Date.now()}`,
      role: "user",
      text: inputValue,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    
    setMessages(prev => [...prev, newUserMessage]);
    setInputValue("");
    setIsTyping(true);
    
    setTimeout(() => {
      const assistantResponse: Message = {
        id: `msg-${Date.now() + 1}`,
        role: "assistant",
        text: "Based on the analysis, I can see several key factors contributing to this trend. The data shows strong correlation with support ticket volume and recent feature changes.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        codeBlock: "SELECT region, churn_rate, ticket_volume FROM analytics WHERE date > '2024-01-01'",
        citations: ["Customer_Churn_2024.csv", "Support_Tickets_2024.csv"],
      };
      setMessages(prev => [...prev, assistantResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const handleTemplateClick = (template: QueryTemplate) => {
    setInputValue(template.query);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const CodeBlock = ({ code }: { code: string }) => (
    <div className="relative group mt-3">
      <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button className="btn btn-ghost text-xs">Copy</button>
      </div>
      <pre className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg p-4 text-sm text-slate-100 overflow-x-auto border border-slate-700/50 shadow-lg">
        <code>{code}</code>
      </pre>
    </div>
  );

  return (
    <div className="flex h-[calc(100vh-10rem)] flex-col overflow-hidden">
      <PageHeader
        title="Query Studio"
        subtitle={`Ask questions and get concrete recommendations scoped to ${activeDataset.name}.`}
        actions={
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-2 border text-xs">
              <svg className="w-4 h-4 text-tertiary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79 8-4" />
              </svg>
              <span className="text">{activeDataset.name}</span>
            </div>
            <button className="btn btn-secondary">
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                </svg>
                Save Context
              </span>
            </button>
            <button className="btn btn-primary">
              <span className="flex items-center gap-2">
                Share
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
              </span>
            </button>
          </div>
        }
      />

      <div className="flex flex-1 gap-6 overflow-hidden">
        <div className="flex flex-1 flex-col overflow-hidden rounded-2xl border bg-surface shadow-xl ring-1 ring-black/5">
          <div className="flex items-center justify-between border-b border-subtle bg-gradient-to-r from-surface to-surface-2/30 px-5 py-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary to-slate-700 flex items-center justify-center shadow-md">
                  <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h2 className="text-sm font-semibold text">Query Assistant</h2>
              </div>
              <StatusPill label="Grounded Mode" tone="success" />
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200/50">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-xs font-medium text-emerald-700">Live</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="btn btn-ghost">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </button>
              <button className="btn btn-ghost">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </button>
              <div className="w-px h-5 bg-border mx-1"></div>
              <button className="btn btn-ghost text-xs">Clear Chat</button>
            </div>
          </div>

          <div className="flex items-center gap-1 border-b border-subtle px-4 bg-surface-2/20">
            <button
              onClick={() => setActiveTab("chat")}
              className={`px-4 py-3 text-sm font-medium transition-all relative ${
                activeTab === "chat" ? "text" : "text-secondary hover:text"
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                Chat
              </span>
              {activeTab === "chat" && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-full"></span>
              )}
            </button>
            <button
              onClick={() => setActiveTab("templates")}
              className={`px-4 py-3 text-sm font-medium transition-all relative ${
                activeTab === "templates" ? "text" : "text-secondary hover:text"
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
                Templates
              </span>
              {activeTab === "templates" && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-full"></span>
              )}
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`px-4 py-3 text-sm font-medium transition-all relative ${
                activeTab === "history" ? "text" : "text-secondary hover:text"
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                History
              </span>
              {activeTab === "history" && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-full"></span>
              )}
            </button>
          </div>

          <div className="flex-1 overflow-hidden flex flex-col">
            {activeTab === "chat" && (
              <>
                <div className="flex-1 overflow-y-auto p-5 space-y-6">
                  {messages.map((msg, idx) => {
                    const isUser = msg.role === "user";
                    const showAvatar = idx === 0 || messages[idx - 1].role !== msg.role;
                    
                    return (
                      <div key={msg.id} className={`flex gap-4 ${isUser ? "flex-row-reverse" : ""} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                        {showAvatar && (
                          <div className={`w-8 h-8 rounded-xl flex-shrink-0 flex items-center justify-center shadow-md ${
                            isUser 
                              ? "bg-gradient-to-br from-primary to-slate-700" 
                              : "bg-gradient-to-br from-emerald-500 to-teal-600"
                          }`}>
                            {isUser ? (
                              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                              </svg>
                            ) : (
                              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                              </svg>
                            )}
                          </div>
                        )}
                        {!showAvatar && <div className="w-8 flex-shrink-0" />}
                        <div className={`max-w-[75%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
                          <div className={`relative rounded-2xl px-4 py-3 shadow-sm ${
                            isUser 
                              ? "bg-gradient-to-r from-primary to-slate-800 text-white rounded-br-md" 
                              : "bg-white border border-subtle rounded-bl-md text"
                          }`}>
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                            {msg.codeBlock && <CodeBlock code={msg.codeBlock} />}
                            {msg.citations && msg.citations.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-subtle">
                                <p className="text-[11px] font-medium text-secondary mb-2">Sources</p>
                                <div className="flex flex-wrap gap-1.5">
                                  {msg.citations.map((cite, i) => (
                                    <span key={i} className="px-2 py-0.5 rounded-md bg-surface-2 text-xs text-secondary">
                                      {cite}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                          <span className={`mt-1.5 text-[11px] font-medium uppercase tracking-wider ${
                            isUser ? "text-secondary" : "text-tertiary"
                          }`}>
                            {msg.timestamp}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                  
                  {isTyping && (
                    <div className="flex gap-4 animate-in fade-in duration-300">
                      <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-md">
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <div className="bg-white border border-subtle rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                        <div className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full bg-tertiary animate-bounce" style={{ animationDelay: "0ms" }}></span>
                          <span className="w-2 h-2 rounded-full bg-tertiary animate-bounce" style={{ animationDelay: "150ms" }}></span>
                          <span className="w-2 h-2 rounded-full bg-tertiary animate-bounce" style={{ animationDelay: "300ms" }}></span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                <div className="border-t border-subtle bg-gradient-to-r from-surface-2/30 to-surface p-4">
                  <div className="relative flex items-end gap-3 rounded-2xl border border-subtle bg-white p-2 shadow-lg focus-within:border-primary/40 focus-within:ring-4 focus-within:ring-primary/5 transition-all">
                    <div className="flex-1">
                      <textarea
                        ref={inputRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="input max-h-40 min-h-[52px]"
                        placeholder="Ask about your data (e.g. why did churn increase)..."
                        rows={1}
                      />
                    </div>
                    <div className="flex items-center gap-2 pr-2">
                      <button className="btn btn-ghost">
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                        </svg>
                      </button>
                      <button
                        onClick={handleSend}
                        disabled={!inputValue.trim()}
                        className="btn btn-primary"
                      >
                        <span>Send</span>
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <p className="mt-2 text-center text-[11px] text-secondary">
                    AI may produce inaccurate information. Always verify results.
                  </p>
                </div>
              </>
            )}

            {activeTab === "templates" && (
              <div className="flex-1 overflow-y-auto p-5">
                <div className="mb-6">
                  <h3 className="text-sm font-semibold text mb-2">Quick Start Templates</h3>
                  <p className="text-xs text-secondary">Click any template to start a new query</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {queryTemplates.map((template) => (
                    <button
                      key={template.id}
                      onClick={() => handleTemplateClick(template)}
                      className="card group flex items-start gap-3 p-4 text-left hover:border-primary/30"
                    >
                      <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-surface-2 to-surface-3 flex items-center justify-center flex-shrink-0 group-hover:from-primary/10 group-hover:to-primary/5 transition-colors">
                        <svg className="w-4 h-4 text-secondary group-hover:text-primary transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      </div>
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-medium text group-hover:text-primary transition-colors block">
                          {template.title}
                        </span>
                        <span className="text-xs text-secondary mt-1 line-clamp-2 block">
                          {template.query}
                        </span>
                        <span className="inline-flex mt-2 px-2 py-0.5 rounded-md bg-surface-2 text-[10px] font-medium text-secondary uppercase tracking-wider">
                          {template.category}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {activeTab === "history" && (
              <div className="flex-1 overflow-y-auto p-5">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-sm font-semibold text mb-1">Query History</h3>
                    <p className="text-xs text-secondary">Your recent queries and favorites</p>
                  </div>
                  <button className="btn btn-ghost text-xs">View All</button>
                </div>
                <div className="space-y-2">
                  {queryHistory.map((item) => (
                    <div
                      key={item.id}
                      className="group flex items-center gap-3 p-3 rounded-lg hover:bg-surface-2/50 transition-colors cursor-pointer"
                    >
                      <button className="flex-shrink-0 text-secondary hover:text-amber-500 transition-colors">
                        <svg className={`w-4 h-4 ${item.starred ? "text-amber-500 fill-amber-500" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                        </svg>
                      </button>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text truncate group-hover:text-primary transition-colors">
                          {item.query}
                        </p>
                      </div>
                      <span className="text-xs text-secondary flex-shrink-0">{item.timestamp}</span>
                      <button className="btn btn-ghost p-1">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
                <div className="mt-8 p-4 rounded-xl bg-gradient-to-r from-primary/5 to-emerald-500/5 border border-primary/10">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-sm font-medium text">Pro Tip</p>
                      <p className="text-xs text-secondary mt-1">
                        Star your favorite queries to quickly access them later. Use keyboard shortcut <kbd className="px-1.5 py-0.5 rounded bg-surface-2 text-xs">↑</kbd> <kbd className="px-1.5 py-0.5 rounded bg-surface-2 text-xs">↑</kbd> to search history.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="hidden w-96 flex-col gap-5 overflow-y-auto xl:flex">
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text">Execution Meta</h3>
              <span className="w-2 h-2 rounded-full bg-success animate-pulse"></span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-gradient-to-br from-surface-2 to-surface-3 border border-subtle">
                <p className="text-[11px] font-medium text-secondary uppercase tracking-wider">Rows Scanned</p>
                <p className="mt-1.5 text-xl font-bold text tabular-nums">{activeDataset.rows.toLocaleString()}</p>
              </div>
              <div className="p-3 rounded-xl bg-gradient-to-br from-surface-2 to-surface-3 border border-subtle">
                <p className="text-[11px] font-medium text-secondary uppercase tracking-wider">Query Time</p>
                <p className="mt-1.5 text-xl font-bold text tabular-nums">1.2s</p>
              </div>
              <div className="p-3 rounded-xl bg-gradient-to-br from-surface-2 to-surface-3 border border-subtle">
                <p className="text-[11px] font-medium text-secondary uppercase tracking-wider">Confidence</p>
                <p className="mt-1.5 text-xl font-bold text-success tabular-nums">94%</p>
              </div>
              <div className="p-3 rounded-xl bg-gradient-to-br from-surface-2 to-surface-3 border border-subtle">
                <p className="text-[11px] font-medium text-secondary uppercase tracking-wider">Cost</p>
                <p className="mt-1.5 text-xl font-bold text tabular-nums">$0.02</p>
              </div>
            </div>
          </div>

          <div className="card flex-1 p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text">Actionable Advice</h3>
              <span className="text-[10px] font-medium text-secondary uppercase tracking-wider bg-surface-2 px-2 py-1 rounded-md">
                3 items
              </span>
            </div>
            <div className="space-y-3">
              {actionAdvices.map((advice, idx) => (
                <div 
                  key={advice.id} 
                  className="group relative rounded-xl border border-subtle bg-gradient-to-br from-surface-2/50 to-white p-4 shadow-sm hover:shadow-md hover:border-primary/20 transition-all cursor-pointer"
                  style={{ animationDelay: `${idx * 100}ms` }}
                >
                  <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="btn btn-primary p-1.5">
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </button>
                  </div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium text">{advice.title}</span>
                    <StatusPill 
                      label={advice.impact} 
                      tone={advice.impact === "high" ? "danger" : advice.impact === "medium" ? "warning" : "info"} 
                    />
                  </div>
                  <p className="text-xs leading-relaxed text-secondary">{advice.why}</p>
                  <div className="mt-3 pt-3 border-t border-subtle">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-secondary mb-1">Next Step</p>
                    <p className="text-xs font-medium text-primary leading-relaxed">{advice.nextStep}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-primary/5 to-transparent p-5">
            <h3 className="text-sm font-semibold text mb-3">Active Dataset</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-secondary">Name</span>
                <span className="font-medium text truncate max-w-[150px]">{activeDataset.name}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-secondary">Owner</span>
                <span className="font-medium text">{activeDataset.owner}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-secondary">Quality Score</span>
                <div className="flex items-center gap-1.5">
                  <div className="w-16 h-1.5 rounded-full bg-surface-2">
                    <div 
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-400" 
                      style={{ width: `${activeDataset.qualityScore}%` }}
                    ></div>
                  </div>
                  <span className="font-medium text">{activeDataset.qualityScore}%</span>
                </div>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-secondary">Freshness</span>
                <span className="inline-flex items-center gap-1 font-medium text-success">
                  <span className="w-1.5 h-1.5 rounded-full bg-success"></span>
                  {activeDataset.freshness}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}