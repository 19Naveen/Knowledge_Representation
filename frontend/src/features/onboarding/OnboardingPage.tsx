import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BrandMark } from "../../components/shared/BrandMark";
import { useWorkspaceContext } from "../../lib/context/WorkspaceContext";

export function OnboardingPage() {
  const { createWorkspace } = useWorkspaceContext();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIsLoading(true);
    try {
      await createWorkspace(name.trim(), description.trim() || undefined);
      navigate("/app", { replace: true });
    } catch {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-bg text-text flex flex-col items-center justify-center px-4 relative overflow-hidden">
      {/* Background radial gradient */}
      <div
        className="pointer-events-none absolute inset-0 opacity-40"
        style={{
          background:
            "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(var(--color-primary-rgb, 99 102 241) / 0.15), transparent)",
        }}
      />
      {/* Subtle grid */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(var(--color-border) 1px, transparent 1px), linear-gradient(90deg, var(--color-border) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />

      <div className="relative z-10 w-full max-w-md">
        {/* Brand */}
        <div className="flex justify-center mb-10">
          <BrandMark size="lg" />
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-3 mb-8">
          {[1, 2, 3].map((step) => (
            <div key={step} className="flex items-center gap-3 flex-1">
              <div
                className={`size-7 rounded-full flex items-center justify-center text-[10px] font-black shrink-0 ${
                  step === 2
                    ? "bg-primary text-white shadow-lg shadow-primary/30"
                    : step < 2
                    ? "bg-success/20 text-success border border-success/30"
                    : "bg-surface-2 text-text-tertiary border border-border"
                }`}
              >
                {step < 2 ? (
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  step
                )}
              </div>
              {step < 3 && <div className="flex-1 h-px bg-border" />}
            </div>
          ))}
        </div>

        <div className="card p-8 shadow-2xl shadow-black/20 border border-border-subtle backdrop-blur-xl">
          <div className="mb-6">
            <h1 className="text-2xl font-black tracking-tight">Create your workspace</h1>
            <p className="text-sm text-text-secondary mt-1">
              A workspace is where your team collaborates on data and models.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-xs font-semibold text-text-secondary mb-1.5 uppercase tracking-wider">
                Workspace Name <span className="text-danger">*</span>
              </label>
              <input
                type="text"
                className="input w-full"
                placeholder="Acme Data Team"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                autoFocus
                maxLength={64}
              />
              <p className="text-[11px] text-text-tertiary mt-1.5">
                You can rename this later in Workspace Settings.
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold text-text-secondary mb-1.5 uppercase tracking-wider">
                Description{" "}
                <span className="text-text-tertiary font-normal normal-case tracking-normal">
                  (optional)
                </span>
              </label>
              <textarea
                className="input w-full resize-none"
                rows={3}
                placeholder="What does your team work on?"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                maxLength={256}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !name.trim()}
              className="btn btn-primary w-full py-3 text-sm font-bold mt-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Creating…" : "Create Workspace & Enter App →"}
            </button>
          </form>
        </div>

        <p className="mt-6 text-center text-xs text-text-tertiary">
          Step 2 of 3 — You can create more workspaces anytime.
        </p>
      </div>
    </div>
  );
}
