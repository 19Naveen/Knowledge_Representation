import { FormEvent, useState } from "react";
import { useAuthContext } from "../../lib/context/AuthContext";
import { cn } from "../../lib/cn";

type AuthMode = "login" | "signup";

interface FormState {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export function AuthPage() {
  const { loginWithCredentials, signupWithCredentials } = useAuthContext();
  const [mode, setMode] = useState<AuthMode>("login");
  const [formState, setFormState] = useState<FormState>({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorMessage(null);

    if (mode === "signup" && formState.password !== formState.confirmPassword) {
      setErrorMessage("Passwords do not match.");
      return;
    }

    setIsSubmitting(true);

    try {
      if (mode === "login") {
        await loginWithCredentials({
          email: formState.email,
          password: formState.password,
        });
      } else {
        await signupWithCredentials({
          username: formState.username,
          email: formState.email,
          password: formState.password,
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to authenticate.";
      setErrorMessage(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-bg text-text">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,hsl(var(--accent)/0.09),transparent_48%),radial-gradient(circle_at_bottom_left,hsl(var(--primary)/0.06),transparent_50%)]" />

      <div className="relative z-10 flex min-h-screen items-center justify-center p-6 sm:p-8">
        <section className="card w-full max-w-md rounded-2xl border border-border/80 p-6 shadow-soft-lg sm:p-8">
          <div className="mb-6">
            <p className="section-label mb-2">Knowledge Representation Platform</p>
            <h1 className="text-2xl font-semibold tracking-tight text-text">
              {mode === "login" ? "Welcome back" : "Create your account"}
            </h1>
            <p className="mt-2 text-sm text-text-secondary">
              Use username and password today. Backend is prepared for OAuth provider integration later.
            </p>
          </div>

          <div className="mb-5 inline-flex rounded-xl bg-surface-2 p-1">
            <button
              type="button"
              className={cn(
                "rounded-lg px-4 py-2 text-sm font-medium transition-all duration-150",
                mode === "login" ? "bg-surface text-text shadow-soft-sm" : "text-text-secondary hover:text-text"
              )}
              onClick={() => setMode("login")}
            >
              Login
            </button>
            <button
              type="button"
              className={cn(
                "rounded-lg px-4 py-2 text-sm font-medium transition-all duration-150",
                mode === "signup" ? "bg-surface text-text shadow-soft-sm" : "text-text-secondary hover:text-text"
              )}
              onClick={() => setMode("signup")}
            >
              Sign up
            </button>
          </div>

          <form className="space-y-4" onSubmit={onSubmit}>
            {mode === "signup" && (
              <label className="block space-y-2">
                <span className="text-xs font-medium uppercase tracking-wider text-text-tertiary">Username</span>
                <input
                  className="input"
                  type="text"
                  value={formState.username}
                  onChange={(event) => setFormState((prev) => ({ ...prev, username: event.target.value }))}
                  placeholder="your_username"
                  autoComplete="username"
                  minLength={3}
                  required
                />
              </label>
            )}

            <label className="block space-y-2">
              <span className="text-xs font-medium uppercase tracking-wider text-text-tertiary">Email</span>
              <input
                className="input"
                type="email"
                value={formState.email}
                onChange={(event) => setFormState((prev) => ({ ...prev, email: event.target.value }))}
                placeholder="name@company.com"
                autoComplete="email"
                required
              />
            </label>

            <label className="block space-y-2">
              <span className="text-xs font-medium uppercase tracking-wider text-text-tertiary">Password</span>
              <input
                className="input"
                type="password"
                value={formState.password}
                onChange={(event) => setFormState((prev) => ({ ...prev, password: event.target.value }))}
                placeholder="At least 8 characters"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                minLength={8}
                required
              />
            </label>

            {mode === "signup" && (
              <label className="block space-y-2">
                <span className="text-xs font-medium uppercase tracking-wider text-text-tertiary">Confirm password</span>
                <input
                  className="input"
                  type="password"
                  value={formState.confirmPassword}
                  onChange={(event) =>
                    setFormState((prev) => ({ ...prev, confirmPassword: event.target.value }))
                  }
                  placeholder="Re-enter your password"
                  autoComplete="new-password"
                  minLength={8}
                  required
                />
              </label>
            )}

            {errorMessage && (
              <div className="rounded-lg border border-danger/30 bg-danger-muted px-3 py-2 text-xs text-danger">
                {errorMessage}
              </div>
            )}

            <button className="btn btn-primary w-full" type="submit" disabled={isSubmitting}>
              {isSubmitting
                ? mode === "login"
                  ? "Signing in..."
                  : "Creating account..."
                : mode === "login"
                  ? "Login"
                  : "Sign up"}
            </button>
          </form>

          <div className="mt-5 rounded-lg border border-border-subtle bg-surface-2/50 px-3 py-2">
            <p className="text-2xs text-text-secondary">
              OAuth sign-in buttons are intentionally hidden for now and will be enabled after provider setup.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
