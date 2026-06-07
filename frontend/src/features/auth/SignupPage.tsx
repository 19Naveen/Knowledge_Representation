import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AuthCardLayout } from "../../components/shared/AuthCardLayout";
import { useAuthContext } from "../../lib/context/AuthContext";

export function SignupPage() {
  const { signupWithCredentials } = useAuthContext();
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsLoading(true);
    try {
      await signupWithCredentials({ username, email, password });
      navigate("/onboarding", { replace: true });
    } catch (err: any) {
      setError(err.message ?? "Sign up failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <AuthCardLayout
      title="Create your account"
      subtitle="Start building your intelligent data workspace"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-semibold text-text-secondary mb-1.5 uppercase tracking-wider">
            Username
          </label>
          <input
            type="text"
            className="input w-full"
            placeholder="yourname"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoFocus
          />
        </div>

        <div>
          <label className="block text-xs font-semibold text-text-secondary mb-1.5 uppercase tracking-wider">
            Email
          </label>
          <input
            type="email"
            className="input w-full"
            placeholder="you@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="block text-xs font-semibold text-text-secondary mb-1.5 uppercase tracking-wider">
            Password
          </label>
          <input
            type="password"
            className="input w-full"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
          />
        </div>

        {error && (
          <p className="text-sm text-danger bg-danger-muted border border-danger/20 rounded-lg px-4 py-3">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={isLoading}
          className="btn btn-primary w-full py-2.5 mt-2 disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {isLoading ? "Creating account…" : "Create Account"}
        </button>
      </form>

      <p className="mt-6 text-center text-sm text-text-secondary">
        Already have an account?{" "}
        <Link to="/signin" className="font-semibold text-accent hover:underline">
          Sign in
        </Link>
      </p>
    </AuthCardLayout>
  );
}
