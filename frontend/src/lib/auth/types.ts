export type OAuthProvider = "google" | "github" | "microsoft";

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  authProvider: "local" | OAuthProvider;
  createdAt: string;
}

export interface AuthSession {
  accessToken: string;
  tokenType: "bearer";
  expiresIn: number;
  user: AuthUser;
}

export interface LoginInput {
  email: string;
  password: string;
}

export interface SignupInput {
  username: string;
  email: string;
  password: string;
}

export interface AuthContextValue {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: AuthUser | null;
  session: AuthSession | null;
  loginWithCredentials: (input: LoginInput) => Promise<void>;
  signupWithCredentials: (input: SignupInput) => Promise<void>;
  logout: () => void;
}
