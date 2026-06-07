export type Dataset = {
  id: string;
  name: string;
  owner: string;
  rows: number;
  columns: number;
  qualityScore: number;
  freshness: "live" | "daily" | "weekly";
  issues: string[];
};

export type InsightCard = {
  id: string;
  title: string;
  detail: string;
  severity: "info" | "warning" | "success";
};

export type ChartCard = {
  id: string;
  title: string;
  type: "line" | "bar" | "scatter" | "heatmap";
  note: string;
};

export type ModelLeaderboardEntry = {
  id: string;
  model: string;
  score: number;
  status: "queued" | "running" | "completed" | "failed";
};

export type DeploymentCard = {
  id: string;
  modelName: string;
  env: "staging" | "production";
  strategy: "shadow" | "canary" | "blue-green";
  latencyMs: number;
  expectedLift: number;
};

export type AskDataMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: string;
};

export type ActionAdvice = {
  id: string;
  title: string;
  impact: "high" | "medium" | "low";
  why: string;
  nextStep: string;
};

export type WorkspaceMember = {
  id: string;
  name: string;
  email: string;
  avatar: string;
  role: "admin" | "editor" | "viewer";
  status: "active" | "pending";
  joinedAt: string;
};

export type WorkspaceContext = {
  id: string;
  name: string;
  slug: string;
  description?: string;
  owner: string;
  activeDatasetId: string;
  members: WorkspaceMember[];
  role: "admin" | "editor" | "viewer";
  createdAt: string;
};
