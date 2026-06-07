import type {
  ActionAdvice,
  AskDataMessage,
  ChartCard,
  Dataset,
  DeploymentCard,
  InsightCard,
  ModelLeaderboardEntry,
  WorkspaceContext
} from "./types";

export const datasets: Dataset[] = [
  {
    id: "ds-churn-2024",
    name: "Customer_Churn_2024.csv",
    owner: "Marketing",
    rows: 184200,
    columns: 42,
    qualityScore: 85,
    freshness: "daily",
    issues: ["120 missing Age values", "High-cardinality ID column"]
  },
  {
    id: "ds-sales-q3",
    name: "Q3_Sales_Data.csv",
    owner: "Revenue Ops",
    rows: 90310,
    columns: 35,
    qualityScore: 91,
    freshness: "weekly",
    issues: ["3 malformed date strings"]
  }
];

export const insights: InsightCard[] = [
  {
    id: "in-1",
    title: "Region A churn is up 40%",
    detail: "Detected from last 12 weeks after campaign change.",
    severity: "warning"
  },
  {
    id: "in-2",
    title: "Support Tickets strongly correlate with churn",
    detail: "Pearson correlation shows significant positive relation.",
    severity: "info"
  },
  {
    id: "in-3",
    title: "Leakage check passed",
    detail: "No feature exceeded configured leakage threshold.",
    severity: "success"
  }
];

export const charts: ChartCard[] = [
  {
    id: "ch-1",
    title: "Monthly Revenue by Region",
    type: "line",
    note: "Steady growth in APAC, softness in NA."
  },
  {
    id: "ch-2",
    title: "Churn by Age Group",
    type: "bar",
    note: "Highest churn concentration in 18-25 segment."
  },
  {
    id: "ch-3",
    title: "Balance vs Tenure",
    type: "scatter",
    note: "High balance + low tenure cluster has elevated risk."
  }
];

export const leaderboard: ModelLeaderboardEntry[] = [
  {
    id: "m-1",
    model: "XGBoost",
    score: 0.942,
    status: "completed"
  },
  {
    id: "m-2",
    model: "Random Forest",
    score: 0.918,
    status: "completed"
  },
  {
    id: "m-3",
    model: "Logistic Regression",
    score: 0.854,
    status: "running"
  }
];

export const deployments: DeploymentCard[] = [
  {
    id: "d-1",
    modelName: "XGBoost Churn v1.2",
    env: "production",
    strategy: "canary",
    latencyMs: 118,
    expectedLift: 3.8
  },
  {
    id: "d-2",
    modelName: "Random Forest Churn v1.0",
    env: "staging",
    strategy: "shadow",
    latencyMs: 133,
    expectedLift: 2.4
  }
];

export const askDataConversation: AskDataMessage[] = [
  {
    id: "msg-1",
    role: "user",
    text: "Why did churn increase in Region A last quarter?",
    timestamp: "10:02"
  },
  {
    id: "msg-2",
    role: "assistant",
    text: "Churn rose 40% after response time crossed 18 hours for premium users. Support ticket backlog appears to be the strongest driver.",
    timestamp: "10:02"
  },
  {
    id: "msg-3",
    role: "user",
    text: "What should we do first to reduce risk this month?",
    timestamp: "10:03"
  },
  {
    id: "msg-4",
    role: "assistant",
    text: "Prioritize fast-lane support for accounts with more than 5 recent tickets and trigger a retention playbook in week 1.",
    timestamp: "10:03"
  }
];

export const actionAdvices: ActionAdvice[] = [
  {
    id: "adv-1",
    title: "Create high-ticket retention queue",
    impact: "high",
    why: "Accounts with 5+ tickets contribute disproportionately to churn.",
    nextStep: "Route these accounts to a 4-hour SLA and assign senior support owners."
  },
  {
    id: "adv-2",
    title: "Offer targeted discount at risk threshold",
    impact: "medium",
    why: "Simulations show churn probability can drop from 78% to 42% at 15% discount.",
    nextStep: "Trigger offer only for users with high model risk and low engagement score."
  },
  {
    id: "adv-3",
    title: "Weekly leakage and drift review",
    impact: "low",
    why: "Keeps leaderboard trust high and prevents silent model decay.",
    nextStep: "Schedule recurring model governance check every Monday."
  }
];

export const workspaces: WorkspaceContext[] = [
  {
    id: "11111111-1111-1111-1111-111111111111",
    name: "Marketing Analytics",
    slug: "marketing",
    description: "Churn analysis, campaign performance, and customer segmentation.",
    owner: "Naveen",
    activeDatasetId: "ds-churn-2024",
    role: "admin",
    createdAt: "2024-01-10",
    members: [
      { id: "u-1", name: "Naveen", email: "naveen@company.com", avatar: "N", role: "admin", status: "active", joinedAt: "2024-01-10" },
      { id: "u-2", name: "Asha", email: "asha@company.com", avatar: "A", role: "editor", status: "active", joinedAt: "2024-01-15" },
      { id: "u-3", name: "Ravi", email: "ravi@company.com", avatar: "R", role: "viewer", status: "pending", joinedAt: "2024-02-01" },
      { id: "u-4", name: "Priya", email: "priya@company.com", avatar: "P", role: "editor", status: "active", joinedAt: "2024-02-10" },
    ],
  },
  {
    id: "22222222-2222-2222-2222-222222222222",
    name: "Revenue Ops",
    slug: "revenue-ops",
    description: "Q3 sales pipeline, forecasting, and deal analytics.",
    owner: "Naveen",
    activeDatasetId: "ds-sales-q3",
    role: "admin",
    createdAt: "2024-02-01",
    members: [
      { id: "u-1", name: "Naveen", email: "naveen@company.com", avatar: "N", role: "admin", status: "active", joinedAt: "2024-02-01" },
      { id: "u-4", name: "Priya", email: "priya@company.com", avatar: "P", role: "editor", status: "active", joinedAt: "2024-02-10" },
    ],
  },
  {
    id: "33333333-3333-3333-3333-333333333333",
    name: "Research Lab",
    slug: "research-lab",
    description: "Experimental models and prototype data pipelines.",
    owner: "Naveen",
    activeDatasetId: "",
    role: "admin",
    createdAt: "2024-03-05",
    members: [
      { id: "u-1", name: "Naveen", email: "naveen@company.com", avatar: "N", role: "admin", status: "active", joinedAt: "2024-03-05" },
    ],
  },
];

export const workspace: WorkspaceContext = workspaces[0];
