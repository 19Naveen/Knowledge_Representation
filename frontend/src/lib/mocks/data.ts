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

export const workspace: WorkspaceContext = {
  id: "ws-marketing",
  name: "Marketing",
  owner: "Naveen",
  activeDatasetId: "ds-churn-2024",
  members: 3,
  role: "admin"
};
