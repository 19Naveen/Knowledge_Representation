# Frontend — Knowledge Representation Platform

React 18 + TypeScript + Vite frontend for the KnowRep platform.

> Part of the monorepo. See [root README](../README.md) for platform overview and [backend README](../backend/README.md) for API detail.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Structure](#structure)
3. [Pages & Routes](#pages--routes)
4. [State Management](#state-management)
5. [API Integration](#api-integration)
6. [Design System](#design-system)
7. [Environment Variables](#environment-variables)
8. [Running the Frontend](#running-the-frontend)
9. [Changelog](#changelog)

---

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| React | 18.3.1 | UI framework |
| TypeScript | 5.6.3 | Type safety |
| Vite | 5.4.10 | Build tool and dev server |
| react-router-dom | 6.30.1 | Client-side routing |
| TailwindCSS | 3.4.14 | Utility-first CSS |
| lucide-react | 0.468.0 | Icon library |

No Redux or Zustand — state is managed via React Context API.

---

## Structure

```
frontend/
├── index.html                     # Root HTML, mounts #root
├── vite.config.ts                 # Vite config (React plugin)
├── tailwind.config.ts             # Tailwind + design tokens
├── tsconfig.json                  # TypeScript config (strict mode)
│
└── src/
    ├── main.tsx                   # Entry point: context providers + BrowserRouter
    ├── App.tsx                    # Route definitions + layout
    │
    ├── components/
    │   ├── auth/
    │   │   ├── ProtectedRoute.tsx # Redirects to /auth if not authenticated
    │   │   └── PublicOnlyRoute.tsx# Redirects to / if already authenticated
    │   └── shared/
    │       ├── AppShell.tsx       # Main layout: sidebar + header + outlet
    │       ├── BrandMark.tsx      # "K" tile + "KnowRep" wordmark (size/inverted props)
    │       ├── AuthCardLayout.tsx # Centered gradient card shell for auth pages
    │       ├── PageHeader.tsx     # Page title + subtitle component
    │       ├── EmptyState.tsx     # Empty state placeholder
    │       ├── MetricTile.tsx     # Single metric display tile
    │       ├── StatCard.tsx       # Stat card with label and value
    │       └── StatusPill.tsx     # Status badge (active, pending, error)
    │
    ├── features/                  # One folder per platform feature
    │   ├── home/
    │   │   └── HomePage.tsx
    │   ├── landing/
    │   │   ├── LandingPage.tsx    # Composition root
    │   │   ├── landingContent.ts  # Typed feature + step arrays
    │   │   └── components/        # LandingNavbar, HeroSection, FeaturesSection, HowItWorksSection, CtaBanner, LandingFooter
    │   ├── auth/
    │   │   ├── AuthPage.tsx       # Legacy combined page (kept for reference)
    │   │   ├── SigninPage.tsx     # Email + password sign in → /app
    │   │   └── SignupPage.tsx     # Registration → /onboarding
    │   ├── onboarding/
    │   │   └── OnboardingPage.tsx # Workspace creation; no AppShell
    │   ├── data-import/
    │   │   └── DataImportPage.tsx
    │   ├── data-transform/
    │   │   └── DataTransformPage.tsx
    │   ├── query-studio/
    │   │   └── QueryStudioPage.tsx
    │   ├── eda-dashboards/
    │   │   └── EDADashboardsPage.tsx
    │   ├── ml-training/
    │   │   └── MLTrainingPage.tsx
    │   ├── ml-prediction/
    │   │   └── MLPredictionPage.tsx
    │   ├── automl-lab/
    │   │   └── AutoMLLabPage.tsx
    │   ├── deploy-sim/
    │   │   └── DeploySimPage.tsx
    │   ├── workspace/
    │   │   └── WorkspaceSettingsPage.tsx
    │   └── about/
    │       └── AboutPage.tsx
    │
    ├── lib/
    │   ├── auth/
    │   │   ├── types.ts           # Auth type definitions (User, Session, etc.)
    │   │   └── storage.ts         # localStorage read/write for session
    │   ├── context/
    │   │   ├── AuthContext.tsx    # Auth state + backend API calls
    │   │   ├── AppContext.tsx     # Datasets and dashboards state
    │   │   └── WorkspaceContext.tsx # Workspace and member state
    │   ├── mocks/
    │   │   ├── data.ts            # Mock data for non-auth features
    │   │   └── types.ts           # Types for mock data
    │   └── cn.ts                  # Classname utility (clsx/twMerge pattern)
    │
    └── styles/
        ├── globals.css            # Tailwind directives + base styles
        └── tokens.css             # CSS custom properties (colors, spacing)
```

---

## Pages & Routes

Public routes are open to all. `/signin` and `/signup` are wrapped in `<PublicOnlyRoute>` (redirect to `/app` if logged in). `/onboarding` is `<ProtectedRoute>` without AppShell. All `/app/*` routes require authentication via `<ProtectedRoute>` and render inside `<AppShell>`.

| Path | Component | Auth Required | Description |
|---|---|---|---|
| `/` | `LandingPage` | No | Marketing landing page |
| `/signin` | `SigninPage` | No (public only) | Email + password sign in |
| `/signup` | `SignupPage` | No (public only) | Username + email + password registration |
| `/auth` | Redirect → `/signin` | — | Legacy redirect |
| `/onboarding` | `OnboardingPage` | Yes | Create first workspace (no AppShell) |
| `/app` | `HomePage` | Yes | Platform dashboard / overview |
| `/app/data-import` | `DataImportPage` | Yes | Ingest files or connect databases |
| `/app/data-transform` | `DataTransformPage` | Yes | Apply transformation steps to datasets |
| `/app/query-studio` | `QueryStudioPage` | Yes | SQL queries on Parquet via DuckDB |
| `/app/eda-dashboards` | `EDADashboardsPage` | Yes | Exploratory data analysis dashboards |
| `/app/automl-lab` | `AutoMLLabPage` | Yes | AutoML pipeline configuration |
| `/app/ml-training` | `MLTrainingPage` | Yes | Manual ML training runs |
| `/app/ml-prediction` | `MLPredictionPage` | Yes | Run and view model predictions |
| `/app/workspace-settings` | `WorkspaceSettingsPage` | Yes | Workspace members, roles, settings |
| `/app/about` | `AboutPage` | Yes | Platform information |
| `*` | Redirect → `/` | — | Catch-all |

### Sidebar navigation groups

| Group | Pages |
|---|---|
| Overview | Home |
| Data Engine | Data Import, Data Transform, Query Studio, EDA Dashboards |
| Intelligence | AutoML Lab, ML Training, ML Prediction |
| Predictive Models | Deploy Sim |

---

## State Management

Three React Contexts with no external state library. Provider hierarchy:

```
AuthProvider
  └── WorkspaceProvider
        └── AppProvider
              └── BrowserRouter
                    └── App
```

### AuthContext (`src/lib/context/AuthContext.tsx`)

Manages authentication state and all auth API calls.

| Export | Type | Description |
|---|---|---|
| `isAuthenticated` | `boolean` | Whether a valid session exists |
| `isLoading` | `boolean` | Auth operation in progress |
| `user` | `User \| null` | Current user profile |
| `session` | `Session \| null` | Raw session with token |
| `loginWithCredentials(email, password)` | `async fn` | POST `/auth/login` |
| `signupWithCredentials(username, email, password)` | `async fn` | POST `/auth/signup` |
| `logout()` | `fn` | Clears session from state and localStorage |

Session is persisted in localStorage under key `knowrep.auth.session` and rehydrated on app load.

### AppContext (`src/lib/context/AppContext.tsx`)

Manages app-level data state (currently backed by mock data).

| Export | Type | Description |
|---|---|---|
| `datasets` | `Dataset[]` | All datasets in the active workspace |
| `activeDatasetId` | `string \| null` | Currently selected dataset |
| `activeDataset` | `Dataset \| null` | Derived from activeDatasetId |
| `dashboards` | `Dashboard[]` | Dashboard configurations |
| `addDataset(dataset)` | `fn` | Add a new dataset |
| `addDashboard(dashboard)` | `fn` | Add a new dashboard |
| `updateDashboard(id, updates)` | `fn` | Update dashboard config |

### WorkspaceContext (`src/lib/context/WorkspaceContext.tsx`)

Manages workspace switching and member management.

| Export | Type | Description |
|---|---|---|
| `workspaces` | `Workspace[]` | All workspaces the user belongs to |
| `activeWorkspace` | `Workspace \| null` | Currently selected workspace |
| `switchWorkspace(id)` | `fn` | Change active workspace |
| `createWorkspace(name, description?)` | `fn` | Create a new workspace |
| `updateWorkspace(updates)` | `fn` | Update active workspace |
| `inviteMember(email, role)` | `fn` | Invite a member to the workspace |
| `updateMemberRole(memberId, role)` | `fn` | Change a member's role |
| `removeMember(memberId)` | `fn` | Remove a member |

---

## API Integration

API calls use the native `fetch` API — no axios or custom HTTP client.

**Base URL** (configured via env var):

```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1"
```

**Auth header pattern** (for all protected requests):

```typescript
headers: {
  "Authorization": `Bearer ${session.access_token}`,
  "Content-Type": "application/json",
}
```

**Implemented API calls** (in `AuthContext`):

| Call | Endpoint | Description |
|---|---|---|
| Login | `POST /auth/login` | Returns `access_token`, `user`, `expires_in` |
| Signup | `POST /auth/signup` | Returns same as login |

All other features currently use mock data from `src/lib/mocks/data.ts` and will be wired to backend endpoints as modules are implemented.

---

## Design System

Built on TailwindCSS with a custom design token layer.

### Color tokens (`src/styles/tokens.css`)

HSL-based CSS custom properties — dark theme by default:

| Token | Usage |
|---|---|
| `--color-primary` | Deep indigo/near-black — backgrounds |
| `--color-accent` | Electric blue — interactive elements |
| `--color-success` | Green |
| `--color-warning` | Amber |
| `--color-danger` | Red |

### Component utility classes (`tailwind.config.ts`)

| Class | Purpose |
|---|---|
| `.card` | Surface container with border and radius |
| `.btn` | Base button |
| `.btn-primary` | Accent-coloured primary action button |
| `.btn-secondary` | Outlined secondary button |
| `.input` | Styled form input |
| `.badge` | Small status label |

### Typography

- **Body:** Inter (sans-serif)
- **Code / monospace:** JetBrains Mono

### Custom animations

`fadeIn`, `slideUp`, `scaleIn` — defined in `tailwind.config.ts`.

---

## Environment Variables

Create `frontend/.env.local` for local overrides (not committed).

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | Backend API base URL |

All Vite env vars must be prefixed with `VITE_` to be accessible in the browser via `import.meta.env`.

---

## Running the Frontend

```bash
# From repo root
make frontend

# Or directly
cd frontend && npm run dev
```

Dev server runs at `http://localhost:5173` (Vite default).

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

---

## Changelog

### 2026-06-07 — Landing page, auth flow, and onboarding added

- Added public landing page at `/` with Navbar, Hero, Features, HowItWorks, CtaBanner, Footer sections
- Added `/signin` (SigninPage) and `/signup` (SignupPage) replacing the combined `/auth` page; `/auth` now redirects to `/signin`
- Added `/onboarding` (OnboardingPage) — full-screen workspace creation step post-signup, no AppShell
- Moved all protected app routes from `/` to `/app/*` namespace to cleanly separate public and authenticated surfaces
- Added `BrandMark` and `AuthCardLayout` shared components reused across auth/onboarding/landing

### 2026-06-06 — frontend/README.md created

- Documented full frontend structure, routing, state management, API integration, and design system
- Based on exploration of current codebase state

### Prior — Initial frontend implementation

- React 18 + Vite + TypeScript project setup
- react-router-dom v6 routing with ProtectedRoute / PublicOnlyRoute guards
- Three-context state management (Auth, App, Workspace)
- TailwindCSS with custom design token system
- AppShell layout with responsive sidebar navigation
- Auth page (login + signup) wired to backend `/auth` endpoints
- All feature pages scaffolded (Data Import, Transform, Query Studio, EDA, ML, AutoML, Deploy Sim, Workspace Settings)
- Mock data layer for non-auth features pending backend integration
