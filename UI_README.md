# UI_README — Frontend UI Structure and Functionality (excluding Dashboard)

This document describes the structure, responsibilities, and behavior of the frontend UI for the AI system, excluding the dashboard feature.

Repository location: `frontend/`

## High-level structure

- `frontend/src/` — main source tree used by the app runtime.
  - `App.tsx` — top-level app component responsible for routing and global providers.
  - `main.tsx` — bootstraps React and renders `App` into the DOM.
  - `components/` — shared UI building blocks and route guards.
    - `auth/ProtectedRoute.tsx` — wrapper that only allows access to authenticated users; redirects otherwise.
    - `auth/PublicOnlyRoute.tsx` — route wrapper that prevents access to authenticated users (for login/register pages).
    - `shared/AppShell.tsx` — application shell (header, nav, main content area) used across pages.
    - `shared/EmptyState.tsx` — generic empty-state UI with optional actions.
    - `shared/MetricTile.tsx` — small summary tile for metrics display.
    - `shared/PageHeader.tsx` — page title + optional actions and breadcrumbs.
    - `shared/StatCard.tsx` — card for a numeric/statistic and its metadata.
    - `shared/StatusPill.tsx` — small pill-shaped status indicator.

  - `features/` — feature-specific pages and UI; each folder is a route target.
    - `about/AboutPage.tsx` — static information about the app and team.
    - `auth/AuthPage.tsx` — authentication UI (login / register flows). Uses `PublicOnlyRoute`.
    - `automl-lab/AutoMLLabPage.tsx` — UI for AutoML experiments and workflows.
    - `data-import/DataImportPage.tsx` — file upload and ingest interface for datasets.
    - `data-transform/DataTransformPage.tsx` — dataset transformation, schema mapping, and preview UI.
    - `deploy-sim/DeploySimPage.tsx` — deployment simulation and configuration UI.
    - `eda-dashboards/EDADashboardsPage.tsx` — exploratory data analysis views (not the main dashboard feature).
    - `home/HomePage.tsx` — landing page after login; high-level navigation and quick actions.
    - `ml-prediction/MLPredictionPage.tsx` — interface to run predictions against models and view outputs.
    - `ml-training/MLTrainingPage.tsx` — model training configuration, progress, and controls.
    - `query-studio/QueryStudioPage.tsx` — natural language / structured query UI to interrogate datasets and models.
    - `workspace/WorkspaceSettingsPage.tsx` — workspace-level settings, API keys, and environment configuration.

  - `lib/` — utilities, contexts, and mocks.
    - `cn.ts` — helper for className composition (utility for conditional CSS classes).
    - `auth/storage.ts` — local storage helpers for auth tokens and user session data.
    - `auth/types.ts` — TypeScript types for auth-related models (User, Credentials, etc.).
    - `context/AppContext.tsx` — global app context (app-level settings, feature flags).
    - `context/AuthContext.tsx` — authentication context that provides user, login, logout, and token state.
    - `mocks/` — mock data and types used in development and storybook-like previews.

  - `styles/` — global CSS tokens and Tailwind-derived styles.
    - `globals.css` — app-wide base styles and Tailwind imports.
    - `tokens.css` — design tokens (colors, spacing, typography variables).

  - `public/` — static assets served by the dev server/production build.

## Routing and Navigation

- Routing is defined in `App.tsx`. Pages under `features/` are mapped to application routes.
- Authorization wrappers used:
  - `ProtectedRoute` protects pages that require an authenticated user.
  - `PublicOnlyRoute` hides pages from already authenticated users (e.g., login).
- `AppShell` composes the header, side nav (if present), and a main content area. Page-specific headers are implemented by `PageHeader` inside individual pages.

## Authentication flow

- `AuthContext` exposes:
  - `user` — current user object or `null`.
  - `login(credentials)` — logs in, stores token in `auth/storage`, sets `user`.
  - `logout()` — clears storage and user state.
  - `refresh()` — optional token refresh method.
- `AuthPage` provides the UI for login/register and uses `auth/storage` to persist tokens. On success, it redirects to `HomePage`.

## Key pages — responsibilities and interactions
`HomePage`:
  - Responsibilities:
    - Provide a post-login overview and entrypoints for main ML workflows.
    - Surface recent activity, recent datasets, and quick actions (upload dataset, start training, open Query Studio).
  - Primary components used: `AppShell`, `PageHeader`, `MetricTile`, `StatCard`, `EmptyState`.
  - Typical API calls:
    - `GET /api/workspace/summary` → { activeRuns, recentDatasets[], usageMetrics }
  - Data & UX flows:
    - On mount, loads workspace summary and displays cards; quick-action buttons open modals or navigate to feature pages.
    - Error state: show `EmptyState` with retry action if summary fails.

`AutoMLLabPage`:
  - Responsibilities:
    - Guide users through configuring AutoML experiments (dataset selection, target, metric, search space).
    - Launch experiments and display run history with status and links to results.
  - Primary components used: config forms, run-table, `StatusPill`, run-detail drawer.
  - Typical API calls:
    - `GET /api/datasets` (for dataset selector)
    - `POST /api/automl/run` (start run)
    - `GET /api/automl/runs` (history)
  - Data & UX flows:
    - Stepper UI: Select dataset → Choose target/metric → Configure search space → Start run.
    - After starting, optimistic UI adds run to history and polls run status. Failures show inline error and link to logs.

`DataImportPage`:
  - Responsibilities:
    - Accept user dataset files, run server-side schema inference, and persist dataset metadata.
    - Provide a column-mapping UI so users can rename/choose types before ingestion.
  - Primary components used: file dropzone, preview table, mapping form, progress modal.
  - Typical API calls:
    - `POST /api/datasets/upload` (multipart) → returns `uploadId`
    - `GET /api/datasets/preview?uploadId=` → { columns: [{name,type,sample}] , rowsPreview[] }
    - `POST /api/datasets` (create dataset with mapping)
  - Data & UX flows:
    - User drops file → preview shows inferred columns → user adjusts types/mappings → confirm ingest → show progress.
    - Error states: parse errors show column-level messages; large files surface a chunked-upload fallback.

`DataTransformPage`:
  - Responsibilities:
    - Provide a pipeline-style UI for dataset transforms (filter, normalize, encode, aggregate).
    - Allow users to preview transform results and persist transformations as new dataset versions.
  - Primary components used: transform stepper, live preview table, save/apply controls.
  - Typical API calls:
    - `GET /api/datasets/{id}`
    - `POST /api/datasets/{id}/transform/preview` (body: transform spec) → preview rows
    - `POST /api/datasets/{id}/transform` (apply and persist)
  - Data & UX flows:
    - Build transform steps in UI; each step updates a local spec. Preview sends spec to backend for sample results.
    - Save creates a new dataset version. If preview fails, UI highlights the step causing the error.

`MLTrainingPage`:
  - Responsibilities:
    - Configure and start model training runs, show run status, view logs and metrics.
    - Provide controls for stopping/canceling runs and viewing artifacts.
  - Primary components used: training form, run list, log viewer, `StatusPill`, charts for metrics.
  - Typical API calls:
    - `GET /api/models/available` (for model templates)
    - `POST /api/training/start` (config + dataset)
    - `GET /api/training/{runId}/status`
    - `GET /api/training/{runId}/logs` (streaming)
  - Data & UX flows:
    - User selects model, hyperparams, dataset → start training. UI shows run card with live-updating status and metric charts.
    - Streaming logs are appended to a log viewer; errors surface with actionable links to common fixes.

`MLPredictionPage`:
  - Responsibilities:
    - Allow users to run inference against a selected trained model and dataset (single or batch).
    - Visualize prediction outputs and provide export/download options.
  - Primary components used: model selector, dataset selector, run-preview, results table, small charts.
  - Typical API calls:
    - `GET /api/models` (list trained models)
    - `POST /api/predictions/run` (body: modelId, datasetId, options) → returns jobId or results
    - `GET /api/predictions/{jobId}/results`
  - Data & UX flows:
    - For single prediction, show inline result. For batch, show job card and poll for results. Exports available as CSV/JSON.
    - Error states: model mismatch or missing feature columns show a preflight validation error with remediation hints.

`QueryStudioPage`:
  - Responsibilities:
    - Provide a query interface supporting both natural-language prompts and SQL-like queries against dataset tables.
    - Allow saving, editing, and rerunning queries; surface query history.
  - Primary components used: query editor (text area with mode toggle), results grid, query history panel.
  - Typical API calls:
    - `POST /api/query/run` (body: { mode: 'nl'|'sql', query, datasetId }) → { columns, rows }
    - `GET /api/query/history`
  - Data & UX flows:
    - User composes a query → run → results displayed in grid with options to save or export. NL mode displays the parsed SQL and confidence.
    - Error states: invalid SQL or ambiguous NL prompts return clear errors and suggestion prompts.

`DeploySimPage`:
  - Responsibilities:
    - Let users configure deployment endpoints and simulate requests against models, including headers and payload templates.
    - Provide cost/latency estimation for proposed deployment settings.
  - Primary components used: endpoint form, request/response inspector, latency/cost estimator.
  - Typical API calls:
    - `POST /api/deploy/simulate` (body: endpointConfig, sampleRequest) → { responseSample, estimatedLatency, costEstimate }
  - Data & UX flows:
    - User configures endpoint → simulate → inspect sample responses and estimated metrics. Errors show invalid config hints.

`WorkspaceSettingsPage`:
  - Responsibilities:
    - Manage workspace metadata, API keys, collaborators, billing toggles, and environment flags.
  - Primary components used: forms, key rotator modal, access-control lists.
  - Typical API calls:
    - `GET /api/workspace` → workspace metadata
    - `POST /api/workspace/keys` (rotate/create)
    - `PUT /api/workspace/collaborators` (update access)
  - Data & UX flows:
    - Editing settings updates backend immediately or via a Save button. Key creation shows ephemeral key once and warns to copy it.
    - Errors: permission errors are surfaced with instructions to contact an admin.

## Shared components details

- `AppShell`:
  - Layout: header with application name and user menu, optional left nav, main content.
  - Receives child content as the routed page.

- `PageHeader`:
  - Standardized page title, subtitle, and action area. All pages should use to keep consistent spacing.

- `StatCard` / `MetricTile`:
  - Accept `title`, `value`, `delta` (optional) and an icon. Designed for dashboard-like summaries across pages.

- `EmptyState`:
  - Generic component shown when no data exists; accepts `title`, `description`, and `actions`.

## Context and state management

- `AuthContext` — authentication state and helpers.
- `AppContext` — feature flags, selected workspace, and other global settings.
- Page-level local state used for forms, previews, and paginated lists.

## API surface used by the UI

- The frontend communicates with the backend API under `backend/api/` (not documented here). Typical API interactions:
  - Auth endpoints (login, register, refresh)
  - Dataset endpoints (upload, preview, schema, transform)
  - Training endpoints (start, status, logs)
  - Prediction endpoints (run, results, export)
  - Workspace / settings endpoints

Requests are made from page components or feature-specific services; auth tokens are included from `auth/storage`.

## Development notes

- Local dev server: `frontend` is built with Vite. Use the project `package.json` scripts (e.g., `npm run dev`) to start.
- Style system: Tailwind + CSS tokens. Edit tokens in `styles/tokens.css` and global imports in `styles/globals.css`.
- Mocks: `lib/mocks` contains sample data for offline UI development.

## Testing and Storybook (recommended)

- Create unit tests for shared components with React Testing Library and jest.
- Add Storybook stories for `AppShell`, `PageHeader`, `StatCard`, and `EmptyState` to capture visual permutations.

## How to add a new UI page (convention)

1. Create a new folder under `frontend/src/features/<feature-name>` and add `<FeaturePage>.tsx`.
2. Use `PageHeader` at the top for consistent layout.
3. If route requires auth, wrap route with `ProtectedRoute` in `App.tsx` routing map.
4. Add styles referencing tokens from `styles/tokens.css` and use shared components where appropriate.
5. Add unit tests and (optionally) a Storybook story.

## Files of interest (quick links)

- `frontend/src/App.tsx` — central routing and layout.
- `frontend/src/main.tsx` — app bootstrap.
- `frontend/src/components/shared/AppShell.tsx` — app chrome.
- `frontend/src/components/auth/ProtectedRoute.tsx` — auth guard.


If you want, I can:

## Route Map (paths → components)

All routes under the protected area are wrapped by `ProtectedRoute` and rendered inside `AppShell`. The `/auth` route uses `PublicOnlyRoute`.

- `/auth` — `AuthPage` (PublicOnlyRoute)
  - Purpose: Login / Register UI. Redirects to `/` on success.

- `/` — `HomePage` (ProtectedRoute → AppShell)
  - Purpose: Post-login landing page, quick actions, and navigation.

- `/data-import` — `DataImportPage` (ProtectedRoute → AppShell)
  - Purpose: Upload datasets (CSV/Parquet), preview, and ingest.

- `/data-transform` — `DataTransformPage` (ProtectedRoute → AppShell)
  - Purpose: Build and apply dataset transforms with preview.

- `/query-studio` — `QueryStudioPage` (ProtectedRoute → AppShell)
  - Purpose: Natural language / SQL-like queries against datasets.

- `/eda-dashboards` — `EDADashboardsPage` (ProtectedRoute → AppShell)
  - Purpose: Exploratory data analysis views and visualizations.

- `/automl-lab` — `AutoMLLabPage` (ProtectedRoute → AppShell)
  - Purpose: Configure and run AutoML experiments.

- `/ml-training` — `MLTrainingPage` (ProtectedRoute → AppShell)
  - Purpose: Configure training runs, view progress, and logs.

- `/ml-prediction` — `MLPredictionPage` (ProtectedRoute → AppShell)
  - Purpose: Run batch or single predictions and review results.

- `/workspace-settings` — `WorkspaceSettingsPage` (ProtectedRoute → AppShell)
  - Purpose: Manage API keys, collaborators, and workspace settings.

- `/about` — `AboutPage` (ProtectedRoute → AppShell)
  - Purpose: App information and credits.

- `*` (unknown) — redirects to `/` via `<Navigate to="/" replace />`.

If you'd like, I can also generate a machine-readable JSON route manifest or add direct code links to each component.
