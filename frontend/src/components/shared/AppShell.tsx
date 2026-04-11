import { useMemo } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { cn } from "../../lib/cn";
import { workspace, datasets } from "../../lib/mocks/data";

const navGroups = [
  {
    title: "Overview",
    items: [{ to: "/", label: "Project Home" }]
  },
  {
    title: "Data Foundation",
    items: [
      { to: "/data-import", label: "Data Import" },
      { to: "/data-transform", label: "Data Transformation" }
    ]
  },
  {
    title: "Exploration",
    items: [
      { to: "/query-studio", label: "AI Query Studio" },
      { to: "/eda-dashboards", label: "Dashboards & Reports" }
    ]
  },
  {
    title: "Machine Learning",
    items: [
      { to: "/ml-training", label: "Model Training" },
      { to: "/ml-prediction", label: "Model Prediction" }
    ]
  },
  {
    title: "Production",
    items: [{ to: "/deploy-sim", label: "Endpoints & Scoring" }]
  }
];

function navClass(isActive: boolean): string {
  return cn(
    "block rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
    isActive ? "bg-surface-2 text-text" : "text-text-secondary hover:bg-surface-2/50 hover:text-text"
  );
}

export function AppShell() {
  const location = useLocation();

  const breadcrumb = useMemo(() => {
    for (const group of navGroups) {
      for (const item of group.items) {
        if (item.to === "/" && location.pathname === "/") return item.label;
        if (item.to !== "/" && location.pathname.startsWith(item.to)) return item.label;
      }
    }
    if (location.pathname.includes("workspace-settings")) return "Workspace Settings";
    return "Project Home";
  }, [location.pathname]);

  const activeDataset = datasets.find((item) => item.id === workspace.activeDatasetId);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-bg text-text">
      {/* Sidebar - Desktop */}
      <aside className="hidden w-64 shrink-0 flex-col border-r bg-surface lg:flex">
        {/* Global/Tenant Switcher */}
        <div className="border-b px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2 font-heading font-semibold text-text">
            <div className="flex size-6 items-center justify-center rounded-md bg-primary text-xs text-white">K</div>
            KnowRep
          </div>
        </div>

        {/* Project Selector */}
        <div className="p-4 pb-2">
          <div className="flex flex-col gap-1 rounded-lg border border-border-subtle bg-surface-2/50 px-3 py-2 shadow-sm cursor-pointer hover:border-accent/30 transition-colors">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-text-tertiary">Current Project</p>
            <div className="flex items-center justify-between">
              <p className="truncate text-sm font-medium text-text">{workspace.name}</p>
              <span className="text-xs text-text-tertiary">▼</span>
            </div>
            <p className="mt-1 text-[10px] uppercase tracking-wider text-text-tertiary">Dataset: <span className="font-medium text-text">{activeDataset?.name ?? "None"}</span></p>
          </div>
        </div>

        {/* Grouped Navigation */}
        <nav className="flex-1 space-y-6 overflow-y-auto px-4 py-4">
          {navGroups.map((group) => (
            <div key={group.title}>
              <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-wider text-text-tertiary">{group.title}</p>
              <div className="space-y-0.5">
                {group.items.map((item) => (
                  <NavLink key={item.to} to={item.to} className={({ isActive }) => navClass(isActive)}>
                    {item.label}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>
        
        {/* Bottom Settings */}
        <div className="border-t border-border-subtle p-4">
          <NavLink to="/workspace-settings" className={({ isActive }) => navClass(isActive)}>
            Workspace Settings
          </NavLink>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-surface px-4 lg:px-8">
          <div className="flex items-center gap-3">
            <p className="text-sm font-medium text-text">
              {breadcrumb}
            </p>
          </div>
          <div className="hidden items-center gap-4 lg:flex">
            <div className="flex items-center gap-2 text-xs text-text-tertiary border-r border-border-subtle pr-4">
              <span className="flex h-2 w-2 rounded-full bg-success"></span>
              Environment: Production
            </div>
            <div className="flex size-7 items-center justify-center rounded-full bg-accent-light text-xs font-medium text-accent">
              {workspace.owner.charAt(0)}
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-4 lg:p-8 bg-bg">
          <div className="mx-auto max-w-[1200px]">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}