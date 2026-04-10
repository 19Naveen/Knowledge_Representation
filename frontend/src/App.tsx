import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/shared/AppShell";
import { HomePage } from "./features/home/HomePage";
import { DataStudioPage } from "./features/data-studio/DataStudioPage";
import { QueryStudioPage } from "./features/query-studio/QueryStudioPage";
import { EDADashboardsPage } from "./features/eda-dashboards/EDADashboardsPage";
import { AutoMLLabPage } from "./features/automl/AutoMLLabPage";
import { DeploySimPage } from "./features/deploy-sim/DeploySimPage";
import { WorkspaceSettingsPage } from "./features/workspace/WorkspaceSettingsPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/data-studio" element={<DataStudioPage />} />
        <Route path="/query-studio" element={<QueryStudioPage />} />
        <Route path="/eda-dashboards" element={<EDADashboardsPage />} />
        <Route path="/automl-lab" element={<AutoMLLabPage />} />
        <Route path="/deploy-sim" element={<DeploySimPage />} />
        <Route path="/workspace-settings" element={<WorkspaceSettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
