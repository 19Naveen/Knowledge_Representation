import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/shared/AppShell";
import { HomePage } from "./features/home/HomePage";
import { DataImportPage } from "./features/data-import/DataImportPage";
import { DataTransformPage } from "./features/data-transform/DataTransformPage";
import { QueryStudioPage } from "./features/query-studio/QueryStudioPage";
import { EDADashboardsPage } from "./features/eda-dashboards/EDADashboardsPage";
import { AutoMLLabPage } from "./features/automl/AutoMLLabPage";
import { MLTrainingPage } from "./features/ml-training/MLTrainingPage";
import { MLPredictionPage } from "./features/ml-prediction/MLPredictionPage";
import { DeploySimPage } from "./features/deploy-sim/DeploySimPage";
import { WorkspaceSettingsPage } from "./features/workspace/WorkspaceSettingsPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/data-import" element={<DataImportPage />} />
        <Route path="/data-transform" element={<DataTransformPage />} />
        <Route path="/query-studio" element={<QueryStudioPage />} />
        <Route path="/eda-dashboards" element={<EDADashboardsPage />} />
        <Route path="/automl-lab" element={<AutoMLLabPage />} />
        <Route path="/ml-training" element={<MLTrainingPage />} />
        <Route path="/ml-prediction" element={<MLPredictionPage />} />
        <Route path="/deploy-sim" element={<DeploySimPage />} />
        <Route path="/workspace-settings" element={<WorkspaceSettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
