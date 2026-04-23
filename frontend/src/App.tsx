import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/shared/AppShell";
import { ProtectedRoute } from "./components/auth/ProtectedRoute";
import { PublicOnlyRoute } from "./components/auth/PublicOnlyRoute";
import { HomePage } from "./features/home/HomePage";
import { DataImportPage } from "./features/data-import/DataImportPage";
import { DataTransformPage } from "./features/data-transform/DataTransformPage";
import { QueryStudioPage } from "./features/query-studio/QueryStudioPage";
import { EDADashboardsPage } from "./features/eda-dashboards/EDADashboardsPage";
import { MLTrainingPage } from "./features/ml-training/MLTrainingPage";
import { MLPredictionPage } from "./features/ml-prediction/MLPredictionPage";
import { WorkspaceSettingsPage } from "./features/workspace/WorkspaceSettingsPage";
import { AutoMLLabPage } from "./features/automl-lab/AutoMLLabPage";
import { AuthPage } from "./features/auth/AuthPage";
import { AboutPage } from "./features/about/AboutPage";

export function App() {
  return (
    <Routes>
      <Route
        path="/auth"
        element={
          <PublicOnlyRoute>
            <AuthPage />
          </PublicOnlyRoute>
        }
      />

      <Route
        element={
          <ProtectedRoute>
            <AppShell />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<HomePage />} />
        <Route path="/data-import" element={<DataImportPage />} />
        <Route path="/data-transform" element={<DataTransformPage />} />
        <Route path="/query-studio" element={<QueryStudioPage />} />
        <Route path="/eda-dashboards" element={<EDADashboardsPage />} />
        <Route path="/automl-lab" element={<AutoMLLabPage />} />
        <Route path="/ml-training" element={<MLTrainingPage />} />
        <Route path="/ml-prediction" element={<MLPredictionPage />} />
        <Route path="/workspace-settings" element={<WorkspaceSettingsPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
