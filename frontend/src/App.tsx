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
import { AboutPage } from "./features/about/AboutPage";
import { LandingPage } from "./features/landing/LandingPage";
import { SigninPage } from "./features/auth/SigninPage";
import { SignupPage } from "./features/auth/SignupPage";
import { OnboardingPage } from "./features/onboarding/OnboardingPage";

export function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<LandingPage />} />

      {/* Auth — redirect to /app if already logged in */}
      <Route
        path="/signin"
        element={
          <PublicOnlyRoute>
            <SigninPage />
          </PublicOnlyRoute>
        }
      />
      <Route path="/signup" element={<SignupPage />} />

      {/* Legacy redirect */}
      <Route path="/auth" element={<Navigate to="/signin" replace />} />

      {/* Onboarding — protected but no AppShell */}
      <Route
        path="/onboarding"
        element={
          <ProtectedRoute>
            <OnboardingPage />
          </ProtectedRoute>
        }
      />

      {/* Protected app routes */}
      <Route
        path="/app"
        element={
          <ProtectedRoute>
            <AppShell />
          </ProtectedRoute>
        }
      >
        <Route index element={<HomePage />} />
        <Route path="data-import" element={<DataImportPage />} />
        <Route path="data-transform" element={<DataTransformPage />} />
        <Route path="query-studio" element={<QueryStudioPage />} />
        <Route path="eda-dashboards" element={<EDADashboardsPage />} />
        <Route path="automl-lab" element={<AutoMLLabPage />} />
        <Route path="ml-training" element={<MLTrainingPage />} />
        <Route path="ml-prediction" element={<MLPredictionPage />} />
        <Route path="workspace-settings" element={<WorkspaceSettingsPage />} />
        <Route path="about" element={<AboutPage />} />
      </Route>

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
