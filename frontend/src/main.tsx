import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { AppProvider } from "./lib/context/AppContext";
import { AuthProvider } from "./lib/context/AuthContext";
import { WorkspaceProvider } from "./lib/context/WorkspaceContext";
import { App } from "./App";
import "./styles/globals.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AuthProvider>
      <WorkspaceProvider>
        <AppProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </AppProvider>
      </WorkspaceProvider>
    </AuthProvider>
  </React.StrictMode>
);
