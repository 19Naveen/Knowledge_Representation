import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { useAuthContext } from "./AuthContext";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export interface Workspace {
  id: string;
  name: string;
  slug: string;
  description?: string;
  owner_id: string;
  created_at: string;
}

interface WorkspaceContextValue {
  workspaces: Workspace[];
  activeWorkspace: Workspace | null;
  isLoading: boolean;
  switchWorkspace: (id: string) => void;
  createWorkspace: (name: string, description?: string) => Promise<Workspace>;
  refetch: () => Promise<void>;
}

const WorkspaceCtx = createContext<WorkspaceContextValue | undefined>(undefined);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const { session, isAuthenticated } = useAuthContext();
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const authHeaders = useCallback(() => ({
    "Content-Type": "application/json",
    Authorization: `Bearer ${session?.accessToken}`,
  }), [session]);

  const fetchWorkspaces = useCallback(async () => {
    if (!isAuthenticated || !session) {
      setWorkspaces([]);
      setIsLoading(false);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/workspaces`, { headers: authHeaders() });
      if (!res.ok) return;
      const data: Workspace[] = await res.json();
      setWorkspaces(data);
      // Auto-select first workspace if none active or active no longer exists
      setActiveWorkspaceId((prev) => {
        if (prev && data.find((w) => w.id === prev)) return prev;
        return data[0]?.id ?? null;
      });
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated, session, authHeaders]);

  useEffect(() => {
    fetchWorkspaces();
  }, [fetchWorkspaces]);

  const switchWorkspace = useCallback((id: string) => {
    setActiveWorkspaceId(id);
  }, []);

  const createWorkspace = useCallback(async (name: string, description?: string): Promise<Workspace> => {
    const res = await fetch(`${API_BASE}/workspaces`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ name, description }),
    });
    if (!res.ok) {
      const body = await res.json();
      throw new Error(body.detail ?? "Failed to create workspace");
    }
    const ws: Workspace = await res.json();
    setWorkspaces((prev) => [...prev, ws]);
    setActiveWorkspaceId(ws.id);
    return ws;
  }, [authHeaders]);

  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId) ?? workspaces[0] ?? null;

  return (
    <WorkspaceCtx.Provider value={{ workspaces, activeWorkspace, isLoading, switchWorkspace, createWorkspace, refetch: fetchWorkspaces }}>
      {children}
    </WorkspaceCtx.Provider>
  );
}

export function useWorkspaceContext(): WorkspaceContextValue {
  const ctx = useContext(WorkspaceCtx);
  if (!ctx) throw new Error("useWorkspaceContext must be used within WorkspaceProvider");
  return ctx;
}
