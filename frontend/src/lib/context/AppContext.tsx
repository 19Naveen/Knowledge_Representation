import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { datasets as initialDatasets, workspace as initialWorkspace } from '../mocks/data';
import { Dataset, WorkspaceContext } from '../mocks/types';

interface Widget {
    id: string;
    type: "line" | "bar" | "pie" | "table" | "kpi";
    title: string;
    xAxis: string;
    yAxis: string;
    aggregation: "sum" | "avg" | "count" | "min" | "max";
    data: { label: string; value: number }[];
}

interface Dashboard {
    id: string;
    name: string;
    datasetId: string;
    widgets: Widget[];
}

interface AppContextType {
    datasets: Dataset[];
    addDataset: (dataset: Dataset) => void;
    activeDatasetId: string;
    setActiveDatasetId: (id: string) => void;
    activeDataset: Dataset | undefined;

    dashboards: Dashboard[];
    addDashboard: (db: Dashboard) => void;
    updateDashboard: (db: Dashboard) => void;

    workspace: WorkspaceContext;
    setWorkspace: (ws: WorkspaceContext) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
    const [datasets, setDatasets] = useState<Dataset[]>(initialDatasets);
    const [activeDatasetId, setActiveDatasetId] = useState<string>(initialWorkspace.activeDatasetId);
    const [workspace, setWorkspace] = useState<WorkspaceContext>(initialWorkspace);
    const [dashboards, setDashboards] = useState<Dashboard[]>([
        {
            id: 'db-1',
            name: 'Default Dashboard',
            datasetId: initialWorkspace.activeDatasetId,
            widgets: [
                { id: "w1", type: "kpi", title: "Total Users", xAxis: "date", yAxis: "users", aggregation: "sum", data: [] },
                { id: "w2", type: "line", title: "Revenue Trend", xAxis: "date", yAxis: "revenue", aggregation: "sum", data: [] },
            ]
        }
    ]);

    const addDataset = useCallback((ds: Dataset) => {
        setDatasets(prev => [...prev, ds]);
    }, []);

    const addDashboard = useCallback((db: Dashboard) => {
        setDashboards(prev => [...prev, db]);
    }, []);

    const updateDashboard = useCallback((db: Dashboard) => {
        setDashboards(prev => prev.map(d => d.id === db.id ? db : d));
    }, []);

    const activeDataset = datasets.find(d => d.id === activeDatasetId);

    return (
        <AppContext.Provider value={{
            datasets,
            addDataset,
            activeDatasetId,
            setActiveDatasetId,
            activeDataset,
            dashboards,
            addDashboard,
            updateDashboard,
            workspace,
            setWorkspace
        }}>
            {children}
        </AppContext.Provider>
    );
}

export function useAppContext() {
    const context = useContext(AppContext);
    if (context === undefined) {
        throw new Error('useAppContext must be used within an AppProvider');
    }
    return context;
}
