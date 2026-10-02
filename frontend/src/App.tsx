/**
 * Routes. A hash router (`/#/scans/...`) is used because the backend serves
 * the built UI as static files at `/` without an SPA fallback: every deep
 * link and reload then still loads `/index.html`, and the relative asset
 * paths of the build keep working.
 *
 * Pages are lazy-loaded; Plotly is loaded on demand by the plot pages only.
 */
import { lazy, Suspense, type ReactNode } from "react";
import { createHashRouter, Link, RouterProvider, useRouteError } from "react-router-dom";

import { Layout } from "./components/Layout";
import { Alert, Loading, PageHeader } from "./components/ui";
import { PreferencesProvider } from "./lib/prefs";
import { SystemProvider } from "./lib/system";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const ScanSetup = lazy(() => import("./pages/ScanSetup"));
const LiveScan = lazy(() => import("./pages/LiveScan"));
const SurfaceViewer = lazy(() => import("./pages/SurfaceViewer"));
const Calibration = lazy(() => import("./pages/Calibration"));
const Hardware = lazy(() => import("./pages/Hardware"));
const Settings = lazy(() => import("./pages/Settings"));
const ScanHistory = lazy(() => import("./pages/ScanHistory"));

function page(element: ReactNode): ReactNode {
  return <Suspense fallback={<Loading what="page" />}>{element}</Suspense>;
}

function NotFound() {
  return (
    <>
      <PageHeader title="Page not found" />
      <p>
        <Link to="/">Back to the dashboard</Link>
      </p>
    </>
  );
}

function RouteError() {
  const error = useRouteError();
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div style={{ padding: "1.5rem" }}>
      <Alert kind="error" title="This page failed to load">
        <p>{message}</p>
        <p>
          The EMERGENCY STOP endpoint is still available. <a href="./">Reload the application</a>.
        </p>
      </Alert>
    </div>
  );
}

const router = createHashRouter([
  {
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: page(<Dashboard />) },
      { path: "setup", element: page(<ScanSetup />) },
      { path: "live", element: page(<LiveScan />) },
      { path: "live/:scanId", element: page(<LiveScan />) },
      { path: "surface", element: page(<SurfaceViewer />) },
      { path: "surface/:scanId", element: page(<SurfaceViewer />) },
      { path: "calibration", element: page(<Calibration />) },
      { path: "hardware", element: page(<Hardware />) },
      { path: "settings", element: page(<Settings />) },
      { path: "history", element: page(<ScanHistory />) },
      { path: "history/:scanId", element: page(<ScanHistory />) },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

export function App() {
  return (
    <PreferencesProvider>
      <SystemProvider>
        <RouterProvider router={router} />
      </SystemProvider>
    </PreferencesProvider>
  );
}
