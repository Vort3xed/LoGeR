import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate, Link } from "react-router-dom";
import "./styles.css";

import Viewer from "./pages/Viewer";
import Camera from "./pages/Camera";

function Home() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4">
      <h1 className="text-3xl font-semibold">LoGeR Real-Time</h1>
      <div className="flex gap-3">
        <Link to="/camera"
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500">
          /camera
        </Link>
        <Link to="/viewer"
              className="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500">
          /viewer
        </Link>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/camera" element={<Camera />} />
        <Route path="/viewer" element={<Viewer />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
