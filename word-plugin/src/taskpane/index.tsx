/**
 * PropelAI Word Plugin - Taskpane Entry Point
 */

import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./taskpane.css";

// Wait for Office to be ready
Office.onReady((info) => {
  if (info.host === Office.HostType.Word) {
    const container = document.getElementById("root");
    if (container) {
      const root = createRoot(container);
      root.render(
        <React.StrictMode>
          <App />
        </React.StrictMode>
      );
    }
  }
});
