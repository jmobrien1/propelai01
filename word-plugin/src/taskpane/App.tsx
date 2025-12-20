/**
 * PropelAI Word Plugin - Main Application Component
 *
 * Root component that orchestrates the taskpane UI with
 * navigation between Requirements, Compliance, and Drafting views.
 */

import React, { useState, useEffect, useCallback, useReducer } from "react";
import {
  Stack,
  Pivot,
  PivotItem,
  MessageBar,
  MessageBarType,
  Spinner,
  SpinnerSize,
  Dropdown,
  IDropdownOption,
  Text,
  initializeIcons,
} from "@fluentui/react";
import { RequirementPanel } from "./components/RequirementPanel";
import { ComplianceChecker } from "./components/ComplianceChecker";
import { DraftAssistant } from "./components/DraftAssistant";
import type { AppState, AppAction, RFP, Requirement } from "../types";
import { listRFPs, getRFP, checkHealth } from "../services/apiClient";

// Initialize Fluent UI icons
initializeIcons();

// ============================================================================
// State Management
// ============================================================================

const initialState: AppState = {
  currentRfp: null,
  selectedRequirement: null,
  complianceReport: null,
  isLoading: false,
  error: null,
  activeTab: "requirements",
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_RFP":
      return { ...state, currentRfp: action.payload, selectedRequirement: null };
    case "SELECT_REQUIREMENT":
      return { ...state, selectedRequirement: action.payload };
    case "SET_COMPLIANCE_REPORT":
      return { ...state, complianceReport: action.payload };
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "SET_ACTIVE_TAB":
      return { ...state, activeTab: action.payload };
    default:
      return state;
  }
}

// ============================================================================
// Main App Component
// ============================================================================

export const App: React.FC = () => {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const [rfpList, setRfpList] = useState<RFP[]>([]);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);

  // Check API connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await checkHealth();
        setIsConnected(response.status === "success");
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
  }, []);

  // Load RFP list
  useEffect(() => {
    const loadRFPs = async () => {
      dispatch({ type: "SET_LOADING", payload: true });

      try {
        const response = await listRFPs();
        if (response.status === "success" && response.data) {
          setRfpList(response.data);
        }
      } catch (err) {
        dispatch({
          type: "SET_ERROR",
          payload: "Failed to load RFP list",
        });
      } finally {
        dispatch({ type: "SET_LOADING", payload: false });
      }
    };

    if (isConnected) {
      loadRFPs();
    }
  }, [isConnected]);

  // Handle RFP selection
  const handleRfpSelect = useCallback(async (rfpId: string) => {
    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });

    try {
      const response = await getRFP(rfpId);
      if (response.status === "success" && response.data) {
        dispatch({ type: "SET_RFP", payload: response.data });
      } else {
        dispatch({
          type: "SET_ERROR",
          payload: response.error || "Failed to load RFP",
        });
      }
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        payload: err instanceof Error ? err.message : "Unknown error",
      });
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  }, []);

  // Handle requirement selection
  const handleSelectRequirement = useCallback((requirement: Requirement) => {
    dispatch({ type: "SELECT_REQUIREMENT", payload: requirement });
    dispatch({ type: "SET_ACTIVE_TAB", payload: "drafting" });
  }, []);

  // RFP dropdown options
  const rfpOptions: IDropdownOption[] = rfpList.map((rfp) => ({
    key: rfp.id,
    text: rfp.name || rfp.id,
  }));

  // Connection status
  if (isConnected === null) {
    return (
      <Stack
        horizontalAlign="center"
        verticalAlign="center"
        style={{ height: "100vh" }}
      >
        <Spinner size={SpinnerSize.large} label="Connecting to PropelAI..." />
      </Stack>
    );
  }

  if (!isConnected) {
    return (
      <Stack style={{ padding: 20 }}>
        <MessageBar messageBarType={MessageBarType.error}>
          Unable to connect to PropelAI API. Please ensure the server is running
          at http://localhost:8000
        </MessageBar>
        <Text style={{ marginTop: 16 }}>
          To start the server, run:
          <pre style={{ backgroundColor: "#f3f2f1", padding: 8, marginTop: 8 }}>
            cd propelai01 && uvicorn api.main:app --reload
          </pre>
        </Text>
      </Stack>
    );
  }

  return (
    <Stack
      style={{ height: "100vh", overflow: "hidden" }}
      tokens={{ childrenGap: 0 }}
    >
      {/* Header */}
      <Stack
        style={{
          padding: "12px 16px",
          backgroundColor: "#0078d4",
          color: "white",
        }}
      >
        <Text
          variant="large"
          style={{ color: "white", fontWeight: 600 }}
        >
          PropelAI
        </Text>
        <Text variant="small" style={{ color: "rgba(255,255,255,0.8)" }}>
          Proposal Intelligence Assistant
        </Text>
      </Stack>

      {/* RFP Selector */}
      <Stack style={{ padding: "12px 16px", borderBottom: "1px solid #edebe9" }}>
        <Dropdown
          label="Select RFP"
          placeholder="Choose an RFP to work with..."
          options={rfpOptions}
          selectedKey={state.currentRfp?.id}
          onChange={(_, option) => {
            if (option?.key) {
              handleRfpSelect(option.key as string);
            }
          }}
          disabled={state.isLoading}
        />
      </Stack>

      {/* Error Message */}
      {state.error && (
        <MessageBar
          messageBarType={MessageBarType.error}
          onDismiss={() => dispatch({ type: "SET_ERROR", payload: null })}
        >
          {state.error}
        </MessageBar>
      )}

      {/* Loading State */}
      {state.isLoading && (
        <Stack
          horizontalAlign="center"
          style={{ padding: 20 }}
        >
          <Spinner size={SpinnerSize.medium} label="Loading..." />
        </Stack>
      )}

      {/* Main Content */}
      {state.currentRfp && !state.isLoading && (
        <Stack style={{ flex: 1, overflow: "hidden" }}>
          <Pivot
            selectedKey={state.activeTab}
            onLinkClick={(item) => {
              if (item?.props.itemKey) {
                dispatch({
                  type: "SET_ACTIVE_TAB",
                  payload: item.props.itemKey as AppState["activeTab"],
                });
              }
            }}
            style={{ paddingLeft: 16 }}
          >
            <PivotItem headerText="Requirements" itemKey="requirements">
              <Stack style={{ height: "calc(100vh - 220px)", overflow: "auto" }}>
                <RequirementPanel
                  requirements={state.currentRfp.requirements}
                  rfpId={state.currentRfp.id}
                  onSelectRequirement={handleSelectRequirement}
                  selectedRequirementId={state.selectedRequirement?.id}
                />
              </Stack>
            </PivotItem>

            <PivotItem headerText="Compliance" itemKey="compliance">
              <Stack style={{ height: "calc(100vh - 220px)", overflow: "auto" }}>
                <ComplianceChecker
                  rfpId={state.currentRfp.id}
                  requirements={state.currentRfp.requirements}
                  onSelectRequirement={handleSelectRequirement}
                />
              </Stack>
            </PivotItem>

            <PivotItem headerText="Drafting" itemKey="drafting">
              <Stack style={{ height: "calc(100vh - 220px)", overflow: "auto" }}>
                <DraftAssistant
                  rfpId={state.currentRfp.id}
                  selectedRequirement={state.selectedRequirement}
                  winThemes={state.currentRfp.strategy?.winThemes}
                />
              </Stack>
            </PivotItem>
          </Pivot>
        </Stack>
      )}

      {/* Empty State */}
      {!state.currentRfp && !state.isLoading && (
        <Stack
          horizontalAlign="center"
          verticalAlign="center"
          style={{ flex: 1, padding: 40 }}
        >
          <Text variant="large" style={{ marginBottom: 12, color: "#605e5c" }}>
            Welcome to PropelAI
          </Text>
          <Text style={{ textAlign: "center", color: "#a19f9d" }}>
            Select an RFP from the dropdown above to begin working on your
            proposal.
          </Text>
        </Stack>
      )}

      {/* Footer */}
      <Stack
        style={{
          padding: "8px 16px",
          borderTop: "1px solid #edebe9",
          backgroundColor: "#faf9f8",
        }}
      >
        <Text variant="tiny" style={{ color: "#a19f9d" }}>
          PropelAI v4.0 | Connected to API
        </Text>
      </Stack>
    </Stack>
  );
};

export default App;
