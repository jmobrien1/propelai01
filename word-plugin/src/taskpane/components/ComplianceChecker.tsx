/**
 * ComplianceChecker Component (Task 4.2.2)
 *
 * Checks document content against RFP requirements and displays
 * compliance status with highlighting capabilities.
 */

import React, { useState, useCallback } from "react";
import {
  Stack,
  Text,
  PrimaryButton,
  DefaultButton,
  ProgressIndicator,
  MessageBar,
  MessageBarType,
  DetailsList,
  DetailsListLayoutMode,
  SelectionMode,
  IColumn,
  Icon,
  TooltipHost,
  Separator,
} from "@fluentui/react";
import type { ComplianceReport, ComplianceCheck, Requirement } from "../../types";
import { checkCompliance } from "../../services/apiClient";
import {
  readDocument,
  highlightComplianceResults,
  clearHighlights,
  navigateToLocation,
} from "../../services/wordService";

interface ComplianceCheckerProps {
  rfpId: string;
  requirements: Requirement[];
  onSelectRequirement: (requirement: Requirement) => void;
}

export const ComplianceChecker: React.FC<ComplianceCheckerProps> = ({
  rfpId,
  requirements,
  onSelectRequirement,
}) => {
  const [isChecking, setIsChecking] = useState(false);
  const [report, setReport] = useState<ComplianceReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [highlightsApplied, setHighlightsApplied] = useState(false);

  // Run compliance check
  const handleCheck = useCallback(async () => {
    setIsChecking(true);
    setError(null);
    setReport(null);
    setHighlightsApplied(false);

    try {
      // Read document content
      const docContent = await readDocument();
      const fullText = docContent.paragraphs.map((p) => p.text).join("\n");

      // Run compliance check
      const response = await checkCompliance(rfpId, fullText);

      if (response.status === "success" && response.data) {
        setReport(response.data);
      } else {
        setError(response.error || "Compliance check failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsChecking(false);
    }
  }, [rfpId]);

  // Apply highlights to document
  const handleApplyHighlights = useCallback(async () => {
    if (!report) return;

    try {
      await clearHighlights();
      await highlightComplianceResults(report.checks);
      setHighlightsApplied(true);
    } catch (err) {
      setError("Failed to apply highlights");
    }
  }, [report]);

  // Clear highlights
  const handleClearHighlights = useCallback(async () => {
    try {
      await clearHighlights();
      setHighlightsApplied(false);
    } catch (err) {
      setError("Failed to clear highlights");
    }
  }, []);

  // Navigate to matched text
  const handleNavigate = useCallback(async (check: ComplianceCheck) => {
    if (check.matchedText) {
      await navigateToLocation(check.matchedText.substring(0, 50));
    }
  }, []);

  // Get status icon
  const getStatusIcon = (status: ComplianceCheck["status"]) => {
    switch (status) {
      case "compliant":
        return { iconName: "CheckMark", color: "#107c10" };
      case "partial":
        return { iconName: "Warning", color: "#ff8c00" };
      case "missing":
        return { iconName: "Cancel", color: "#d13438" };
      default:
        return { iconName: "Unknown", color: "#605e5c" };
    }
  };

  // Table columns
  const columns: IColumn[] = [
    {
      key: "status",
      name: "",
      minWidth: 24,
      maxWidth: 24,
      onRender: (item: ComplianceCheck) => {
        const { iconName, color } = getStatusIcon(item.status);
        return (
          <TooltipHost content={item.status}>
            <Icon iconName={iconName} style={{ color, fontSize: 16 }} />
          </TooltipHost>
        );
      },
    },
    {
      key: "requirement",
      name: "Requirement",
      minWidth: 200,
      isResizable: true,
      onRender: (item: ComplianceCheck) => (
        <Text
          variant="small"
          style={{ cursor: "pointer" }}
          onClick={() => onSelectRequirement(item.requirement)}
        >
          {item.requirement.text.length > 100
            ? `${item.requirement.text.substring(0, 100)}...`
            : item.requirement.text}
        </Text>
      ),
    },
    {
      key: "confidence",
      name: "Match",
      minWidth: 50,
      maxWidth: 60,
      onRender: (item: ComplianceCheck) => (
        <Text variant="small">{Math.round(item.confidence * 100)}%</Text>
      ),
    },
    {
      key: "action",
      name: "",
      minWidth: 40,
      maxWidth: 40,
      onRender: (item: ComplianceCheck) =>
        item.matchedText ? (
          <TooltipHost content="Navigate to match">
            <Icon
              iconName="NavigateForward"
              style={{ cursor: "pointer", color: "#0078d4" }}
              onClick={() => handleNavigate(item)}
            />
          </TooltipHost>
        ) : null,
    },
  ];

  return (
    <Stack tokens={{ childrenGap: 16 }} style={{ padding: 12 }}>
      {/* Header */}
      <Text variant="large" style={{ fontWeight: 600 }}>
        Compliance Checker
      </Text>

      <Text variant="small" style={{ color: "#605e5c" }}>
        Check your document against RFP requirements to identify gaps and
        ensure full compliance.
      </Text>

      {/* Actions */}
      <Stack horizontal tokens={{ childrenGap: 8 }}>
        <PrimaryButton
          text={isChecking ? "Checking..." : "Run Compliance Check"}
          onClick={handleCheck}
          disabled={isChecking || requirements.length === 0}
          iconProps={{ iconName: "Shield" }}
        />
        {report && (
          <>
            <DefaultButton
              text={highlightsApplied ? "Clear Highlights" : "Apply Highlights"}
              onClick={
                highlightsApplied
                  ? handleClearHighlights
                  : handleApplyHighlights
              }
              iconProps={{
                iconName: highlightsApplied ? "ClearFormatting" : "Highlight",
              }}
            />
          </>
        )}
      </Stack>

      {/* Progress */}
      {isChecking && (
        <ProgressIndicator
          label="Analyzing document..."
          description="Checking against RFP requirements"
        />
      )}

      {/* Error */}
      {error && (
        <MessageBar
          messageBarType={MessageBarType.error}
          onDismiss={() => setError(null)}
        >
          {error}
        </MessageBar>
      )}

      {/* Results Summary */}
      {report && (
        <>
          <Separator />
          <Stack tokens={{ childrenGap: 8 }}>
            <Text variant="mediumPlus" style={{ fontWeight: 600 }}>
              Compliance Summary
            </Text>

            <Stack
              horizontal
              tokens={{ childrenGap: 24 }}
              style={{ padding: "12px 0" }}
            >
              <Stack horizontalAlign="center">
                <Text
                  variant="xxLarge"
                  style={{ color: "#107c10", fontWeight: 600 }}
                >
                  {report.compliant}
                </Text>
                <Text variant="small">Compliant</Text>
              </Stack>
              <Stack horizontalAlign="center">
                <Text
                  variant="xxLarge"
                  style={{ color: "#ff8c00", fontWeight: 600 }}
                >
                  {report.partial}
                </Text>
                <Text variant="small">Partial</Text>
              </Stack>
              <Stack horizontalAlign="center">
                <Text
                  variant="xxLarge"
                  style={{ color: "#d13438", fontWeight: 600 }}
                >
                  {report.missing}
                </Text>
                <Text variant="small">Missing</Text>
              </Stack>
              <Stack horizontalAlign="center">
                <Text variant="xxLarge" style={{ fontWeight: 600 }}>
                  {Math.round(
                    (report.compliant / report.totalRequirements) * 100
                  )}
                  %
                </Text>
                <Text variant="small">Overall</Text>
              </Stack>
            </Stack>

            {/* Detailed Results */}
            <Text variant="medium" style={{ fontWeight: 600, marginTop: 12 }}>
              Details
            </Text>

            <DetailsList
              items={report.checks.sort((a, b) => {
                const order = { missing: 0, partial: 1, compliant: 2, unknown: 3 };
                return order[a.status] - order[b.status];
              })}
              columns={columns}
              layoutMode={DetailsListLayoutMode.justified}
              selectionMode={SelectionMode.none}
              isHeaderVisible={true}
              compact={true}
              styles={{
                root: {
                  maxHeight: 300,
                  overflowY: "auto",
                },
              }}
            />

            {/* Suggestions for missing requirements */}
            {report.missing > 0 && (
              <MessageBar messageBarType={MessageBarType.info}>
                <strong>{report.missing} requirements</strong> need attention.
                Select a missing requirement and use the Draft Assistant to
                generate compliant content.
              </MessageBar>
            )}
          </Stack>
        </>
      )}

      {/* Empty state */}
      {!report && !isChecking && (
        <Stack
          horizontalAlign="center"
          style={{ padding: 24, backgroundColor: "#f3f2f1", borderRadius: 4 }}
        >
          <Icon
            iconName="DocumentSearch"
            style={{ fontSize: 48, color: "#605e5c", marginBottom: 12 }}
          />
          <Text variant="medium">Ready to check compliance</Text>
          <Text
            variant="small"
            style={{ color: "#605e5c", textAlign: "center" }}
          >
            Click "Run Compliance Check" to analyze your document against{" "}
            {requirements.length} RFP requirements
          </Text>
        </Stack>
      )}
    </Stack>
  );
};

export default ComplianceChecker;
