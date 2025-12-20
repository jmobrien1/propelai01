/**
 * PropelAI Word Plugin - Ribbon Commands (Task 4.6.1)
 *
 * Implements ribbon button actions that execute without
 * opening the taskpane.
 */

import {
  readDocument,
  highlightComplianceResults,
  clearHighlights,
  insertTextAtCursor,
  setCustomProperty,
  getCustomProperty,
} from "../services/wordService";
import { checkCompliance, generateDraft, getRFP } from "../services/apiClient";
import type { ComplianceCheck } from "../types";

// ============================================================================
// Command Handlers
// ============================================================================

/**
 * Check Compliance Command
 *
 * Analyzes the current document against the linked RFP and
 * highlights compliant/non-compliant sections.
 */
async function checkComplianceCommand(event: Office.AddinCommands.Event): Promise<void> {
  try {
    // Get linked RFP ID from document properties
    const rfpId = await getCustomProperty("PropelAI_RFP_ID");

    if (!rfpId) {
      showNotification(
        "No RFP Linked",
        "Please open the PropelAI panel and select an RFP first.",
        "warning"
      );
      event.completed();
      return;
    }

    showNotification("Checking Compliance", "Analyzing document...", "info");

    // Read document content
    const docContent = await readDocument();
    const fullText = docContent.paragraphs.map((p) => p.text).join("\n");

    // Run compliance check
    const response = await checkCompliance(rfpId, fullText);

    if (response.status === "success" && response.data) {
      const report = response.data;

      // Apply highlights
      await clearHighlights();
      await highlightComplianceResults(report.checks);

      const message = `Compliance: ${report.compliant}/${report.totalRequirements} compliant, ${report.missing} missing`;

      showNotification("Compliance Check Complete", message, "success");
    } else {
      showNotification(
        "Compliance Check Failed",
        response.error || "Unknown error",
        "error"
      );
    }
  } catch (error) {
    showNotification(
      "Error",
      error instanceof Error ? error.message : "Unknown error",
      "error"
    );
  }

  event.completed();
}

/**
 * Insert Draft Command
 *
 * Generates and inserts draft content for the requirement
 * nearest to the current cursor position.
 */
async function insertDraftCommand(event: Office.AddinCommands.Event): Promise<void> {
  try {
    // Get linked RFP ID
    const rfpId = await getCustomProperty("PropelAI_RFP_ID");

    if (!rfpId) {
      showNotification(
        "No RFP Linked",
        "Please open the PropelAI panel and select an RFP first.",
        "warning"
      );
      event.completed();
      return;
    }

    // Get selected requirement ID (if set)
    const requirementId = await getCustomProperty("PropelAI_Selected_Requirement");

    if (!requirementId) {
      showNotification(
        "No Requirement Selected",
        "Please select a requirement in the PropelAI panel first.",
        "warning"
      );
      event.completed();
      return;
    }

    showNotification("Generating Draft", "Creating content...", "info");

    // Generate draft
    const response = await generateDraft(rfpId, requirementId, 250);

    if (response.status === "success" && response.data) {
      const draft = response.data;

      // Insert at cursor
      await insertTextAtCursor((draft as any).draft || (draft as any).text || "");

      showNotification(
        "Draft Inserted",
        `Inserted ${(draft as any).word_count || (draft as any).wordCount || 0} words`,
        "success"
      );
    } else {
      showNotification(
        "Draft Generation Failed",
        response.error || "Unknown error",
        "error"
      );
    }
  } catch (error) {
    showNotification(
      "Error",
      error instanceof Error ? error.message : "Unknown error",
      "error"
    );
  }

  event.completed();
}

/**
 * Sync with RFP Command
 *
 * Synchronizes the current document with the PropelAI RFP,
 * updating section markers and requirement mappings.
 */
async function syncWithRFPCommand(event: Office.AddinCommands.Event): Promise<void> {
  try {
    // Get linked RFP ID
    const rfpId = await getCustomProperty("PropelAI_RFP_ID");

    if (!rfpId) {
      showNotification(
        "No RFP Linked",
        "Please open the PropelAI panel and select an RFP first.",
        "warning"
      );
      event.completed();
      return;
    }

    showNotification("Syncing", "Synchronizing with RFP...", "info");

    // Get RFP data
    const response = await getRFP(rfpId);

    if (response.status === "success" && response.data) {
      const rfp = response.data;

      // Store sync timestamp
      await setCustomProperty(
        "PropelAI_Last_Sync",
        new Date().toISOString()
      );

      showNotification(
        "Sync Complete",
        `Synced with RFP: ${rfp.name} (${rfp.requirements.length} requirements)`,
        "success"
      );
    } else {
      showNotification(
        "Sync Failed",
        response.error || "Unknown error",
        "error"
      );
    }
  } catch (error) {
    showNotification(
      "Error",
      error instanceof Error ? error.message : "Unknown error",
      "error"
    );
  }

  event.completed();
}

// ============================================================================
// Notification Helper
// ============================================================================

type NotificationType = "info" | "success" | "warning" | "error";

function showNotification(
  title: string,
  message: string,
  type: NotificationType
): void {
  // Use Office notification API if available
  if (Office.context.mailbox?.item?.notificationMessages) {
    // Outlook-specific notifications
    Office.context.mailbox.item.notificationMessages.replaceAsync(
      "propelai-notification",
      {
        type: Office.MailboxEnums.ItemNotificationMessageType.InformationalMessage,
        message: `${title}: ${message}`,
        icon: "Icon.16x16",
        persistent: false,
      }
    );
  } else {
    // Fallback: log to console (Word doesn't have built-in notifications)
    const emoji = {
      info: "ℹ️",
      success: "✅",
      warning: "⚠️",
      error: "❌",
    };

    console.log(`${emoji[type]} PropelAI: ${title} - ${message}`);

    // For Word, we could insert a content control with the notification
    // but that's intrusive. Instead, notifications appear in taskpane.
  }
}

// ============================================================================
// Register Commands
// ============================================================================

// Register the command functions globally so Office can call them
Office.onReady(() => {
  // Make functions available to the manifest
  (globalThis as any).checkCompliance = checkComplianceCommand;
  (globalThis as any).insertDraft = insertDraftCommand;
  (globalThis as any).syncWithRFP = syncWithRFPCommand;
});

// Export for testing
export { checkComplianceCommand, insertDraftCommand, syncWithRFPCommand };
