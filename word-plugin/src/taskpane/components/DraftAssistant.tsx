/**
 * DraftAssistant Component (Task 4.2.3)
 *
 * AI-powered drafting assistant that generates proposal content
 * using the F-B-P framework and allows insertion into the document.
 */

import React, { useState, useCallback } from "react";
import {
  Stack,
  Text,
  PrimaryButton,
  DefaultButton,
  SpinButton,
  TextField,
  MessageBar,
  MessageBarType,
  ProgressIndicator,
  Separator,
  Label,
  Slider,
  Pivot,
  PivotItem,
  Icon,
  TooltipHost,
  Dropdown,
  IDropdownOption,
} from "@fluentui/react";
import type { Requirement, Draft, WinTheme, GhostingItem } from "../../types";
import {
  generateDraft,
  getDraft,
  submitDraftFeedback,
  getGhostingLibrary,
} from "../../services/apiClient";
import { insertTextAtCursor, insertFormattedContent } from "../../services/wordService";

interface DraftAssistantProps {
  rfpId: string;
  selectedRequirement: Requirement | null;
  winThemes?: WinTheme[];
}

export const DraftAssistant: React.FC<DraftAssistantProps> = ({
  rfpId,
  selectedRequirement,
  winThemes = [],
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isInserting, setIsInserting] = useState(false);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState("");
  const [targetWords, setTargetWords] = useState(250);
  const [ghostingItems, setGhostingItems] = useState<GhostingItem[]>([]);
  const [selectedGhosting, setSelectedGhosting] = useState<string>("");

  // Generate draft
  const handleGenerate = useCallback(async () => {
    if (!selectedRequirement) return;

    setIsGenerating(true);
    setError(null);
    setDraft(null);

    try {
      const response = await generateDraft(
        rfpId,
        selectedRequirement.id,
        targetWords
      );

      if (response.status === "success" && response.data) {
        setDraft(response.data as unknown as Draft);
      } else {
        setError(response.error || "Draft generation failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsGenerating(false);
    }
  }, [rfpId, selectedRequirement, targetWords]);

  // Load existing draft
  const handleLoadDraft = useCallback(async () => {
    if (!selectedRequirement) return;

    try {
      const response = await getDraft(rfpId, selectedRequirement.id);
      if (response.status === "success" && response.data?.draft) {
        setDraft(response.data.draft);
      }
    } catch (err) {
      // No existing draft - that's okay
    }
  }, [rfpId, selectedRequirement]);

  // Submit feedback and revise
  const handleRevise = useCallback(async () => {
    if (!selectedRequirement || !feedback) return;

    setIsGenerating(true);
    setError(null);

    try {
      const response = await submitDraftFeedback(
        rfpId,
        selectedRequirement.id,
        feedback,
        false
      );

      if (response.status === "success" && response.data) {
        setDraft(response.data as unknown as Draft);
        setFeedback("");
      } else {
        setError(response.error || "Revision failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsGenerating(false);
    }
  }, [rfpId, selectedRequirement, feedback]);

  // Approve draft
  const handleApprove = useCallback(async () => {
    if (!selectedRequirement) return;

    try {
      await submitDraftFeedback(
        rfpId,
        selectedRequirement.id,
        "Approved",
        true
      );

      if (draft) {
        setDraft({ ...draft, approved: true });
      }
    } catch (err) {
      setError("Failed to approve draft");
    }
  }, [rfpId, selectedRequirement, draft]);

  // Insert draft into document
  const handleInsert = useCallback(async () => {
    if (!draft?.text) return;

    setIsInserting(true);

    try {
      await insertTextAtCursor(draft.text);
    } catch (err) {
      setError("Failed to insert text");
    } finally {
      setIsInserting(false);
    }
  }, [draft]);

  // Insert with formatting
  const handleInsertFormatted = useCallback(
    async (format: "plain" | "heading2" | "bullet") => {
      if (!draft?.text) return;

      setIsInserting(true);

      try {
        await insertFormattedContent(draft.text, format);
      } catch (err) {
        setError("Failed to insert formatted text");
      } finally {
        setIsInserting(false);
      }
    },
    [draft]
  );

  // Load ghosting library
  const handleLoadGhosting = useCallback(async () => {
    try {
      const response = await getGhostingLibrary(rfpId);
      if (response.status === "success" && response.data?.ghostingLibrary) {
        setGhostingItems(response.data.ghostingLibrary as unknown as GhostingItem[]);
      }
    } catch (err) {
      // Ghosting not available
    }
  }, [rfpId]);

  // Insert ghosting language
  const handleInsertGhosting = useCallback(async () => {
    if (!selectedGhosting) return;

    try {
      await insertTextAtCursor(selectedGhosting);
    } catch (err) {
      setError("Failed to insert ghosting text");
    }
  }, [selectedGhosting]);

  // Quality score color
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "#107c10";
    if (score >= 0.6) return "#ff8c00";
    return "#d13438";
  };

  // Format options dropdown
  const formatOptions: IDropdownOption[] = [
    { key: "plain", text: "Plain Text" },
    { key: "heading2", text: "As Heading" },
    { key: "bullet", text: "As Bullet Point" },
  ];

  return (
    <Stack tokens={{ childrenGap: 16 }} style={{ padding: 12 }}>
      {/* Header */}
      <Text variant="large" style={{ fontWeight: 600 }}>
        Draft Assistant
      </Text>

      {/* Requirement Context */}
      {selectedRequirement ? (
        <Stack
          tokens={{ childrenGap: 4 }}
          style={{
            padding: 12,
            backgroundColor: "#f3f2f1",
            borderRadius: 4,
            borderLeft: "4px solid #0078d4",
          }}
        >
          <Text variant="small" style={{ fontWeight: 600 }}>
            Requirement: {selectedRequirement.id}
          </Text>
          <Text variant="small" style={{ color: "#605e5c" }}>
            {selectedRequirement.text.length > 200
              ? `${selectedRequirement.text.substring(0, 200)}...`
              : selectedRequirement.text}
          </Text>
        </Stack>
      ) : (
        <MessageBar messageBarType={MessageBarType.info}>
          Select a requirement from the Requirements panel to generate draft
          content.
        </MessageBar>
      )}

      {/* Generation Controls */}
      <Stack tokens={{ childrenGap: 12 }}>
        <Stack horizontal tokens={{ childrenGap: 12 }} verticalAlign="end">
          <Stack.Item grow>
            <Label>Target Word Count</Label>
            <Slider
              min={100}
              max={500}
              step={50}
              value={targetWords}
              onChange={(value) => setTargetWords(value)}
              showValue
              valueFormat={(value) => `${value} words`}
            />
          </Stack.Item>
        </Stack>

        <Stack horizontal tokens={{ childrenGap: 8 }}>
          <PrimaryButton
            text={isGenerating ? "Generating..." : "Generate Draft"}
            onClick={handleGenerate}
            disabled={!selectedRequirement || isGenerating}
            iconProps={{ iconName: "Edit" }}
          />
          {selectedRequirement && (
            <DefaultButton
              text="Load Existing"
              onClick={handleLoadDraft}
              iconProps={{ iconName: "Download" }}
            />
          )}
        </Stack>
      </Stack>

      {/* Progress */}
      {isGenerating && (
        <ProgressIndicator
          label="Generating draft..."
          description="Using F-B-P framework with AI"
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

      {/* Draft Result */}
      {draft && (
        <>
          <Separator />
          <Pivot>
            <PivotItem headerText="Draft">
              <Stack tokens={{ childrenGap: 12 }} style={{ paddingTop: 12 }}>
                {/* Draft Text */}
                <TextField
                  multiline
                  rows={8}
                  value={draft.text}
                  readOnly
                  styles={{
                    field: {
                      backgroundColor: "#faf9f8",
                      fontFamily: "Georgia, serif",
                      lineHeight: 1.6,
                    },
                  }}
                />

                {/* Word Count */}
                <Text variant="small" style={{ color: "#605e5c" }}>
                  Word count: {draft.wordCount} / {targetWords} target
                </Text>

                {/* Insert Actions */}
                <Stack horizontal tokens={{ childrenGap: 8 }}>
                  <PrimaryButton
                    text={isInserting ? "Inserting..." : "Insert at Cursor"}
                    onClick={handleInsert}
                    disabled={isInserting}
                    iconProps={{ iconName: "Add" }}
                  />
                  <Dropdown
                    placeholder="Insert as..."
                    options={formatOptions}
                    onChange={(_, option) => {
                      if (option?.key) {
                        handleInsertFormatted(option.key as "plain" | "heading2" | "bullet");
                      }
                    }}
                    styles={{ root: { width: 140 } }}
                  />
                </Stack>
              </Stack>
            </PivotItem>

            <PivotItem headerText="Quality">
              <Stack tokens={{ childrenGap: 12 }} style={{ paddingTop: 12 }}>
                {draft.qualityScores && (
                  <>
                    <Stack horizontal tokens={{ childrenGap: 16 }} wrap>
                      {Object.entries(draft.qualityScores).map(
                        ([key, value]) => (
                          <Stack key={key} horizontalAlign="center">
                            <Text
                              variant="large"
                              style={{
                                color: getScoreColor(value as number),
                                fontWeight: 600,
                              }}
                            >
                              {Math.round((value as number) * 100)}%
                            </Text>
                            <Text variant="tiny" style={{ color: "#605e5c" }}>
                              {key.replace(/([A-Z])/g, " $1").trim()}
                            </Text>
                          </Stack>
                        )
                      )}
                    </Stack>

                    {draft.qualityScores.overall < 0.7 && (
                      <MessageBar messageBarType={MessageBarType.warning}>
                        Quality score is below 70%. Consider revising with
                        specific feedback.
                      </MessageBar>
                    )}
                  </>
                )}
              </Stack>
            </PivotItem>

            <PivotItem headerText="Revise">
              <Stack tokens={{ childrenGap: 12 }} style={{ paddingTop: 12 }}>
                <TextField
                  label="Revision Feedback"
                  placeholder="Provide feedback to improve the draft..."
                  multiline
                  rows={3}
                  value={feedback}
                  onChange={(_, value) => setFeedback(value || "")}
                />

                <Stack horizontal tokens={{ childrenGap: 8 }}>
                  <PrimaryButton
                    text="Revise Draft"
                    onClick={handleRevise}
                    disabled={!feedback || isGenerating}
                    iconProps={{ iconName: "Refresh" }}
                  />
                  <DefaultButton
                    text={draft.approved ? "Approved" : "Approve"}
                    onClick={handleApprove}
                    disabled={draft.approved}
                    iconProps={{
                      iconName: draft.approved ? "CheckMark" : "Accept",
                    }}
                  />
                </Stack>

                {draft.revisionCount > 0 && (
                  <Text variant="small" style={{ color: "#605e5c" }}>
                    Revision #{draft.revisionCount}
                  </Text>
                )}
              </Stack>
            </PivotItem>

            <PivotItem headerText="Ghosting">
              <Stack tokens={{ childrenGap: 12 }} style={{ paddingTop: 12 }}>
                <DefaultButton
                  text="Load Ghosting Library"
                  onClick={handleLoadGhosting}
                  iconProps={{ iconName: "Library" }}
                />

                {ghostingItems.length > 0 ? (
                  <>
                    <Dropdown
                      label="Select ghosting language"
                      placeholder="Choose a ghosting statement..."
                      options={ghostingItems.map((item, idx) => ({
                        key: idx.toString(),
                        text: `${item.ourStrength}: ${item.languageTemplate.substring(0, 60)}...`,
                        data: item.languageTemplate,
                      }))}
                      onChange={(_, option) => {
                        setSelectedGhosting(option?.data || "");
                      }}
                    />

                    {selectedGhosting && (
                      <Stack tokens={{ childrenGap: 8 }}>
                        <TextField
                          value={selectedGhosting}
                          multiline
                          rows={3}
                          readOnly
                          styles={{ field: { backgroundColor: "#faf9f8" } }}
                        />
                        <PrimaryButton
                          text="Insert Ghosting"
                          onClick={handleInsertGhosting}
                          iconProps={{ iconName: "Add" }}
                        />
                      </Stack>
                    )}
                  </>
                ) : (
                  <Text variant="small" style={{ color: "#605e5c" }}>
                    Run competitive analysis to generate ghosting language.
                  </Text>
                )}
              </Stack>
            </PivotItem>
          </Pivot>
        </>
      )}

      {/* Empty state */}
      {!draft && !isGenerating && selectedRequirement && (
        <Stack
          horizontalAlign="center"
          style={{ padding: 24, backgroundColor: "#f3f2f1", borderRadius: 4 }}
        >
          <Icon
            iconName="EditCreate"
            style={{ fontSize: 48, color: "#605e5c", marginBottom: 12 }}
          />
          <Text variant="medium">Ready to generate</Text>
          <Text
            variant="small"
            style={{ color: "#605e5c", textAlign: "center" }}
          >
            Click "Generate Draft" to create AI-powered content for this
            requirement using the F-B-P framework.
          </Text>
        </Stack>
      )}
    </Stack>
  );
};

export default DraftAssistant;
