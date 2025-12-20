/**
 * RequirementPanel Component (Task 4.2.1)
 *
 * Displays RFP requirements and allows selection for drafting.
 * Supports filtering, search, and navigation to source documents.
 */

import React, { useState, useMemo } from "react";
import {
  Stack,
  Text,
  SearchBox,
  Dropdown,
  IDropdownOption,
  DetailsList,
  DetailsListLayoutMode,
  SelectionMode,
  IColumn,
  IconButton,
  TooltipHost,
  MessageBar,
  MessageBarType,
  Spinner,
  SpinnerSize,
} from "@fluentui/react";
import type { Requirement } from "../../types";
import { getRequirementSource } from "../../services/apiClient";
import { navigateToLocation, highlightRequirementMatch } from "../../services/wordService";

interface RequirementPanelProps {
  requirements: Requirement[];
  rfpId: string;
  onSelectRequirement: (requirement: Requirement) => void;
  selectedRequirementId?: string;
  isLoading?: boolean;
}

const categoryOptions: IDropdownOption[] = [
  { key: "ALL", text: "All Categories" },
  { key: "TECHNICAL", text: "Technical" },
  { key: "MANAGEMENT", text: "Management" },
  { key: "ADMINISTRATIVE", text: "Administrative" },
  { key: "EVALUATION", text: "Evaluation" },
];

const priorityOptions: IDropdownOption[] = [
  { key: "ALL", text: "All Priorities" },
  { key: "High", text: "High" },
  { key: "Medium", text: "Medium" },
  { key: "Low", text: "Low" },
];

export const RequirementPanel: React.FC<RequirementPanelProps> = ({
  requirements,
  rfpId,
  onSelectRequirement,
  selectedRequirementId,
  isLoading = false,
}) => {
  const [searchText, setSearchText] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");
  const [priorityFilter, setPriorityFilter] = useState<string>("ALL");
  const [loadingSource, setLoadingSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Filter requirements based on search and filters
  const filteredRequirements = useMemo(() => {
    return requirements.filter((req) => {
      const matchesSearch =
        !searchText ||
        req.text.toLowerCase().includes(searchText.toLowerCase()) ||
        req.id.toLowerCase().includes(searchText.toLowerCase()) ||
        req.section.toLowerCase().includes(searchText.toLowerCase());

      const matchesCategory =
        categoryFilter === "ALL" || req.category === categoryFilter;

      const matchesPriority =
        priorityFilter === "ALL" || req.priority === priorityFilter;

      return matchesSearch && matchesCategory && matchesPriority;
    });
  }, [requirements, searchText, categoryFilter, priorityFilter]);

  // Handle viewing source document
  const handleViewSource = async (requirement: Requirement) => {
    setLoadingSource(requirement.id);
    setError(null);

    try {
      const response = await getRequirementSource(rfpId, requirement.id);
      if (response.status === "success" && response.data?.source) {
        // Highlight the requirement text in the current document
        const matches = await highlightRequirementMatch(
          requirement.text.substring(0, 50),
          "yellow"
        );

        if (matches === 0) {
          setError("Requirement text not found in current document");
        }
      } else {
        setError("Source location not available");
      }
    } catch (err) {
      setError("Failed to locate source");
    } finally {
      setLoadingSource(null);
    }
  };

  // Handle finding requirement in document
  const handleFindInDoc = async (requirement: Requirement) => {
    const found = await navigateToLocation(requirement.text.substring(0, 50));
    if (!found) {
      setError("Text not found in document. Try drafting content for this requirement.");
    }
  };

  // Table columns
  const columns: IColumn[] = [
    {
      key: "id",
      name: "ID",
      fieldName: "id",
      minWidth: 60,
      maxWidth: 80,
      isResizable: true,
      onRender: (item: Requirement) => (
        <Text variant="small" style={{ fontFamily: "monospace" }}>
          {item.id.substring(0, 8)}
        </Text>
      ),
    },
    {
      key: "text",
      name: "Requirement",
      fieldName: "text",
      minWidth: 200,
      isResizable: true,
      isMultiline: true,
      onRender: (item: Requirement) => (
        <Text
          variant="small"
          style={{
            cursor: "pointer",
            fontWeight: item.id === selectedRequirementId ? 600 : 400,
          }}
          onClick={() => onSelectRequirement(item)}
        >
          {item.text.length > 150 ? `${item.text.substring(0, 150)}...` : item.text}
        </Text>
      ),
    },
    {
      key: "priority",
      name: "Priority",
      fieldName: "priority",
      minWidth: 60,
      maxWidth: 70,
      onRender: (item: Requirement) => {
        const colors = {
          High: "#d13438",
          Medium: "#ff8c00",
          Low: "#107c10",
        };
        return (
          <Text
            variant="small"
            style={{ color: colors[item.priority], fontWeight: 600 }}
          >
            {item.priority}
          </Text>
        );
      },
    },
    {
      key: "actions",
      name: "",
      minWidth: 60,
      maxWidth: 60,
      onRender: (item: Requirement) => (
        <Stack horizontal tokens={{ childrenGap: 4 }}>
          <TooltipHost content="Find in document">
            <IconButton
              iconProps={{ iconName: "Search" }}
              onClick={() => handleFindInDoc(item)}
              ariaLabel="Find in document"
            />
          </TooltipHost>
          <TooltipHost content="View source">
            <IconButton
              iconProps={{ iconName: "DocumentSearch" }}
              onClick={() => handleViewSource(item)}
              disabled={loadingSource === item.id}
              ariaLabel="View source"
            />
          </TooltipHost>
        </Stack>
      ),
    },
  ];

  if (isLoading) {
    return (
      <Stack
        horizontalAlign="center"
        verticalAlign="center"
        style={{ padding: 40 }}
      >
        <Spinner size={SpinnerSize.large} label="Loading requirements..." />
      </Stack>
    );
  }

  return (
    <Stack tokens={{ childrenGap: 12 }} style={{ padding: 12 }}>
      {/* Header */}
      <Text variant="large" style={{ fontWeight: 600 }}>
        RFP Requirements
      </Text>

      {/* Search and Filters */}
      <Stack tokens={{ childrenGap: 8 }}>
        <SearchBox
          placeholder="Search requirements..."
          value={searchText}
          onChange={(_, value) => setSearchText(value || "")}
          styles={{ root: { width: "100%" } }}
        />

        <Stack horizontal tokens={{ childrenGap: 8 }}>
          <Dropdown
            placeholder="Category"
            options={categoryOptions}
            selectedKey={categoryFilter}
            onChange={(_, option) =>
              setCategoryFilter(option?.key as string || "ALL")
            }
            styles={{ root: { width: 120 } }}
          />
          <Dropdown
            placeholder="Priority"
            options={priorityOptions}
            selectedKey={priorityFilter}
            onChange={(_, option) =>
              setPriorityFilter(option?.key as string || "ALL")
            }
            styles={{ root: { width: 100 } }}
          />
        </Stack>
      </Stack>

      {/* Error Message */}
      {error && (
        <MessageBar
          messageBarType={MessageBarType.warning}
          onDismiss={() => setError(null)}
          dismissButtonAriaLabel="Close"
        >
          {error}
        </MessageBar>
      )}

      {/* Stats */}
      <Text variant="small" style={{ color: "#605e5c" }}>
        Showing {filteredRequirements.length} of {requirements.length} requirements
      </Text>

      {/* Requirements List */}
      <DetailsList
        items={filteredRequirements}
        columns={columns}
        layoutMode={DetailsListLayoutMode.justified}
        selectionMode={SelectionMode.single}
        isHeaderVisible={true}
        compact={true}
        styles={{
          root: {
            maxHeight: 400,
            overflowY: "auto",
          },
        }}
      />
    </Stack>
  );
};

export default RequirementPanel;
