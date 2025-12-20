/**
 * PropelAI Word Plugin - Word Document Service
 *
 * Provides services for reading and manipulating Word documents
 * using the Office JavaScript API.
 */

import type {
  DocumentContent,
  Paragraph,
  Heading,
  Table,
  DocumentLocation,
  ComplianceCheck,
} from "../types";

// ============================================================================
// Document Reader Service (Task 4.3.1)
// ============================================================================

/**
 * Read the entire document content
 */
export async function readDocument(): Promise<DocumentContent> {
  return Word.run(async (context) => {
    const body = context.document.body;
    const paragraphs = body.paragraphs;
    const tables = body.tables;

    paragraphs.load("items");
    tables.load("items");

    await context.sync();

    const parsedParagraphs: Paragraph[] = [];
    const headings: Heading[] = [];
    let totalWords = 0;

    for (let i = 0; i < paragraphs.items.length; i++) {
      const para = paragraphs.items[i];
      para.load(["text", "style"]);
    }

    await context.sync();

    for (let i = 0; i < paragraphs.items.length; i++) {
      const para = paragraphs.items[i];
      const text = para.text.trim();
      const style = para.style || "";
      const isHeading = style.toLowerCase().includes("heading");

      if (text) {
        parsedParagraphs.push({
          index: i,
          text,
          style,
          isHeading,
        });

        totalWords += text.split(/\s+/).length;

        if (isHeading) {
          const level = extractHeadingLevel(style);
          headings.push({ index: i, text, level });
        }
      }
    }

    const parsedTables: Table[] = [];
    for (let i = 0; i < tables.items.length; i++) {
      const table = tables.items[i];
      table.load(["rowCount", "values"]);
    }

    await context.sync();

    for (let i = 0; i < tables.items.length; i++) {
      const table = tables.items[i];
      parsedTables.push({
        index: i,
        rows: table.rowCount,
        columns: table.values[0]?.length || 0,
        content: table.values as string[][],
      });
    }

    return {
      paragraphs: parsedParagraphs,
      headings,
      tables: parsedTables,
      totalWords,
    };
  });
}

/**
 * Get selected text in the document
 */
export async function getSelectedText(): Promise<string> {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.load("text");
    await context.sync();
    return selection.text;
  });
}

/**
 * Search for text in the document
 */
export async function searchText(
  searchTerm: string,
  options?: { matchCase?: boolean; matchWholeWord?: boolean }
): Promise<DocumentLocation[]> {
  return Word.run(async (context) => {
    const body = context.document.body;
    const searchResults = body.search(searchTerm, {
      matchCase: options?.matchCase ?? false,
      matchWholeWord: options?.matchWholeWord ?? false,
    });

    searchResults.load("items");
    await context.sync();

    const locations: DocumentLocation[] = [];

    for (let i = 0; i < searchResults.items.length; i++) {
      const range = searchResults.items[i];
      range.load(["text"]);

      // Get paragraph context
      const paragraph = range.paragraphs.getFirst();
      paragraph.load("text");
    }

    await context.sync();

    for (let i = 0; i < searchResults.items.length; i++) {
      locations.push({
        paragraphIndex: i,
        startOffset: 0, // Would need more complex logic for exact offset
        endOffset: searchResults.items[i].text.length,
      });
    }

    return locations;
  });
}

// ============================================================================
// Content Insertion Service (Task 4.3.2)
// ============================================================================

/**
 * Insert text at the current cursor position
 */
export async function insertTextAtCursor(text: string): Promise<void> {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.insertText(text, Word.InsertLocation.replace);
    await context.sync();
  });
}

/**
 * Insert formatted content at cursor
 */
export async function insertFormattedContent(
  content: string,
  format: "plain" | "heading1" | "heading2" | "heading3" | "bullet" | "numbered"
): Promise<void> {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    const insertedRange = selection.insertText(content, Word.InsertLocation.replace);

    switch (format) {
      case "heading1":
        insertedRange.style = "Heading 1";
        break;
      case "heading2":
        insertedRange.style = "Heading 2";
        break;
      case "heading3":
        insertedRange.style = "Heading 3";
        break;
      case "bullet":
        insertedRange.insertParagraph("", Word.InsertLocation.after);
        // Note: Bullet formatting requires additional steps
        break;
      case "numbered":
        insertedRange.insertParagraph("", Word.InsertLocation.after);
        // Note: Numbered list formatting requires additional steps
        break;
    }

    await context.sync();
  });
}

/**
 * Insert a paragraph after the current selection
 */
export async function insertParagraph(text: string): Promise<void> {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.insertParagraph(text, Word.InsertLocation.after);
    await context.sync();
  });
}

/**
 * Insert OOXML content (for rich formatting)
 */
export async function insertOoxml(ooxml: string): Promise<void> {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.insertOoxml(ooxml, Word.InsertLocation.replace);
    await context.sync();
  });
}

/**
 * Insert a comment on the current selection
 */
export async function insertComment(commentText: string): Promise<void> {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.insertComment(commentText);
    await context.sync();
  });
}

// ============================================================================
// Compliance Highlighting Service (Task 4.3.3)
// ============================================================================

/**
 * Highlight text matching a requirement
 */
export async function highlightRequirementMatch(
  requirement: string,
  color: "green" | "yellow" | "red" = "green"
): Promise<number> {
  return Word.run(async (context) => {
    const body = context.document.body;

    // Search for the requirement text (first 50 chars for matching)
    const searchTerm = requirement.substring(0, Math.min(50, requirement.length));
    const searchResults = body.search(searchTerm, {
      matchCase: false,
      matchWholeWord: false,
    });

    searchResults.load("items");
    await context.sync();

    const highlightColors: Record<string, string> = {
      green: "#90EE90", // Light green - compliant
      yellow: "#FFFF99", // Light yellow - partial
      red: "#FFB6C1", // Light pink - missing
    };

    for (const range of searchResults.items) {
      range.font.highlightColor = highlightColors[color];
    }

    await context.sync();
    return searchResults.items.length;
  });
}

/**
 * Highlight multiple compliance results
 */
export async function highlightComplianceResults(
  checks: ComplianceCheck[]
): Promise<void> {
  return Word.run(async (context) => {
    const body = context.document.body;

    for (const check of checks) {
      if (check.matchedText) {
        const searchResults = body.search(check.matchedText.substring(0, 50), {
          matchCase: false,
        });

        searchResults.load("items");
        await context.sync();

        const color =
          check.status === "compliant"
            ? "#90EE90"
            : check.status === "partial"
            ? "#FFFF99"
            : "#FFB6C1";

        for (const range of searchResults.items) {
          range.font.highlightColor = color;
        }
      }
    }

    await context.sync();
  });
}

/**
 * Clear all highlights from the document
 */
export async function clearHighlights(): Promise<void> {
  return Word.run(async (context) => {
    const body = context.document.body;
    body.load("text");
    await context.sync();

    // This is a simplified approach - full implementation would
    // iterate through all ranges and clear highlight colors
    const range = body.getRange();
    range.font.highlightColor = null;

    await context.sync();
  });
}

/**
 * Navigate to a specific location in the document
 */
export async function navigateToLocation(
  searchText: string
): Promise<boolean> {
  return Word.run(async (context) => {
    const body = context.document.body;
    const searchResults = body.search(searchText.substring(0, 50), {
      matchCase: false,
    });

    searchResults.load("items");
    await context.sync();

    if (searchResults.items.length > 0) {
      searchResults.items[0].select();
      await context.sync();
      return true;
    }

    return false;
  });
}

// ============================================================================
// Document Metadata Service
// ============================================================================

/**
 * Get document properties
 */
export async function getDocumentProperties(): Promise<Record<string, string>> {
  return Word.run(async (context) => {
    const properties = context.document.properties;
    properties.load(["title", "author", "subject", "keywords", "comments"]);

    await context.sync();

    return {
      title: properties.title || "",
      author: properties.author || "",
      subject: properties.subject || "",
      keywords: properties.keywords || "",
      comments: properties.comments || "",
    };
  });
}

/**
 * Set a custom document property
 */
export async function setCustomProperty(
  name: string,
  value: string
): Promise<void> {
  return Word.run(async (context) => {
    const customProperties = context.document.properties.customProperties;
    customProperties.add(name, value);
    await context.sync();
  });
}

/**
 * Get a custom document property
 */
export async function getCustomProperty(name: string): Promise<string | null> {
  return Word.run(async (context) => {
    const customProperties = context.document.properties.customProperties;
    const property = customProperties.getItemOrNullObject(name);
    property.load("value");

    await context.sync();

    if (property.isNullObject) {
      return null;
    }

    return property.value as string;
  });
}

// ============================================================================
// Helper Functions
// ============================================================================

function extractHeadingLevel(style: string): number {
  const match = style.match(/Heading\s*(\d)/i);
  return match ? parseInt(match[1], 10) : 1;
}

/**
 * Get word count for selected text
 */
export async function getSelectionWordCount(): Promise<number> {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.load("text");
    await context.sync();

    const text = selection.text.trim();
    if (!text) return 0;

    return text.split(/\s+/).length;
  });
}

/**
 * Replace all occurrences of text
 */
export async function replaceAll(
  searchText: string,
  replaceText: string
): Promise<number> {
  return Word.run(async (context) => {
    const body = context.document.body;
    const searchResults = body.search(searchText, { matchCase: false });

    searchResults.load("items");
    await context.sync();

    for (const range of searchResults.items) {
      range.insertText(replaceText, Word.InsertLocation.replace);
    }

    await context.sync();
    return searchResults.items.length;
  });
}
