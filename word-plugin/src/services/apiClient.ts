/**
 * PropelAI Word Plugin - API Client Service
 *
 * Client for communicating with the PropelAI backend API.
 * Handles all RFP, requirements, strategy, and drafting operations.
 */

import type {
  ApiConfig,
  ApiResponse,
  RFP,
  Requirement,
  Strategy,
  Draft,
  ComplianceReport,
  ComplianceCheck,
  GhostingItem,
} from "../types";

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_CONFIG: ApiConfig = {
  baseUrl: "http://localhost:8000/api",
  timeout: 30000,
};

let config: ApiConfig = { ...DEFAULT_CONFIG };

/**
 * Configure the API client
 */
export function configureApi(newConfig: Partial<ApiConfig>): void {
  config = { ...config, ...newConfig };
}

/**
 * Get current API configuration
 */
export function getApiConfig(): ApiConfig {
  return { ...config };
}

// ============================================================================
// HTTP Helpers
// ============================================================================

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  const url = `${config.baseUrl}${endpoint}`;

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  if (config.apiKey) {
    headers["Authorization"] = `Bearer ${config.apiKey}`;
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), config.timeout);

    const response = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return {
        status: "error",
        error: errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    return { status: "success", data };
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        return { status: "error", error: "Request timeout" };
      }
      return { status: "error", error: error.message };
    }
    return { status: "error", error: "Unknown error occurred" };
  }
}

// ============================================================================
// RFP Operations
// ============================================================================

/**
 * Get list of all RFPs
 */
export async function listRFPs(): Promise<ApiResponse<RFP[]>> {
  return request<RFP[]>("/rfp/list");
}

/**
 * Get a specific RFP by ID
 */
export async function getRFP(rfpId: string): Promise<ApiResponse<RFP>> {
  return request<RFP>(`/rfp/${rfpId}`);
}

/**
 * Get requirements for an RFP
 */
export async function getRequirements(
  rfpId: string
): Promise<ApiResponse<Requirement[]>> {
  const response = await request<{ requirements: Requirement[] }>(
    `/rfp/${rfpId}/requirements`
  );

  if (response.status === "success" && response.data) {
    return { status: "success", data: response.data.requirements };
  }
  return { status: "error", error: response.error };
}

/**
 * Get source coordinates for a requirement (Trust Gate)
 */
export async function getRequirementSource(
  rfpId: string,
  requirementId: string
): Promise<ApiResponse<{ source: any; cached: boolean }>> {
  return request(`/rfp/${rfpId}/requirements/${requirementId}/source`);
}

// ============================================================================
// Strategy Operations
// ============================================================================

/**
 * Generate strategy for an RFP
 */
export async function generateStrategy(
  rfpId: string
): Promise<ApiResponse<Strategy>> {
  return request<Strategy>(`/rfp/${rfpId}/strategy`, {
    method: "POST",
  });
}

/**
 * Get existing strategy for an RFP
 */
export async function getStrategy(
  rfpId: string
): Promise<ApiResponse<{ strategy: Strategy }>> {
  return request(`/rfp/${rfpId}/strategy`);
}

/**
 * Run competitive analysis
 */
export async function analyzeCompetitors(
  rfpId: string,
  competitors?: Array<{
    name: string;
    isIncumbent?: boolean;
    strengths?: string[];
    weaknesses?: string[];
  }>
): Promise<ApiResponse<any>> {
  const formData = new FormData();
  if (competitors) {
    formData.append("competitors", JSON.stringify(competitors));
  }

  const response = await fetch(
    `${config.baseUrl}/rfp/${rfpId}/competitive-analysis`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    return { status: "error", error: error.detail || "Analysis failed" };
  }

  const data = await response.json();
  return { status: "success", data };
}

/**
 * Get ghosting language library
 */
export async function getGhostingLibrary(
  rfpId: string
): Promise<ApiResponse<{ ghostingLibrary: GhostingItem[] }>> {
  return request(`/rfp/${rfpId}/ghosting-library`);
}

// ============================================================================
// Drafting Operations
// ============================================================================

/**
 * Generate a draft for a requirement
 */
export async function generateDraft(
  rfpId: string,
  requirementId: string,
  targetWordCount: number = 250
): Promise<ApiResponse<Draft>> {
  const formData = new FormData();
  formData.append("requirement_id", requirementId);
  formData.append("target_word_count", targetWordCount.toString());

  const response = await fetch(`${config.baseUrl}/rfp/${rfpId}/draft`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    return { status: "error", error: error.detail || "Draft generation failed" };
  }

  const data = await response.json();
  return { status: "success", data };
}

/**
 * Get all drafts for an RFP
 */
export async function getDrafts(
  rfpId: string
): Promise<ApiResponse<{ drafts: Record<string, Draft> }>> {
  return request(`/rfp/${rfpId}/drafts`);
}

/**
 * Get a specific draft
 */
export async function getDraft(
  rfpId: string,
  requirementId: string
): Promise<ApiResponse<{ draft: Draft }>> {
  return request(`/rfp/${rfpId}/drafts/${requirementId}`);
}

/**
 * Submit feedback on a draft
 */
export async function submitDraftFeedback(
  rfpId: string,
  requirementId: string,
  feedback: string,
  approved: boolean = false
): Promise<ApiResponse<Draft>> {
  const formData = new FormData();
  formData.append("feedback", feedback);
  formData.append("approved", approved.toString());

  const response = await fetch(
    `${config.baseUrl}/rfp/${rfpId}/drafts/${requirementId}/feedback`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    return { status: "error", error: error.detail || "Feedback submission failed" };
  }

  const data = await response.json();
  return { status: "success", data };
}

// ============================================================================
// Compliance Operations
// ============================================================================

/**
 * Check document compliance against RFP requirements
 */
export async function checkCompliance(
  rfpId: string,
  documentContent: string
): Promise<ApiResponse<ComplianceReport>> {
  // This would typically call a compliance checking endpoint
  // For now, we'll implement client-side matching

  const requirementsResponse = await getRequirements(rfpId);
  if (requirementsResponse.status !== "success" || !requirementsResponse.data) {
    return { status: "error", error: "Failed to fetch requirements" };
  }

  const requirements = requirementsResponse.data;
  const checks: ComplianceCheck[] = [];
  const contentLower = documentContent.toLowerCase();

  for (const req of requirements) {
    // Extract key phrases from requirement
    const keyPhrases = extractKeyPhrases(req.text);
    let matchCount = 0;
    let matchedText = "";

    for (const phrase of keyPhrases) {
      if (contentLower.includes(phrase.toLowerCase())) {
        matchCount++;
        // Find the matched text in original case
        const idx = contentLower.indexOf(phrase.toLowerCase());
        if (!matchedText && idx >= 0) {
          matchedText = documentContent.substring(idx, idx + 100);
        }
      }
    }

    const matchRatio = keyPhrases.length > 0 ? matchCount / keyPhrases.length : 0;

    checks.push({
      requirementId: req.id,
      requirement: req,
      status:
        matchRatio >= 0.7
          ? "compliant"
          : matchRatio >= 0.3
          ? "partial"
          : "missing",
      matchedText: matchedText || undefined,
      confidence: matchRatio,
      suggestions:
        matchRatio < 0.7
          ? [`Address requirement: ${req.text.substring(0, 100)}...`]
          : undefined,
    });
  }

  const report: ComplianceReport = {
    rfpId,
    documentName: "Current Document",
    checkedAt: new Date().toISOString(),
    totalRequirements: requirements.length,
    compliant: checks.filter((c) => c.status === "compliant").length,
    partial: checks.filter((c) => c.status === "partial").length,
    missing: checks.filter((c) => c.status === "missing").length,
    checks,
  };

  return { status: "success", data: report };
}

// ============================================================================
// Document Sync Operations
// ============================================================================

/**
 * Sync document metadata with PropelAI
 */
export async function syncDocument(
  rfpId: string,
  documentData: {
    name: string;
    content: string;
    wordCount: number;
  }
): Promise<ApiResponse<{ synced: boolean }>> {
  // This would sync document state with the server
  // Implementation depends on backend support
  return { status: "success", data: { synced: true } };
}

// ============================================================================
// Health Check
// ============================================================================

/**
 * Check API health
 */
export async function checkHealth(): Promise<
  ApiResponse<{ status: string; version: string }>
> {
  return request("/health");
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Extract key phrases from requirement text for matching
 */
function extractKeyPhrases(text: string): string[] {
  // Remove common words and extract key phrases
  const stopWords = new Set([
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "their",
    "them",
    "we",
    "our",
    "us",
    "you",
    "your",
  ]);

  const words = text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 3 && !stopWords.has(w));

  // Return unique words as key phrases
  return [...new Set(words)].slice(0, 10);
}
