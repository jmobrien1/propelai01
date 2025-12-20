/**
 * PropelAI Word Plugin - Type Definitions
 */

// ============================================================================
// RFP & Requirements Types
// ============================================================================

export interface Requirement {
  id: string;
  text: string;
  section: string;
  category: "TECHNICAL" | "MANAGEMENT" | "ADMINISTRATIVE" | "EVALUATION";
  priority: "High" | "Medium" | "Low";
  sourceDoc?: string;
  sourcePage?: number;
  sourceCoordinates?: SourceCoordinate;
}

export interface SourceCoordinate {
  documentId: string;
  pageNumber: number;
  boundingBox: BoundingBox;
  textSnippet: string;
  confidence: number;
}

export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  pageWidth: number;
  pageHeight: number;
}

export interface RFP {
  id: string;
  name: string;
  solicitationNumber?: string;
  requirements: Requirement[];
  strategy?: Strategy;
  drafts?: Record<string, Draft>;
}

// ============================================================================
// Strategy Types
// ============================================================================

export interface Strategy {
  winThemes: WinTheme[];
  discriminators: Discriminator[];
  ghostingLibrary: GhostingItem[];
  annotatedOutline?: AnnotatedOutline;
}

export interface WinTheme {
  id: string;
  headline: string;
  narrative: string;
  discriminators: Discriminator[];
  proofPoints: string[];
  priority: number;
}

export interface Discriminator {
  id: string;
  category: "technical" | "management" | "past_performance" | "price" | "innovation";
  claim: string;
  evidenceType: string;
  evidenceSource?: string;
  quantifiedImpact?: string;
  ghostingAngle?: string;
}

export interface GhostingItem {
  competitorWeakness: string;
  ourStrength: string;
  languageTemplate: string;
  evalCriteriaLink?: string;
  subtletyLevel: number;
}

export interface AnnotatedOutline {
  volumes: Volume[];
  totalPageLimit: number;
}

export interface Volume {
  id: string;
  title: string;
  pageAllocation: number;
  sections: OutlineSection[];
}

export interface OutlineSection {
  sectionNumber: string;
  title: string;
  winTheme?: string;
  pageAllocation: number;
  linkedRequirements: string[];
}

// ============================================================================
// Draft Types
// ============================================================================

export interface Draft {
  requirementId: string;
  text: string;
  wordCount: number;
  qualityScores: QualityScores;
  fbpBlocks: FBPBlock[];
  approved: boolean;
  generatedAt: string;
  revisedAt?: string;
  revisionCount: number;
}

export interface QualityScores {
  compliance: number;
  clarity: number;
  citationCoverage: number;
  wordCountRatio: number;
  themeAlignment: number;
  overall: number;
}

export interface FBPBlock {
  feature: {
    description: string;
    technicalDetail: string;
  };
  benefit: {
    statement: string;
    quantifiedImpact?: string;
  };
  proofs: Proof[];
}

export interface Proof {
  type: "past_performance" | "case_study" | "metric" | "certification";
  summary: string;
  sourceDocument?: string;
}

// ============================================================================
// Compliance Types
// ============================================================================

export interface ComplianceCheck {
  requirementId: string;
  requirement: Requirement;
  status: "compliant" | "partial" | "missing" | "unknown";
  matchedText?: string;
  matchLocation?: DocumentLocation;
  confidence: number;
  suggestions?: string[];
}

export interface DocumentLocation {
  paragraphIndex: number;
  startOffset: number;
  endOffset: number;
  pageNumber?: number;
}

export interface ComplianceReport {
  rfpId: string;
  documentName: string;
  checkedAt: string;
  totalRequirements: number;
  compliant: number;
  partial: number;
  missing: number;
  checks: ComplianceCheck[];
}

// ============================================================================
// Document Types
// ============================================================================

export interface DocumentContent {
  paragraphs: Paragraph[];
  headings: Heading[];
  tables: Table[];
  totalWords: number;
}

export interface Paragraph {
  index: number;
  text: string;
  style: string;
  isHeading: boolean;
}

export interface Heading {
  index: number;
  text: string;
  level: number;
}

export interface Table {
  index: number;
  rows: number;
  columns: number;
  content: string[][];
}

// ============================================================================
// API Types
// ============================================================================

export interface ApiConfig {
  baseUrl: string;
  apiKey?: string;
  timeout: number;
}

export interface ApiResponse<T> {
  status: "success" | "error";
  data?: T;
  error?: string;
}

// ============================================================================
// UI State Types
// ============================================================================

export interface AppState {
  currentRfp: RFP | null;
  selectedRequirement: Requirement | null;
  complianceReport: ComplianceReport | null;
  isLoading: boolean;
  error: string | null;
  activeTab: "requirements" | "compliance" | "drafting" | "strategy";
}

export type AppAction =
  | { type: "SET_RFP"; payload: RFP }
  | { type: "SELECT_REQUIREMENT"; payload: Requirement | null }
  | { type: "SET_COMPLIANCE_REPORT"; payload: ComplianceReport }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_ACTIVE_TAB"; payload: AppState["activeTab"] };
