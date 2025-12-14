"""
LLM Client Interfaces and Base Classes

Defines the common interface for all LLM providers in PropelAI's
heterogeneous model mesh architecture.

Model Roles (from Long-Form Generation Strategy):
- Gemini 1.5 Pro: "The Librarian" - 1M token context for RFP ingestion
- Claude 3.5 Sonnet: "The Architect/Critic" - Planning and Blue Team critique
- GPT-4 (fine-tuned): "The Writer" - Prose generation
- Llama 3 (local): "The Verifier" - Quick compliance checks
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, AsyncIterator
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRole(str, Enum):
    """Model roles in the heterogeneous mesh"""
    LIBRARIAN = "librarian"       # Gemini - massive context ingestion
    ARCHITECT = "architect"       # Claude - planning, orchestration
    CRITIC = "critic"             # Claude - Blue Team critique
    WRITER = "writer"             # GPT-4 - prose generation
    VERIFIER = "verifier"         # Llama - quick checks


class TaskType(str, Enum):
    """Types of tasks for model routing"""
    # Ingestion tasks (Librarian)
    RFP_INGESTION = "rfp_ingestion"
    CROSS_REFERENCE_DETECTION = "cross_reference_detection"
    COMPLIANCE_MATRIX_GENERATION = "compliance_matrix_generation"

    # Planning tasks (Architect)
    SECTION_PLANNING = "section_planning"
    OUTLINE_GENERATION = "outline_generation"
    CRITERIA_EXTRACTION = "criteria_extraction"

    # Critique tasks (Critic)
    DRAFT_CRITIQUE = "draft_critique"
    COMPLIANCE_VERIFICATION = "compliance_verification"
    BLUE_TEAM_REVIEW = "blue_team_review"

    # Writing tasks (Writer)
    SECTION_DRAFTING = "section_drafting"
    NARRATIVE_GENERATION = "narrative_generation"
    EXECUTIVE_SUMMARY = "executive_summary"

    # Verification tasks (Verifier)
    FORMAT_CHECK = "format_check"
    P0_CONSTRAINT_CHECK = "p0_constraint_check"
    QUICK_VALIDATION = "quick_validation"


@dataclass
class TokenUsage:
    """Token usage tracking for cost management"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd
        )


@dataclass
class LLMMessage:
    """Standard message format across all providers"""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Standard response format across all providers"""
    content: str
    model: str
    role: ModelRole
    task_type: Optional[TaskType] = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def success(self) -> bool:
        return self.finish_reason in ("stop", "end_turn")


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    max_output_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: List[str] = field(default_factory=list)

    # For long-form generation (combating brevity bias)
    min_output_tokens: Optional[int] = None  # Force minimum length

    # Safety settings
    enable_safety_filters: bool = True


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM providers.

    Each provider must implement:
    - generate(): Single completion
    - generate_stream(): Streaming completion
    - count_tokens(): Token counting for context management
    """

    def __init__(self, role: ModelRole):
        self.role = role
        self._total_usage = TokenUsage()
        self._request_count = 0

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier"""
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Return maximum context window size"""
        pass

    @property
    def total_usage(self) -> TokenUsage:
        """Get cumulative token usage"""
        return self._total_usage

    @property
    def request_count(self) -> int:
        """Get total request count"""
        return self._request_count

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        task_type: Optional[TaskType] = None
    ) -> LLMResponse:
        """
        Generate a completion from the model.

        Args:
            messages: List of conversation messages
            config: Generation configuration
            task_type: Type of task (for logging/routing)

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        task_type: Optional[TaskType] = None
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion from the model.

        Args:
            messages: List of conversation messages
            config: Generation configuration
            task_type: Type of task (for logging/routing)

        Yields:
            String chunks as they're generated
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    def _track_usage(self, usage: TokenUsage) -> None:
        """Track token usage for cost management"""
        self._total_usage = self._total_usage + usage
        self._request_count += 1

        logger.debug(
            f"[{self.model_name}] Request #{self._request_count}: "
            f"{usage.total_tokens} tokens (${usage.estimated_cost_usd:.4f})"
        )

    def reset_usage(self) -> TokenUsage:
        """Reset and return usage statistics"""
        usage = self._total_usage
        self._total_usage = TokenUsage()
        self._request_count = 0
        return usage


# Cost per 1M tokens (as of Dec 2024)
MODEL_COSTS = {
    # Gemini
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # Claude
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},

    # OpenAI
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},

    # Local (free)
    "llama-3": {"input": 0.0, "output": 0.0},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost for a request"""
    costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost
