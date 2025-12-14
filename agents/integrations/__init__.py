"""
PropelAI External Integrations

Third-party service integrations for enhanced document processing
and heterogeneous model mesh for long-form proposal generation.

Model Mesh Architecture (from Long-Form Generation Strategy):
- Gemini 1.5 Pro: "The Librarian" - 1M token context for RFP ingestion
- Claude 3.5 Sonnet: "The Architect/Critic" - Planning and Blue Team critique
- GPT-4 (fine-tuned): "The Writer" - Prose generation
- Llama 3 (local): "The Verifier" - Quick compliance checks
"""

from .tensorlake_processor import TensorlakeProcessor, TensorlakeConfig

# LLM Client Base Classes
from .llm_clients import (
    BaseLLMClient,
    ModelRole,
    TaskType,
    TokenUsage,
    LLMMessage,
    LLMResponse,
    GenerationConfig,
    calculate_cost,
    MODEL_COSTS,
)

# Gemini Client (The Librarian)
from .gemini_client import (
    GeminiClient,
    RFPIngestionResult,
    ComplianceMatrixEntry,
)

# Claude Client (The Architect / The Critic)
from .claude_client import (
    ClaudeClient,
    CritiqueResult,
    SectionPlan,
    OutlinePlanResult,
)

# Model Router (Intelligent Dispatch)
from .model_router import (
    ModelRouter,
    RoutingStrategy,
    RoutingConfig,
    RoutingDecision,
    create_router,
    TASK_ROUTING,
)

__all__ = [
    # Tensorlake
    "TensorlakeProcessor",
    "TensorlakeConfig",
    # LLM Base
    "BaseLLMClient",
    "ModelRole",
    "TaskType",
    "TokenUsage",
    "LLMMessage",
    "LLMResponse",
    "GenerationConfig",
    "calculate_cost",
    "MODEL_COSTS",
    # Gemini
    "GeminiClient",
    "RFPIngestionResult",
    "ComplianceMatrixEntry",
    # Claude
    "ClaudeClient",
    "CritiqueResult",
    "SectionPlan",
    "OutlinePlanResult",
    # Model Router
    "ModelRouter",
    "RoutingStrategy",
    "RoutingConfig",
    "RoutingDecision",
    "create_router",
    "TASK_ROUTING",
]
