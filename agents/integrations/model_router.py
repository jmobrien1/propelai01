"""
Model Router - Intelligent Task-Based Dispatch

Routes tasks to the optimal model based on:
- Task type (ingestion, planning, critique, writing, verification)
- Context size requirements
- Cost optimization
- Availability/fallback handling

From Long-Form Generation Strategy:
"The key insight is that different models excel at different aspects of
proposal generation. Route each subtask to its specialist."
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from .llm_clients import (
    BaseLLMClient,
    ModelRole,
    TaskType,
    TokenUsage,
    LLMMessage,
    LLMResponse,
    GenerationConfig,
)
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Strategy for model selection"""
    OPTIMAL = "optimal"      # Best model for task (default)
    COST_AWARE = "cost_aware"  # Balance quality vs cost
    SPEED = "speed"          # Fastest available
    FALLBACK = "fallback"    # Use when primary unavailable


@dataclass
class RoutingConfig:
    """Configuration for the model router"""
    strategy: RoutingStrategy = RoutingStrategy.OPTIMAL
    max_cost_per_request: Optional[float] = None  # USD limit
    enable_fallback: bool = True
    log_routing_decisions: bool = True


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_model: ModelRole
    reason: str
    fallback_models: List[ModelRole]
    estimated_cost: Optional[float] = None


# Task to Model Role mapping (optimal routing)
TASK_ROUTING = {
    # Librarian tasks (Gemini - massive context)
    TaskType.RFP_INGESTION: ModelRole.LIBRARIAN,
    TaskType.CROSS_REFERENCE_DETECTION: ModelRole.LIBRARIAN,
    TaskType.COMPLIANCE_MATRIX_GENERATION: ModelRole.LIBRARIAN,

    # Architect tasks (Claude - planning/structure)
    TaskType.SECTION_PLANNING: ModelRole.ARCHITECT,
    TaskType.OUTLINE_GENERATION: ModelRole.ARCHITECT,
    TaskType.CRITERIA_EXTRACTION: ModelRole.ARCHITECT,

    # Critic tasks (Claude - adversarial review)
    TaskType.DRAFT_CRITIQUE: ModelRole.CRITIC,
    TaskType.COMPLIANCE_VERIFICATION: ModelRole.CRITIC,
    TaskType.BLUE_TEAM_REVIEW: ModelRole.CRITIC,

    # Writer tasks (GPT-4 fine-tuned - prose)
    TaskType.SECTION_DRAFTING: ModelRole.WRITER,
    TaskType.NARRATIVE_GENERATION: ModelRole.WRITER,
    TaskType.EXECUTIVE_SUMMARY: ModelRole.WRITER,

    # Verifier tasks (Llama - quick checks)
    TaskType.FORMAT_CHECK: ModelRole.VERIFIER,
    TaskType.P0_CONSTRAINT_CHECK: ModelRole.VERIFIER,
    TaskType.QUICK_VALIDATION: ModelRole.VERIFIER,
}

# Fallback chain when primary model unavailable
FALLBACK_CHAIN = {
    ModelRole.LIBRARIAN: [ModelRole.ARCHITECT, ModelRole.WRITER],
    ModelRole.ARCHITECT: [ModelRole.CRITIC, ModelRole.WRITER],
    ModelRole.CRITIC: [ModelRole.ARCHITECT, ModelRole.WRITER],
    ModelRole.WRITER: [ModelRole.ARCHITECT, ModelRole.CRITIC],
    ModelRole.VERIFIER: [ModelRole.CRITIC, ModelRole.ARCHITECT],
}


class ModelRouter:
    """
    Intelligent router that dispatches tasks to optimal models.

    The router maintains a pool of model clients and routes requests
    based on task type, context size, and cost constraints.

    Usage:
        router = ModelRouter()
        router.initialize()

        # Route a task to optimal model
        response = await router.route(
            task_type=TaskType.SECTION_PLANNING,
            messages=[LLMMessage(role="user", content="Plan this section...")],
        )
    """

    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self._clients: Dict[ModelRole, BaseLLMClient] = {}
        self._initialized = False
        self._total_usage = TokenUsage()

    @property
    def total_usage(self) -> TokenUsage:
        """Get cumulative usage across all models"""
        return self._total_usage

    def initialize(
        self,
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize model clients.

        Args:
            gemini_api_key: Google API key (or from GOOGLE_API_KEY env)
            anthropic_api_key: Anthropic API key (or from ANTHROPIC_API_KEY env)
            openai_api_key: OpenAI API key (or from OPENAI_API_KEY env)
        """
        # Initialize Gemini (Librarian)
        try:
            self._clients[ModelRole.LIBRARIAN] = GeminiClient(
                role=ModelRole.LIBRARIAN,
                api_key=gemini_api_key
            )
            logger.info("Initialized Gemini client (Librarian)")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")

        # Initialize Claude (Architect)
        try:
            self._clients[ModelRole.ARCHITECT] = ClaudeClient(
                role=ModelRole.ARCHITECT,
                api_key=anthropic_api_key
            )
            logger.info("Initialized Claude client (Architect)")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude Architect: {e}")

        # Initialize Claude (Critic) - same API, different role
        try:
            self._clients[ModelRole.CRITIC] = ClaudeClient(
                role=ModelRole.CRITIC,
                api_key=anthropic_api_key
            )
            logger.info("Initialized Claude client (Critic)")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude Critic: {e}")

        # TODO: Initialize GPT-4 Writer
        # TODO: Initialize Llama Verifier

        self._initialized = True
        logger.info(
            f"Model Router initialized with {len(self._clients)} clients: "
            f"{list(self._clients.keys())}"
        )

    def get_client(self, role: ModelRole) -> Optional[BaseLLMClient]:
        """Get client for a specific role"""
        return self._clients.get(role)

    def _decide_routing(
        self,
        task_type: TaskType,
        context_tokens: Optional[int] = None
    ) -> RoutingDecision:
        """
        Decide which model to route to based on task and context.
        """
        # Get optimal model for task
        optimal_role = TASK_ROUTING.get(task_type)

        if optimal_role is None:
            # Default to Architect for unknown tasks
            optimal_role = ModelRole.ARCHITECT
            reason = f"No routing rule for {task_type}, defaulting to Architect"
        else:
            reason = f"Task {task_type.value} routes to {optimal_role.value}"

        # Check if optimal model is available
        if optimal_role not in self._clients:
            # Use fallback
            fallbacks = FALLBACK_CHAIN.get(optimal_role, [])
            for fallback in fallbacks:
                if fallback in self._clients:
                    reason = f"Primary {optimal_role.value} unavailable, falling back to {fallback.value}"
                    optimal_role = fallback
                    break
            else:
                # No fallback available
                available = list(self._clients.keys())
                if available:
                    optimal_role = available[0]
                    reason = f"Using only available model: {optimal_role.value}"
                else:
                    raise RuntimeError("No LLM clients available")

        # Check context size constraints
        if context_tokens:
            client = self._clients.get(optimal_role)
            if client and context_tokens > client.max_context_tokens:
                # Need larger context - try Librarian
                if ModelRole.LIBRARIAN in self._clients:
                    librarian = self._clients[ModelRole.LIBRARIAN]
                    if context_tokens <= librarian.max_context_tokens:
                        reason = f"Context size {context_tokens} exceeds {optimal_role.value} limit, routing to Librarian"
                        optimal_role = ModelRole.LIBRARIAN

        fallback_models = [
            r for r in FALLBACK_CHAIN.get(optimal_role, [])
            if r in self._clients
        ]

        if self.config.log_routing_decisions:
            logger.info(f"Routing decision: {reason}")

        return RoutingDecision(
            selected_model=optimal_role,
            reason=reason,
            fallback_models=fallback_models
        )

    async def route(
        self,
        task_type: TaskType,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        context_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Route a task to the optimal model and execute.

        Args:
            task_type: Type of task to execute
            messages: Messages to send to model
            config: Generation configuration
            context_tokens: Estimated context size (for routing decisions)

        Returns:
            LLMResponse from the selected model
        """
        if not self._initialized:
            raise RuntimeError("Router not initialized. Call initialize() first.")

        # Decide routing
        decision = self._decide_routing(task_type, context_tokens)
        client = self._clients[decision.selected_model]

        try:
            response = await client.generate(messages, config, task_type)
            self._total_usage = self._total_usage + response.token_usage
            return response

        except Exception as e:
            logger.error(f"Error with {decision.selected_model.value}: {e}")

            # Try fallbacks
            if self.config.enable_fallback and decision.fallback_models:
                for fallback_role in decision.fallback_models:
                    try:
                        logger.info(f"Trying fallback: {fallback_role.value}")
                        fallback_client = self._clients[fallback_role]
                        response = await fallback_client.generate(
                            messages, config, task_type
                        )
                        self._total_usage = self._total_usage + response.token_usage
                        return response
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback {fallback_role.value} failed: {fallback_error}"
                        )
                        continue

            raise

    async def route_to_role(
        self,
        role: ModelRole,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        task_type: Optional[TaskType] = None
    ) -> LLMResponse:
        """
        Route directly to a specific role (bypassing task-based routing).

        Useful when you know exactly which model you need.
        """
        if not self._initialized:
            raise RuntimeError("Router not initialized. Call initialize() first.")

        if role not in self._clients:
            raise ValueError(f"No client available for role: {role.value}")

        client = self._clients[role]
        response = await client.generate(messages, config, task_type)
        self._total_usage = self._total_usage + response.token_usage
        return response

    def get_usage_report(self) -> Dict[str, Any]:
        """Get detailed usage report across all models"""
        report = {
            "total": {
                "prompt_tokens": self._total_usage.prompt_tokens,
                "completion_tokens": self._total_usage.completion_tokens,
                "total_tokens": self._total_usage.total_tokens,
                "estimated_cost_usd": self._total_usage.estimated_cost_usd,
            },
            "by_model": {}
        }

        for role, client in self._clients.items():
            usage = client.total_usage
            report["by_model"][role.value] = {
                "model": client.model_name,
                "requests": client.request_count,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost_usd": usage.estimated_cost_usd,
            }

        return report

    def reset_usage(self) -> Dict[str, TokenUsage]:
        """Reset usage tracking and return final stats"""
        stats = {}
        for role, client in self._clients.items():
            stats[role.value] = client.reset_usage()
        self._total_usage = TokenUsage()
        return stats


# Convenience function for quick setup
def create_router(
    strategy: RoutingStrategy = RoutingStrategy.OPTIMAL,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> ModelRouter:
    """
    Create and initialize a model router.

    Args:
        strategy: Routing strategy to use
        gemini_api_key: Google API key
        anthropic_api_key: Anthropic API key
        openai_api_key: OpenAI API key

    Returns:
        Initialized ModelRouter
    """
    config = RoutingConfig(strategy=strategy)
    router = ModelRouter(config)
    router.initialize(
        gemini_api_key=gemini_api_key,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
    )
    return router
