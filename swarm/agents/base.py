"""
PropelAI v6.0 Base Agent
Foundation class for all agents in the swarm.
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from swarm.state import ProposalState, AgentAction


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    model: str = "gemini-1.5-pro"
    temperature: float = 0.2
    max_tokens: int = 8192
    system_prompt: str = ""

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # Tracing
    trace_enabled: bool = True

    # Tools
    tools: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Base class for all agents in the PropelAI swarm.

    Each agent:
    - Receives the full ProposalState
    - Performs its specialized function
    - Returns an updated ProposalState
    - Logs all actions to the trace log
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self._llm = None
        self._tools: Dict[str, Callable] = {}

    @property
    def llm(self):
        """Lazy-load the LLM client."""
        if self._llm is None:
            self._llm = self._init_llm()
        return self._llm

    def _init_llm(self):
        """Initialize the LLM client based on config."""
        model = self.config.model.lower()

        if "gemini" in model:
            return self._init_gemini()
        elif "gpt" in model or "openai" in model:
            return self._init_openai()
        elif "claude" in model:
            return self._init_anthropic()
        else:
            # Default to Gemini
            return self._init_gemini()

    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.config.model)
        except ImportError:
            logger.warning("google-generativeai not installed")
            return None

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            return OpenAI()
        except ImportError:
            logger.warning("openai not installed")
            return None

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            return anthropic.Anthropic()
        except ImportError:
            logger.warning("anthropic not installed")
            return None

    def register_tool(self, name: str, func: Callable):
        """Register a tool for this agent."""
        self._tools[name] = func

    async def __call__(self, state: ProposalState) -> ProposalState:
        """Execute the agent on the given state."""
        return await self.run(state)

    async def run(self, state: ProposalState) -> ProposalState:
        """
        Execute the agent's main logic.
        Wraps _execute with tracing and error handling.
        """
        start_time = time.time()

        # Log start
        self._log_action(
            state,
            "thought",
            f"Starting {self.name} execution",
        )

        try:
            # Execute the agent's logic
            new_state = await self._execute(state)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Log completion
            self._log_action(
                new_state,
                "output",
                f"Completed {self.name} execution in {latency_ms}ms",
                latency_ms=latency_ms,
            )

            # Update state metadata
            new_state["updated_at"] = datetime.utcnow().isoformat()
            new_state["current_agent"] = self.name

            return new_state

        except Exception as e:
            logger.exception(f"Error in {self.name}")

            # Log error
            self._log_action(
                state,
                "observation",
                f"Error in {self.name}: {str(e)}",
            )

            # Update error state
            state["last_error"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1

            return state

    @abstractmethod
    async def _execute(self, state: ProposalState) -> ProposalState:
        """
        The agent's core logic. Must be implemented by subclasses.

        Args:
            state: The current proposal state

        Returns:
            Updated proposal state
        """
        pass

    def _log_action(
        self,
        state: ProposalState,
        action_type: str,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_outputs: Optional[List[Dict]] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        latency_ms: int = 0,
    ):
        """Log an action to the trace log."""
        if not self.config.trace_enabled:
            return

        action = AgentAction(
            timestamp=datetime.utcnow(),
            agent_name=self.name,
            action_type=action_type,
            content=content,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
        )

        # Append to trace log
        trace_log = state.get("agent_trace_log", [])
        trace_log.append(action.to_dict())
        state["agent_trace_log"] = trace_log

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt override
            json_mode: Whether to force JSON output

        Returns:
            The LLM response text
        """
        system = system_prompt or self.config.system_prompt
        model = self.config.model.lower()

        if "gemini" in model:
            return await self._call_gemini(prompt, system, json_mode)
        elif "gpt" in model or "openai" in model:
            return await self._call_openai(prompt, system, json_mode)
        elif "claude" in model:
            return await self._call_anthropic(prompt, system, json_mode)
        else:
            return await self._call_gemini(prompt, system, json_mode)

    async def _call_gemini(
        self,
        prompt: str,
        system: str,
        json_mode: bool,
    ) -> str:
        """Call Gemini API."""
        import asyncio

        if not self.llm:
            return ""

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        # Run in executor since Gemini client is sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llm.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                },
            )
        )

        return response.text if response else ""

    async def _call_openai(
        self,
        prompt: str,
        system: str,
        json_mode: bool,
    ) -> str:
        """Call OpenAI API."""
        if not self.llm:
            return ""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.llm.chat.completions.create(**kwargs)
        return response.choices[0].message.content if response.choices else ""

    async def _call_anthropic(
        self,
        prompt: str,
        system: str,
        json_mode: bool,
    ) -> str:
        """Call Anthropic API."""
        if not self.llm:
            return ""

        if json_mode:
            prompt += "\n\nRespond with valid JSON only."

        response = await self.llm.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text if response.content else ""

    def _call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        return self._tools[tool_name](**kwargs)

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response, handling common issues."""
        # Try to find JSON in the response
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            # Try to find JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON: {text[:500]}")
            return {}
