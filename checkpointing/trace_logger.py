"""
PropelAI Agent Trace Logger
Detailed execution logging for agent workflows with cost tracking
"""

import uuid
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of agent execution steps."""
    START = "start"
    AGENT_CALL = "agent_call"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    DECISION = "decision"
    END = "end"
    ERROR = "error"


@dataclass
class TraceLogEntry:
    """A single entry in the agent trace log."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    proposal_id: Optional[str] = None
    trace_run_id: str = ""
    agent_name: str = ""
    step_type: StepType = StepType.AGENT_CALL

    # State snapshots
    input_state: Dict[str, Any] = field(default_factory=dict)
    output_state: Dict[str, Any] = field(default_factory=dict)

    # Execution details
    reasoning_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: int = 0
    model_version: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["step_type"] = self.step_type.value
        d["created_at"] = self.created_at.isoformat()
        return d

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this step."""
        return self.tokens_input + self.tokens_output

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost based on model and tokens."""
        # Pricing per 1M tokens (approximate)
        pricing = {
            "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gpt-4o": {"input": 2.50, "output": 10.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }

        model_key = None
        for key in pricing:
            if key in self.model_version.lower():
                model_key = key
                break

        if not model_key:
            # Default to GPT-4o pricing
            model_key = "gpt-4o"

        rates = pricing[model_key]
        input_cost = (self.tokens_input / 1_000_000) * rates["input"]
        output_cost = (self.tokens_output / 1_000_000) * rates["output"]

        return round(input_cost + output_cost, 6)


@dataclass
class TraceRun:
    """A complete trace run containing multiple log entries."""
    trace_run_id: str
    tenant_id: str
    proposal_id: Optional[str]
    entries: List[TraceLogEntry] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "running"
    error_message: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens across all entries."""
        return sum(e.total_tokens for e in self.entries)

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost."""
        return sum(e.estimated_cost_usd for e in self.entries)

    @property
    def total_latency_ms(self) -> int:
        """Total latency across all entries."""
        return sum(e.latency_ms for e in self.entries)

    @property
    def duration_ms(self) -> int:
        """Total duration of the run."""
        if self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return int((datetime.utcnow() - self.started_at).total_seconds() * 1000)


class AgentTraceLogger:
    """
    Agent trace logger for detailed execution tracking.

    Captures:
    - Input/output states for each agent step
    - LLM reasoning and tool calls
    - Token usage and latency metrics
    - Cost estimation per run
    """

    def __init__(
        self,
        db_pool: Optional[Any] = None,
        buffer_size: int = 10,
    ):
        self.db_pool = db_pool
        self.buffer_size = buffer_size
        self._buffer: List[TraceLogEntry] = []
        self._active_runs: Dict[str, TraceRun] = {}
        self._connection = None

    async def _get_connection(self):
        """Get database connection."""
        if self.db_pool:
            return await self.db_pool.acquire()
        return None

    async def _release_connection(self, conn):
        """Release connection back to pool."""
        if self.db_pool and conn:
            await self.db_pool.release(conn)

    def start_run(
        self,
        tenant_id: str,
        proposal_id: Optional[str] = None,
    ) -> str:
        """Start a new trace run and return its ID."""
        trace_run_id = str(uuid.uuid4())

        run = TraceRun(
            trace_run_id=trace_run_id,
            tenant_id=tenant_id,
            proposal_id=proposal_id,
        )
        self._active_runs[trace_run_id] = run

        # Log start entry
        self._add_entry(TraceLogEntry(
            tenant_id=tenant_id,
            proposal_id=proposal_id,
            trace_run_id=trace_run_id,
            agent_name="system",
            step_type=StepType.START,
            input_state={"proposal_id": proposal_id},
        ))

        return trace_run_id

    def end_run(
        self,
        trace_run_id: str,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> Optional[TraceRun]:
        """End a trace run."""
        run = self._active_runs.get(trace_run_id)
        if not run:
            return None

        run.completed_at = datetime.utcnow()
        run.status = status
        run.error_message = error_message

        # Log end entry
        self._add_entry(TraceLogEntry(
            tenant_id=run.tenant_id,
            proposal_id=run.proposal_id,
            trace_run_id=trace_run_id,
            agent_name="system",
            step_type=StepType.END,
            output_state={
                "status": status,
                "total_tokens": run.total_tokens,
                "total_cost_usd": run.total_cost_usd,
                "duration_ms": run.duration_ms,
            },
        ))

        # Remove from active runs
        del self._active_runs[trace_run_id]

        return run

    @asynccontextmanager
    async def trace_run(
        self,
        tenant_id: str,
        proposal_id: Optional[str] = None,
    ):
        """Context manager for tracing a complete run."""
        trace_run_id = self.start_run(tenant_id, proposal_id)
        try:
            yield trace_run_id
            self.end_run(trace_run_id, status="completed")
        except Exception as e:
            self.end_run(trace_run_id, status="failed", error_message=str(e))
            raise
        finally:
            await self.flush()

    def log_agent_step(
        self,
        trace_run_id: str,
        agent_name: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_outputs: Optional[List[Dict[str, Any]]] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        latency_ms: int = 0,
        model_version: str = "",
    ) -> TraceLogEntry:
        """Log an agent execution step."""
        run = self._active_runs.get(trace_run_id)
        if not run:
            logger.warning(f"Trace run {trace_run_id} not found")
            run = TraceRun(
                trace_run_id=trace_run_id,
                tenant_id="unknown",
                proposal_id=None,
            )

        entry = TraceLogEntry(
            tenant_id=run.tenant_id,
            proposal_id=run.proposal_id,
            trace_run_id=trace_run_id,
            agent_name=agent_name,
            step_type=StepType.AGENT_CALL,
            input_state=self._sanitize_state(input_state),
            output_state=self._sanitize_state(output_state),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or [],
            tool_outputs=tool_outputs or [],
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            model_version=model_version,
        )

        self._add_entry(entry)
        run.entries.append(entry)

        return entry

    def log_tool_call(
        self,
        trace_run_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        latency_ms: int = 0,
    ) -> TraceLogEntry:
        """Log a tool call within an agent step."""
        run = self._active_runs.get(trace_run_id)
        if not run:
            run = TraceRun(trace_run_id=trace_run_id, tenant_id="unknown", proposal_id=None)

        entry = TraceLogEntry(
            tenant_id=run.tenant_id,
            proposal_id=run.proposal_id,
            trace_run_id=trace_run_id,
            agent_name=f"tool:{tool_name}",
            step_type=StepType.TOOL_CALL,
            input_state=tool_input,
            output_state={"output": str(tool_output)[:1000]},  # Truncate
            latency_ms=latency_ms,
        )

        self._add_entry(entry)
        return entry

    def log_decision(
        self,
        trace_run_id: str,
        agent_name: str,
        decision: str,
        reasoning: Optional[str] = None,
    ) -> TraceLogEntry:
        """Log an agent decision point."""
        run = self._active_runs.get(trace_run_id)
        if not run:
            run = TraceRun(trace_run_id=trace_run_id, tenant_id="unknown", proposal_id=None)

        entry = TraceLogEntry(
            tenant_id=run.tenant_id,
            proposal_id=run.proposal_id,
            trace_run_id=trace_run_id,
            agent_name=agent_name,
            step_type=StepType.DECISION,
            output_state={"decision": decision},
            reasoning_content=reasoning,
        )

        self._add_entry(entry)
        return entry

    def log_error(
        self,
        trace_run_id: str,
        agent_name: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> TraceLogEntry:
        """Log an error during execution."""
        run = self._active_runs.get(trace_run_id)
        if not run:
            run = TraceRun(trace_run_id=trace_run_id, tenant_id="unknown", proposal_id=None)

        entry = TraceLogEntry(
            tenant_id=run.tenant_id,
            proposal_id=run.proposal_id,
            trace_run_id=trace_run_id,
            agent_name=agent_name,
            step_type=StepType.ERROR,
            output_state={
                "error": error_message,
                "details": error_details or {},
            },
        )

        self._add_entry(entry)
        return entry

    def _add_entry(self, entry: TraceLogEntry):
        """Add entry to buffer."""
        self._buffer.append(entry)

        # Auto-flush when buffer is full
        if len(self._buffer) >= self.buffer_size:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.flush())
                else:
                    loop.run_until_complete(self.flush())
            except RuntimeError:
                pass  # No event loop, will flush later

    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state for storage (remove large/sensitive data)."""
        sanitized = {}
        for key, value in state.items():
            if isinstance(value, str) and len(value) > 10000:
                sanitized[key] = value[:10000] + "... [truncated]"
            elif isinstance(value, bytes):
                sanitized[key] = f"<bytes: {len(value)} bytes>"
            elif isinstance(value, (list, dict)):
                try:
                    json.dumps(value)  # Test serializability
                    sanitized[key] = value
                except (TypeError, ValueError):
                    sanitized[key] = str(value)[:1000]
            else:
                sanitized[key] = value
        return sanitized

    async def flush(self):
        """Flush buffered entries to database."""
        if not self._buffer:
            return

        entries = self._buffer.copy()
        self._buffer.clear()

        conn = await self._get_connection()
        if not conn:
            logger.warning("No database connection, entries lost")
            return

        try:
            for entry in entries:
                await conn.execute(
                    """
                    INSERT INTO agent_trace_log (
                        id, tenant_id, proposal_id, trace_run_id, agent_name,
                        step_type, input_state, output_state, reasoning_content,
                        tool_calls, tool_outputs, tokens_input, tokens_output,
                        latency_ms, model_version, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """,
                    uuid.UUID(entry.id),
                    uuid.UUID(entry.tenant_id) if entry.tenant_id else None,
                    entry.proposal_id,
                    entry.trace_run_id,
                    entry.agent_name,
                    entry.step_type.value,
                    json.dumps(entry.input_state),
                    json.dumps(entry.output_state),
                    entry.reasoning_content,
                    json.dumps(entry.tool_calls),
                    json.dumps(entry.tool_outputs),
                    entry.tokens_input,
                    entry.tokens_output,
                    entry.latency_ms,
                    entry.model_version,
                    entry.created_at,
                )
        finally:
            await self._release_connection(conn)

    async def get_run_trace(self, trace_run_id: str) -> List[TraceLogEntry]:
        """Get all entries for a trace run."""
        conn = await self._get_connection()
        if not conn:
            return []

        try:
            rows = await conn.fetch(
                """
                SELECT * FROM agent_trace_log
                WHERE trace_run_id = $1
                ORDER BY created_at
                """,
                trace_run_id
            )

            return [
                TraceLogEntry(
                    id=str(row["id"]),
                    tenant_id=str(row["tenant_id"]) if row["tenant_id"] else "",
                    proposal_id=row["proposal_id"],
                    trace_run_id=row["trace_run_id"],
                    agent_name=row["agent_name"],
                    step_type=StepType(row["step_type"]),
                    input_state=row["input_state"],
                    output_state=row["output_state"],
                    reasoning_content=row["reasoning_content"],
                    tool_calls=row["tool_calls"] or [],
                    tool_outputs=row["tool_outputs"] or [],
                    tokens_input=row["tokens_input"] or 0,
                    tokens_output=row["tokens_output"] or 0,
                    latency_ms=row["latency_ms"] or 0,
                    model_version=row["model_version"] or "",
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        finally:
            await self._release_connection(conn)

    async def get_proposal_traces(
        self,
        proposal_id: str,
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all trace runs for a proposal."""
        conn = await self._get_connection()
        if not conn:
            return []

        try:
            rows = await conn.fetch(
                """
                SELECT DISTINCT trace_run_id,
                       MIN(created_at) as started_at,
                       MAX(created_at) as ended_at,
                       SUM(tokens_input + tokens_output) as total_tokens,
                       SUM(latency_ms) as total_latency_ms,
                       COUNT(*) as step_count
                FROM agent_trace_log
                WHERE proposal_id = $1 AND tenant_id = $2
                GROUP BY trace_run_id
                ORDER BY MIN(created_at) DESC
                """,
                proposal_id, uuid.UUID(tenant_id)
            )

            return [
                {
                    "trace_run_id": row["trace_run_id"],
                    "started_at": row["started_at"].isoformat(),
                    "ended_at": row["ended_at"].isoformat(),
                    "total_tokens": row["total_tokens"],
                    "total_latency_ms": row["total_latency_ms"],
                    "step_count": row["step_count"],
                }
                for row in rows
            ]
        finally:
            await self._release_connection(conn)


# Singleton instance
_trace_logger: Optional[AgentTraceLogger] = None


def get_trace_logger(db_pool: Optional[Any] = None) -> AgentTraceLogger:
    """Get or create trace logger singleton."""
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = AgentTraceLogger(db_pool=db_pool)
    return _trace_logger
