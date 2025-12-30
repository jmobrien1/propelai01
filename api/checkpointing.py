"""
PropelAI Checkpointing & Tracing API Endpoints
Agent state persistence, trace viewing, and feedback collection
"""

import json
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user, get_current_tenant
from checkpointing.postgres_saver import PostgresCheckpointSaver, get_checkpoint_saver
from checkpointing.trace_logger import AgentTraceLogger, get_trace_logger, StepType
from checkpointing.feedback import (
    FeedbackCollector,
    FeedbackPair,
    EditType,
    get_feedback_collector,
)


router = APIRouter(prefix="/api/checkpointing", tags=["checkpointing"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ThreadInfo(BaseModel):
    """Information about an agent thread."""
    thread_id: str
    checkpoint_id: str
    last_updated: str
    metadata: dict


class CheckpointInfo(BaseModel):
    """Checkpoint details."""
    thread_id: str
    checkpoint_id: str
    parent_checkpoint_id: Optional[str]
    created_at: str
    state_summary: dict


class TraceEntry(BaseModel):
    """A trace log entry."""
    id: str
    agent_name: str
    step_type: str
    input_summary: dict
    output_summary: dict
    reasoning_content: Optional[str]
    tokens_input: int
    tokens_output: int
    latency_ms: int
    model_version: str
    created_at: str


class TraceRunSummary(BaseModel):
    """Summary of a trace run."""
    trace_run_id: str
    started_at: str
    ended_at: str
    total_tokens: int
    total_latency_ms: int
    step_count: int
    estimated_cost_usd: Optional[float] = None


class FeedbackRequest(BaseModel):
    """Request to submit feedback."""
    original_text: str = Field(..., description="Original AI-generated text")
    edited_text: str = Field(..., description="Human-edited text")
    proposal_id: Optional[str] = None
    section_id: Optional[str] = None
    section_type: Optional[str] = None
    requirement_text: Optional[str] = None
    trace_log_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response after feedback submission."""
    feedback_id: str
    edit_type: str
    edit_distance: float
    change_ratio: float
    message: str


class FeedbackAnalysisResponse(BaseModel):
    """Feedback analysis results."""
    total_pairs: int
    pairs_by_type: dict
    pairs_by_section: dict
    average_edit_distance: float
    most_edited_sections: List[str]
    common_issues: List[str]


# ============================================================================
# Dependencies
# ============================================================================


async def get_checkpointer() -> PostgresCheckpointSaver:
    """Get checkpoint saver service."""
    return get_checkpoint_saver()


async def get_tracer() -> AgentTraceLogger:
    """Get trace logger service."""
    return get_trace_logger()


async def get_feedback_service() -> FeedbackCollector:
    """Get feedback collector service."""
    return get_feedback_collector()


# ============================================================================
# Thread Management Endpoints
# ============================================================================


@router.get("/threads", response_model=List[ThreadInfo])
async def list_threads(
    limit: int = Query(default=50, ge=1, le=200),
    tenant_id: str = Depends(get_current_tenant),
    checkpointer: PostgresCheckpointSaver = Depends(get_checkpointer),
):
    """List active agent threads for the tenant."""
    threads = await checkpointer.get_active_threads(
        tenant_id=tenant_id,
        limit=limit,
    )

    return [
        ThreadInfo(
            thread_id=t["thread_id"],
            checkpoint_id=t["checkpoint_id"],
            last_updated=t["last_updated"],
            metadata=t["metadata"] or {},
        )
        for t in threads
    ]


@router.get("/threads/{thread_id}", response_model=List[CheckpointInfo])
async def get_thread_history(
    thread_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    tenant_id: str = Depends(get_current_tenant),
    checkpointer: PostgresCheckpointSaver = Depends(get_checkpointer),
):
    """Get checkpoint history for a thread."""
    history = await checkpointer.get_thread_history(
        thread_id=thread_id,
        limit=limit,
    )

    return [
        CheckpointInfo(
            thread_id=h.thread_id,
            checkpoint_id=h.checkpoint_id,
            parent_checkpoint_id=h.parent_checkpoint_id,
            created_at=h.created_at.isoformat(),
            state_summary=_summarize_state(h.checkpoint_data),
        )
        for h in history
    ]


@router.post("/threads/{thread_id}/rollback")
async def rollback_thread(
    thread_id: str,
    checkpoint_id: str = Query(..., description="Checkpoint to rollback to"),
    tenant_id: str = Depends(get_current_tenant),
    checkpointer: PostgresCheckpointSaver = Depends(get_checkpointer),
):
    """Rollback a thread to a specific checkpoint."""
    success = await checkpointer.rollback_to_checkpoint(
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return {
        "status": "rolled_back",
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id,
    }


@router.post("/threads/{thread_id}/fork")
async def fork_thread(
    thread_id: str,
    new_thread_id: Optional[str] = Query(default=None),
    from_checkpoint_id: Optional[str] = Query(default=None),
    tenant_id: str = Depends(get_current_tenant),
    checkpointer: PostgresCheckpointSaver = Depends(get_checkpointer),
):
    """Fork a thread to create a new branch."""
    new_id = await checkpointer.fork_thread(
        source_thread_id=thread_id,
        new_thread_id=new_thread_id,
        from_checkpoint_id=from_checkpoint_id,
    )

    return {
        "status": "forked",
        "source_thread_id": thread_id,
        "new_thread_id": new_id,
    }


@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    tenant_id: str = Depends(get_current_tenant),
    checkpointer: PostgresCheckpointSaver = Depends(get_checkpointer),
):
    """Delete a thread and all its checkpoints."""
    success = await checkpointer.delete_thread(thread_id)

    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")

    return {"status": "deleted", "thread_id": thread_id}


# ============================================================================
# Trace Viewing Endpoints
# ============================================================================


@router.get("/traces/proposal/{proposal_id}", response_model=List[TraceRunSummary])
async def get_proposal_traces(
    proposal_id: str,
    tenant_id: str = Depends(get_current_tenant),
    tracer: AgentTraceLogger = Depends(get_tracer),
):
    """Get all trace runs for a proposal."""
    traces = await tracer.get_proposal_traces(
        proposal_id=proposal_id,
        tenant_id=tenant_id,
    )

    return [
        TraceRunSummary(
            trace_run_id=t["trace_run_id"],
            started_at=t["started_at"],
            ended_at=t["ended_at"],
            total_tokens=t["total_tokens"] or 0,
            total_latency_ms=t["total_latency_ms"] or 0,
            step_count=t["step_count"],
        )
        for t in traces
    ]


@router.get("/traces/{trace_run_id}", response_model=List[TraceEntry])
async def get_trace_entries(
    trace_run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    tracer: AgentTraceLogger = Depends(get_tracer),
):
    """Get all entries for a trace run."""
    entries = await tracer.get_run_trace(trace_run_id)

    return [
        TraceEntry(
            id=e.id,
            agent_name=e.agent_name,
            step_type=e.step_type.value,
            input_summary=_summarize_state(e.input_state),
            output_summary=_summarize_state(e.output_state),
            reasoning_content=e.reasoning_content,
            tokens_input=e.tokens_input,
            tokens_output=e.tokens_output,
            latency_ms=e.latency_ms,
            model_version=e.model_version,
            created_at=e.created_at.isoformat(),
        )
        for e in entries
    ]


@router.get("/traces/{trace_run_id}/cost")
async def get_trace_cost(
    trace_run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    tracer: AgentTraceLogger = Depends(get_tracer),
):
    """Get cost breakdown for a trace run."""
    entries = await tracer.get_run_trace(trace_run_id)

    total_input_tokens = sum(e.tokens_input for e in entries)
    total_output_tokens = sum(e.tokens_output for e in entries)
    total_cost = sum(e.estimated_cost_usd for e in entries)

    by_agent = {}
    for e in entries:
        if e.agent_name not in by_agent:
            by_agent[e.agent_name] = {
                "tokens_input": 0,
                "tokens_output": 0,
                "cost_usd": 0.0,
                "call_count": 0,
            }
        by_agent[e.agent_name]["tokens_input"] += e.tokens_input
        by_agent[e.agent_name]["tokens_output"] += e.tokens_output
        by_agent[e.agent_name]["cost_usd"] += e.estimated_cost_usd
        by_agent[e.agent_name]["call_count"] += 1

    return {
        "trace_run_id": trace_run_id,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "by_agent": by_agent,
    }


# ============================================================================
# Feedback Endpoints
# ============================================================================


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    tenant_id: str = Depends(get_current_tenant),
    user = Depends(get_current_user),
    feedback_service: FeedbackCollector = Depends(get_feedback_service),
):
    """
    Submit human feedback on AI-generated content.

    Used to capture human edits for quality improvement and fine-tuning.
    """
    pair = feedback_service.collect(
        original_text=request.original_text,
        edited_text=request.edited_text,
        tenant_id=tenant_id,
        proposal_id=request.proposal_id,
        section_id=request.section_id,
        section_type=request.section_type,
        requirement_text=request.requirement_text,
        trace_log_id=request.trace_log_id,
        user_id=user.id if hasattr(user, 'id') else None,
        user_role=user.role if hasattr(user, 'role') else None,
    )

    # Save immediately
    await feedback_service.save()

    return FeedbackResponse(
        feedback_id=pair.id,
        edit_type=pair.edit_type.value,
        edit_distance=round(pair.edit_distance, 4),
        change_ratio=round(pair.change_ratio, 4),
        message="Feedback captured successfully",
    )


@router.get("/feedback", response_model=List[dict])
async def list_feedback(
    proposal_id: Optional[str] = Query(default=None),
    edit_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    tenant_id: str = Depends(get_current_tenant),
    feedback_service: FeedbackCollector = Depends(get_feedback_service),
):
    """List feedback pairs for the tenant."""
    edit_types = None
    if edit_type:
        try:
            edit_types = [EditType(edit_type)]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid edit type: {edit_type}")

    pairs = await feedback_service.get_pairs(
        tenant_id=tenant_id,
        proposal_id=proposal_id,
        edit_types=edit_types,
        limit=limit,
    )

    return [p.to_dict() for p in pairs]


@router.get("/feedback/analysis", response_model=FeedbackAnalysisResponse)
async def analyze_feedback(
    tenant_id: str = Depends(get_current_tenant),
    feedback_service: FeedbackCollector = Depends(get_feedback_service),
):
    """Analyze feedback patterns for the tenant."""
    analysis = await feedback_service.analyze(tenant_id)

    return FeedbackAnalysisResponse(
        total_pairs=analysis.total_pairs,
        pairs_by_type=analysis.pairs_by_type,
        pairs_by_section=analysis.pairs_by_section,
        average_edit_distance=round(analysis.average_edit_distance, 4),
        most_edited_sections=analysis.most_edited_sections,
        common_issues=analysis.common_issues,
    )


@router.get("/feedback/export")
async def export_feedback(
    format: str = Query(default="jsonl", description="Export format: jsonl, openai, anthropic"),
    min_quality: float = Query(default=0.0, ge=0, le=1),
    tenant_id: str = Depends(get_current_tenant),
    feedback_service: FeedbackCollector = Depends(get_feedback_service),
):
    """Export feedback pairs for fine-tuning."""
    if format not in ["jsonl", "openai", "anthropic"]:
        raise HTTPException(status_code=400, detail=f"Invalid format: {format}")

    data = await feedback_service.export_for_training(
        tenant_id=tenant_id,
        format=format,
        min_quality_delta=min_quality,
    )

    return {
        "format": format,
        "count": len(data),
        "data": data,
    }


# ============================================================================
# Helper Functions
# ============================================================================


def _summarize_state(state: dict) -> dict:
    """Summarize state for display (truncate large values)."""
    if not state:
        return {}

    summary = {}
    for key, value in state.items():
        if isinstance(value, str):
            summary[key] = value[:200] + "..." if len(value) > 200 else value
        elif isinstance(value, list):
            summary[key] = f"[{len(value)} items]"
        elif isinstance(value, dict):
            summary[key] = f"{{{len(value)} keys}}"
        else:
            summary[key] = str(value)[:100]

    return summary
