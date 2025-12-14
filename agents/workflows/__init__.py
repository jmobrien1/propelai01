"""
PropelAI Workflow Orchestration

LangGraph-based workflows for long-form proposal generation.

Workflows:
- Draft-Critique-Expand (DCE): Cyclic loop to defeat brevity bias
- (Future) Multi-section orchestration
- (Future) Parallel volume generation
"""

from .draft_critique_expand import (
    DCEWorkflow,
    DCEResult,
    DCEState,
    DCEPhase,
    build_dce_workflow,
    run_dce_loop,
)

__all__ = [
    "DCEWorkflow",
    "DCEResult",
    "DCEState",
    "DCEPhase",
    "build_dce_workflow",
    "run_dce_loop",
]
