"""
PropelAI Feedback Collection System
Captures human edits for fine-tuning and quality improvement
"""

import uuid
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum
import difflib


logger = logging.getLogger(__name__)


class EditType(str, Enum):
    """Types of edits captured."""
    MINOR_CORRECTION = "minor_correction"      # Typos, grammar
    FACTUAL_FIX = "factual_fix"                # Correcting facts
    STYLE_CHANGE = "style_change"              # Tone, voice changes
    CONTENT_ADDITION = "content_addition"      # Adding new content
    CONTENT_REMOVAL = "content_removal"        # Removing content
    RESTRUCTURE = "restructure"                # Reorganizing content
    COMPLETE_REWRITE = "complete_rewrite"      # Full rewrite


@dataclass
class FeedbackPair:
    """A pair of original and edited text for training."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_log_id: Optional[str] = None
    tenant_id: str = ""
    proposal_id: Optional[str] = None
    section_id: Optional[str] = None

    # The content
    original_text: str = ""
    original_score: Optional[float] = None
    human_edited_text: str = ""

    # Classification
    edit_type: EditType = EditType.MINOR_CORRECTION
    edit_distance: float = 0.0  # Normalized edit distance
    change_ratio: float = 0.0   # Ratio of changed content

    # Context for training
    prompt_context: str = ""
    requirement_text: Optional[str] = None
    section_type: Optional[str] = None

    # Metadata
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["edit_type"] = self.edit_type.value
        d["created_at"] = self.created_at.isoformat()
        return d

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to format suitable for fine-tuning."""
        return {
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt()},
                {"role": "assistant", "content": self.human_edited_text},
            ],
            "metadata": {
                "section_type": self.section_type,
                "edit_type": self.edit_type.value,
                "quality_delta": self._estimate_quality_delta(),
            }
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt for training."""
        return """You are a proposal writing assistant specialized in government contracting.
Write professional, compliant proposal sections that directly address requirements.
Use specific, concrete language and quantify claims when possible."""

    def _build_user_prompt(self) -> str:
        """Build user prompt for training."""
        parts = []
        if self.requirement_text:
            parts.append(f"Requirement: {self.requirement_text}")
        if self.section_type:
            parts.append(f"Section Type: {self.section_type}")
        if self.prompt_context:
            parts.append(f"Context: {self.prompt_context}")
        return "\n\n".join(parts)

    def _estimate_quality_delta(self) -> float:
        """Estimate the quality improvement from the edit."""
        # Simple heuristic based on edit type and distance
        base_scores = {
            EditType.MINOR_CORRECTION: 0.1,
            EditType.FACTUAL_FIX: 0.3,
            EditType.STYLE_CHANGE: 0.2,
            EditType.CONTENT_ADDITION: 0.4,
            EditType.CONTENT_REMOVAL: 0.3,
            EditType.RESTRUCTURE: 0.5,
            EditType.COMPLETE_REWRITE: 0.7,
        }
        return base_scores.get(self.edit_type, 0.2)


@dataclass
class FeedbackAnalysis:
    """Analysis of feedback patterns."""
    total_pairs: int
    pairs_by_type: Dict[str, int]
    pairs_by_section: Dict[str, int]
    average_edit_distance: float
    most_edited_sections: List[str]
    common_issues: List[str]


class FeedbackCollector:
    """
    Collects and manages human feedback for model improvement.

    Features:
    - Captures original vs. edited content pairs
    - Classifies edit types automatically
    - Computes quality metrics
    - Exports in fine-tuning formats (JSONL, OpenAI, Anthropic)
    """

    def __init__(
        self,
        db_pool: Optional[Any] = None,
        auto_classify: bool = True,
    ):
        self.db_pool = db_pool
        self.auto_classify = auto_classify
        self._buffer: List[FeedbackPair] = []
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

    def collect(
        self,
        original_text: str,
        edited_text: str,
        tenant_id: str,
        proposal_id: Optional[str] = None,
        section_id: Optional[str] = None,
        section_type: Optional[str] = None,
        requirement_text: Optional[str] = None,
        prompt_context: Optional[str] = None,
        original_score: Optional[float] = None,
        trace_log_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> FeedbackPair:
        """
        Collect a feedback pair from human editing.

        Args:
            original_text: The AI-generated original text
            edited_text: The human-edited text
            tenant_id: Tenant ID
            proposal_id: Associated proposal
            section_id: Section identifier
            section_type: Type of section (technical, management, etc.)
            requirement_text: The requirement being addressed
            prompt_context: Context provided to the AI
            original_score: AI's confidence score
            trace_log_id: Link to trace log
            user_id: User who made the edit
            user_role: User's role

        Returns:
            FeedbackPair object
        """
        # Calculate edit metrics
        edit_distance = self._calculate_edit_distance(original_text, edited_text)
        change_ratio = self._calculate_change_ratio(original_text, edited_text)

        # Auto-classify edit type
        edit_type = EditType.MINOR_CORRECTION
        if self.auto_classify:
            edit_type = self._classify_edit(
                original_text, edited_text, edit_distance, change_ratio
            )

        pair = FeedbackPair(
            trace_log_id=trace_log_id,
            tenant_id=tenant_id,
            proposal_id=proposal_id,
            section_id=section_id,
            original_text=original_text,
            original_score=original_score,
            human_edited_text=edited_text,
            edit_type=edit_type,
            edit_distance=edit_distance,
            change_ratio=change_ratio,
            prompt_context=prompt_context or "",
            requirement_text=requirement_text,
            section_type=section_type,
            user_id=user_id,
            user_role=user_role,
        )

        self._buffer.append(pair)
        return pair

    def _calculate_edit_distance(self, original: str, edited: str) -> float:
        """Calculate normalized Levenshtein distance."""
        if not original and not edited:
            return 0.0

        # Use difflib ratio (1 - distance normalized)
        ratio = difflib.SequenceMatcher(None, original, edited).ratio()
        return 1.0 - ratio

    def _calculate_change_ratio(self, original: str, edited: str) -> float:
        """Calculate ratio of changed words."""
        orig_words = set(original.lower().split())
        edit_words = set(edited.lower().split())

        if not orig_words:
            return 1.0 if edit_words else 0.0

        # Symmetric difference
        changed = orig_words.symmetric_difference(edit_words)
        total = orig_words.union(edit_words)

        return len(changed) / len(total) if total else 0.0

    def _classify_edit(
        self,
        original: str,
        edited: str,
        edit_distance: float,
        change_ratio: float,
    ) -> EditType:
        """Automatically classify the type of edit."""
        # Complete rewrite if most content changed
        if change_ratio > 0.7 or edit_distance > 0.6:
            return EditType.COMPLETE_REWRITE

        # Restructure if length similar but words reordered
        if change_ratio > 0.4 and abs(len(original) - len(edited)) < len(original) * 0.2:
            return EditType.RESTRUCTURE

        # Content addition if significantly longer
        if len(edited) > len(original) * 1.3:
            return EditType.CONTENT_ADDITION

        # Content removal if significantly shorter
        if len(edited) < len(original) * 0.7:
            return EditType.CONTENT_REMOVAL

        # Minor correction if small changes
        if edit_distance < 0.15:
            return EditType.MINOR_CORRECTION

        # Style change for moderate edits
        if change_ratio < 0.3:
            return EditType.STYLE_CHANGE

        # Default to factual fix
        return EditType.FACTUAL_FIX

    async def save(self) -> int:
        """Save buffered feedback pairs to database."""
        if not self._buffer:
            return 0

        pairs = self._buffer.copy()
        self._buffer.clear()

        conn = await self._get_connection()
        if not conn:
            logger.warning("No database connection, feedback pairs lost")
            return 0

        saved_count = 0
        try:
            for pair in pairs:
                await conn.execute(
                    """
                    INSERT INTO feedback_pairs (
                        id, trace_log_id, tenant_id, proposal_id, section_id,
                        original_text, original_score, human_edited_text,
                        edit_type, prompt_context, user_role, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    uuid.UUID(pair.id),
                    uuid.UUID(pair.trace_log_id) if pair.trace_log_id else None,
                    uuid.UUID(pair.tenant_id) if pair.tenant_id else None,
                    pair.proposal_id,
                    pair.section_id,
                    pair.original_text,
                    pair.original_score,
                    pair.human_edited_text,
                    pair.edit_type.value,
                    pair.prompt_context,
                    pair.user_role,
                    pair.created_at,
                )
                saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save feedback pairs: {e}")
        finally:
            await self._release_connection(conn)

        return saved_count

    async def get_pairs(
        self,
        tenant_id: str,
        proposal_id: Optional[str] = None,
        edit_types: Optional[List[EditType]] = None,
        min_change_ratio: float = 0.0,
        limit: int = 100,
    ) -> List[FeedbackPair]:
        """Retrieve feedback pairs with filters."""
        conn = await self._get_connection()
        if not conn:
            return []

        try:
            sql = """
                SELECT * FROM feedback_pairs
                WHERE tenant_id = $1
            """
            params = [uuid.UUID(tenant_id)]
            param_idx = 2

            if proposal_id:
                sql += f" AND proposal_id = ${param_idx}"
                params.append(proposal_id)
                param_idx += 1

            if edit_types:
                type_values = [t.value for t in edit_types]
                sql += f" AND edit_type = ANY(${param_idx}::text[])"
                params.append(type_values)
                param_idx += 1

            sql += f" ORDER BY created_at DESC LIMIT ${param_idx}"
            params.append(limit)

            rows = await conn.fetch(sql, *params)

            return [
                FeedbackPair(
                    id=str(row["id"]),
                    trace_log_id=str(row["trace_log_id"]) if row["trace_log_id"] else None,
                    tenant_id=str(row["tenant_id"]),
                    proposal_id=row["proposal_id"],
                    section_id=row["section_id"],
                    original_text=row["original_text"],
                    original_score=row["original_score"],
                    human_edited_text=row["human_edited_text"],
                    edit_type=EditType(row["edit_type"]),
                    prompt_context=row["prompt_context"] or "",
                    user_role=row["user_role"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        finally:
            await self._release_connection(conn)

    async def export_for_training(
        self,
        tenant_id: str,
        format: str = "jsonl",
        min_quality_delta: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Export feedback pairs in training format."""
        pairs = await self.get_pairs(tenant_id, limit=10000)

        training_data = []
        for pair in pairs:
            if pair._estimate_quality_delta() >= min_quality_delta:
                if format == "jsonl":
                    training_data.append(pair.to_training_format())
                elif format == "openai":
                    training_data.append(self._to_openai_format(pair))
                elif format == "anthropic":
                    training_data.append(self._to_anthropic_format(pair))

        return training_data

    def _to_openai_format(self, pair: FeedbackPair) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format."""
        return {
            "messages": [
                {"role": "system", "content": pair._build_system_prompt()},
                {"role": "user", "content": pair._build_user_prompt()},
                {"role": "assistant", "content": pair.human_edited_text},
            ]
        }

    def _to_anthropic_format(self, pair: FeedbackPair) -> Dict[str, Any]:
        """Convert to Anthropic fine-tuning format."""
        return {
            "prompt": f"\n\nHuman: {pair._build_system_prompt()}\n\n{pair._build_user_prompt()}\n\nAssistant:",
            "completion": f" {pair.human_edited_text}",
        }

    async def analyze(self, tenant_id: str) -> FeedbackAnalysis:
        """Analyze feedback patterns for a tenant."""
        pairs = await self.get_pairs(tenant_id, limit=10000)

        pairs_by_type: Dict[str, int] = {}
        pairs_by_section: Dict[str, int] = {}
        total_edit_distance = 0.0

        for pair in pairs:
            # Count by type
            type_key = pair.edit_type.value
            pairs_by_type[type_key] = pairs_by_type.get(type_key, 0) + 1

            # Count by section
            if pair.section_type:
                pairs_by_section[pair.section_type] = pairs_by_section.get(pair.section_type, 0) + 1

            total_edit_distance += pair.edit_distance

        # Most edited sections
        most_edited = sorted(pairs_by_section.items(), key=lambda x: x[1], reverse=True)[:5]

        # Common issues based on edit types
        common_issues = []
        if pairs_by_type.get("factual_fix", 0) > len(pairs) * 0.2:
            common_issues.append("High rate of factual corrections needed")
        if pairs_by_type.get("complete_rewrite", 0) > len(pairs) * 0.1:
            common_issues.append("Frequent complete rewrites suggest tone/style mismatch")
        if pairs_by_type.get("content_addition", 0) > len(pairs) * 0.3:
            common_issues.append("Model outputs often lack sufficient detail")

        return FeedbackAnalysis(
            total_pairs=len(pairs),
            pairs_by_type=pairs_by_type,
            pairs_by_section=pairs_by_section,
            average_edit_distance=total_edit_distance / len(pairs) if pairs else 0,
            most_edited_sections=[s[0] for s in most_edited],
            common_issues=common_issues,
        )


# Singleton instance
_feedback_collector: Optional[FeedbackCollector] = None


def get_feedback_collector(db_pool: Optional[Any] = None) -> FeedbackCollector:
    """Get or create feedback collector singleton."""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector(db_pool=db_pool)
    return _feedback_collector
