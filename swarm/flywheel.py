"""
PropelAI v6.0 RLHF Data Flywheel
Continuous learning from proposal outcomes and human feedback.
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class OutcomeType(str, Enum):
    """Proposal outcome types"""
    WIN = "win"
    LOSS = "loss"
    NO_BID = "no_bid"
    PENDING = "pending"


class FeedbackType(str, Enum):
    """Types of feedback"""
    HUMAN_EDIT = "human_edit"  # Human edited the draft
    HUMAN_APPROVE = "human_approve"  # Human approved without changes
    HUMAN_REJECT = "human_reject"  # Human rejected the draft
    RED_TEAM_SCORE = "red_team_score"  # Red team scoring
    DEBRIEF = "debrief"  # Post-award debrief feedback


@dataclass
class TrajectoryStep:
    """Single step in a proposal trajectory"""
    agent: str
    action: str
    input_hash: str  # Hash of input state
    output_hash: str  # Hash of output state
    timestamp: str
    duration_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete trajectory for a proposal"""
    proposal_id: str
    rfp_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    outcome: OutcomeType = OutcomeType.PENDING
    final_score: float = 0.0
    created_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class FeedbackRecord:
    """Record of human feedback"""
    id: str
    proposal_id: str
    section_id: str
    feedback_type: FeedbackType
    original_text: str
    revised_text: Optional[str]  # For edits
    score: Optional[float]  # For scores
    comments: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WinningPattern:
    """Pattern extracted from winning proposals"""
    pattern_id: str
    pattern_type: str  # "win_theme", "structure", "language", "strategy"
    description: str
    examples: List[str]
    success_rate: float  # 0-1
    sample_size: int
    agency_context: Optional[str]  # e.g., "DoD", "HHS", "GSA"
    requirement_type: Optional[str]  # e.g., "technical", "past_performance"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataFlywheel:
    """
    RLHF Data Flywheel for continuous improvement.

    Captures:
    1. Agent trajectories (decision paths)
    2. Human feedback (edits, approvals, rejections)
    3. Proposal outcomes (win/loss with debrief notes)
    4. Red team scores over time

    Uses this data to:
    1. Identify winning patterns
    2. Fine-tune agent prompts
    3. Improve strategy recommendations
    4. Reduce revision cycles
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./flywheel_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self.trajectories: Dict[str, Trajectory] = {}
        self.feedback: List[FeedbackRecord] = []
        self.winning_patterns: List[WinningPattern] = []

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing flywheel data from storage"""
        trajectories_file = self.storage_path / "trajectories.jsonl"
        if trajectories_file.exists():
            with open(trajectories_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    traj = Trajectory(**data)
                    self.trajectories[traj.proposal_id] = traj

        feedback_file = self.storage_path / "feedback.jsonl"
        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.feedback.append(FeedbackRecord(**data))

        patterns_file = self.storage_path / "patterns.json"
        if patterns_file.exists():
            with open(patterns_file, "r") as f:
                patterns = json.load(f)
                self.winning_patterns = [WinningPattern(**p) for p in patterns]

    def _save_trajectory(self, trajectory: Trajectory):
        """Append trajectory to storage"""
        trajectories_file = self.storage_path / "trajectories.jsonl"
        with open(trajectories_file, "a") as f:
            f.write(json.dumps(asdict(trajectory)) + "\n")

    def _save_feedback(self, record: FeedbackRecord):
        """Append feedback to storage"""
        feedback_file = self.storage_path / "feedback.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def _save_patterns(self):
        """Save winning patterns"""
        patterns_file = self.storage_path / "patterns.json"
        with open(patterns_file, "w") as f:
            json.dump([asdict(p) for p in self.winning_patterns], f, indent=2)

    # =========================================================================
    # Trajectory Recording
    # =========================================================================

    def start_trajectory(self, proposal_id: str, rfp_id: str) -> Trajectory:
        """Start recording a new proposal trajectory"""
        trajectory = Trajectory(proposal_id=proposal_id, rfp_id=rfp_id)
        self.trajectories[proposal_id] = trajectory
        logger.info(f"Started trajectory for {proposal_id}")
        return trajectory

    def record_step(
        self,
        proposal_id: str,
        agent: str,
        action: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        duration_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a step in the trajectory"""
        if proposal_id not in self.trajectories:
            logger.warning(f"No trajectory found for {proposal_id}")
            return

        # Hash states for compact storage
        input_hash = hashlib.md5(json.dumps(input_state, sort_keys=True).encode()).hexdigest()[:16]
        output_hash = hashlib.md5(json.dumps(output_state, sort_keys=True).encode()).hexdigest()[:16]

        step = TrajectoryStep(
            agent=agent,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self.trajectories[proposal_id].steps.append(step)

    def complete_trajectory(
        self,
        proposal_id: str,
        outcome: OutcomeType,
        final_score: float,
    ):
        """Mark trajectory as complete with outcome"""
        if proposal_id not in self.trajectories:
            logger.warning(f"No trajectory found for {proposal_id}")
            return

        trajectory = self.trajectories[proposal_id]
        trajectory.outcome = outcome
        trajectory.final_score = final_score
        trajectory.completed_at = datetime.utcnow().isoformat()

        self._save_trajectory(trajectory)
        logger.info(f"Completed trajectory for {proposal_id}: {outcome.value}")

    # =========================================================================
    # Feedback Recording
    # =========================================================================

    def record_feedback(
        self,
        proposal_id: str,
        section_id: str,
        feedback_type: FeedbackType,
        original_text: str,
        revised_text: Optional[str] = None,
        score: Optional[float] = None,
        comments: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """Record human feedback on a proposal section"""
        record = FeedbackRecord(
            id=f"FB-{hashlib.md5(f'{proposal_id}{section_id}{datetime.utcnow()}'.encode()).hexdigest()[:8]}",
            proposal_id=proposal_id,
            section_id=section_id,
            feedback_type=feedback_type,
            original_text=original_text,
            revised_text=revised_text,
            score=score,
            comments=comments,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )

        self.feedback.append(record)
        self._save_feedback(record)

        logger.info(f"Recorded {feedback_type.value} feedback for {proposal_id}/{section_id}")
        return record

    def record_human_edit(
        self,
        proposal_id: str,
        section_id: str,
        original_text: str,
        revised_text: str,
        comments: str = "",
    ) -> FeedbackRecord:
        """Convenience method for recording human edits"""
        return self.record_feedback(
            proposal_id=proposal_id,
            section_id=section_id,
            feedback_type=FeedbackType.HUMAN_EDIT,
            original_text=original_text,
            revised_text=revised_text,
            comments=comments,
        )

    def record_red_team_score(
        self,
        proposal_id: str,
        section_id: str,
        score: float,
        feedback_text: str,
        strengths: List[str] = None,
        weaknesses: List[str] = None,
    ) -> FeedbackRecord:
        """Record red team scoring feedback"""
        return self.record_feedback(
            proposal_id=proposal_id,
            section_id=section_id,
            feedback_type=FeedbackType.RED_TEAM_SCORE,
            original_text="",
            score=score,
            comments=feedback_text,
            metadata={
                "strengths": strengths or [],
                "weaknesses": weaknesses or [],
            },
        )

    def record_debrief(
        self,
        proposal_id: str,
        outcome: OutcomeType,
        evaluator_feedback: str,
        strengths: List[str] = None,
        weaknesses: List[str] = None,
        competitor_info: Optional[Dict[str, Any]] = None,
    ):
        """Record post-award debrief information"""
        # Update trajectory outcome
        if proposal_id in self.trajectories:
            self.trajectories[proposal_id].outcome = outcome
            self._save_trajectory(self.trajectories[proposal_id])

        # Record debrief as feedback
        self.record_feedback(
            proposal_id=proposal_id,
            section_id="DEBRIEF",
            feedback_type=FeedbackType.DEBRIEF,
            original_text="",
            comments=evaluator_feedback,
            metadata={
                "outcome": outcome.value,
                "strengths": strengths or [],
                "weaknesses": weaknesses or [],
                "competitor_info": competitor_info or {},
            },
        )

    # =========================================================================
    # Pattern Extraction
    # =========================================================================

    def extract_winning_patterns(self) -> List[WinningPattern]:
        """
        Analyze winning proposals to extract patterns.

        Looks for:
        - Common win themes
        - Structural patterns
        - Language that correlates with wins
        - Strategies that worked
        """
        winning_trajectories = [
            t for t in self.trajectories.values()
            if t.outcome == OutcomeType.WIN
        ]

        if len(winning_trajectories) < 3:
            logger.info("Not enough winning proposals for pattern extraction")
            return []

        patterns = []

        # Pattern 1: Agent sequence patterns
        sequence_counts: Dict[str, int] = {}
        for traj in winning_trajectories:
            seq = "->".join([s.agent for s in traj.steps])
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1

        most_common_seq = max(sequence_counts.items(), key=lambda x: x[1])
        if most_common_seq[1] >= 2:
            patterns.append(WinningPattern(
                pattern_id=f"SEQ-{hashlib.md5(most_common_seq[0].encode()).hexdigest()[:8]}",
                pattern_type="structure",
                description=f"Common winning agent sequence: {most_common_seq[0]}",
                examples=[most_common_seq[0]],
                success_rate=most_common_seq[1] / len(winning_trajectories),
                sample_size=len(winning_trajectories),
            ))

        # Pattern 2: High-scoring sections
        winning_feedback = [
            f for f in self.feedback
            if f.proposal_id in [t.proposal_id for t in winning_trajectories]
            and f.feedback_type == FeedbackType.RED_TEAM_SCORE
            and f.score and f.score >= 0.9
        ]

        section_scores: Dict[str, List[float]] = {}
        for f in winning_feedback:
            if f.section_id not in section_scores:
                section_scores[f.section_id] = []
            section_scores[f.section_id].append(f.score)

        for section_id, scores in section_scores.items():
            if len(scores) >= 2:
                avg_score = sum(scores) / len(scores)
                patterns.append(WinningPattern(
                    pattern_id=f"SCORE-{section_id[:8]}",
                    pattern_type="quality",
                    description=f"Section {section_id} consistently scores high in wins",
                    examples=[],
                    success_rate=avg_score,
                    sample_size=len(scores),
                    requirement_type=section_id,
                ))

        self.winning_patterns = patterns
        self._save_patterns()

        logger.info(f"Extracted {len(patterns)} winning patterns")
        return patterns

    # =========================================================================
    # Learning & Recommendations
    # =========================================================================

    def get_recommendations(
        self,
        rfp_context: Dict[str, Any],
        section_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on historical data.

        Args:
            rfp_context: Context about the current RFP (agency, type, etc.)
            section_type: Optional specific section type

        Returns:
            List of recommendations with confidence scores
        """
        recommendations = []

        # Get relevant winning patterns
        for pattern in self.winning_patterns:
            relevance = 1.0

            # Adjust relevance based on context match
            if pattern.agency_context and rfp_context.get("agency"):
                if pattern.agency_context == rfp_context["agency"]:
                    relevance *= 1.5
                else:
                    relevance *= 0.7

            if section_type and pattern.requirement_type:
                if pattern.requirement_type == section_type:
                    relevance *= 1.3

            if relevance > 0.5:
                recommendations.append({
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": min(pattern.success_rate * relevance, 1.0),
                    "sample_size": pattern.sample_size,
                })

        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)

        return recommendations[:10]

    def get_similar_wins(
        self,
        rfp_id: str,
        limit: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find similar winning proposals.

        Returns list of (proposal_id, similarity_score) tuples.
        """
        # Simple similarity based on trajectory length and outcome
        # In production, use embeddings for semantic similarity
        winning = [
            t for t in self.trajectories.values()
            if t.outcome == OutcomeType.WIN
        ]

        if not winning:
            return []

        # Sort by score
        winning.sort(key=lambda t: t.final_score, reverse=True)

        return [(t.proposal_id, t.final_score) for t in winning[:limit]]

    def get_improvement_suggestions(
        self,
        proposal_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get improvement suggestions based on feedback history.
        """
        suggestions = []

        # Get feedback for this proposal
        proposal_feedback = [
            f for f in self.feedback
            if f.proposal_id == proposal_id
        ]

        # Analyze edits to find common issues
        edits = [f for f in proposal_feedback if f.feedback_type == FeedbackType.HUMAN_EDIT]
        for edit in edits:
            if edit.comments:
                suggestions.append({
                    "type": "edit_based",
                    "section": edit.section_id,
                    "suggestion": f"Previous edit reason: {edit.comments}",
                    "priority": "medium",
                })

        # Analyze low red team scores
        low_scores = [
            f for f in proposal_feedback
            if f.feedback_type == FeedbackType.RED_TEAM_SCORE
            and f.score and f.score < 0.7
        ]
        for score_fb in low_scores:
            weaknesses = score_fb.metadata.get("weaknesses", [])
            for weakness in weaknesses:
                suggestions.append({
                    "type": "red_team",
                    "section": score_fb.section_id,
                    "suggestion": weakness,
                    "priority": "high",
                })

        return suggestions

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get flywheel statistics"""
        total_proposals = len(self.trajectories)
        wins = len([t for t in self.trajectories.values() if t.outcome == OutcomeType.WIN])
        losses = len([t for t in self.trajectories.values() if t.outcome == OutcomeType.LOSS])

        return {
            "total_proposals": total_proposals,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_proposals if total_proposals > 0 else 0,
            "total_feedback_records": len(self.feedback),
            "human_edits": len([f for f in self.feedback if f.feedback_type == FeedbackType.HUMAN_EDIT]),
            "red_team_scores": len([f for f in self.feedback if f.feedback_type == FeedbackType.RED_TEAM_SCORE]),
            "patterns_extracted": len(self.winning_patterns),
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_flywheel(storage_path: Optional[str] = None) -> DataFlywheel:
    """Create a new data flywheel instance"""
    return DataFlywheel(storage_path)
