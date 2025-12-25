"""
PropelAI PostgreSQL Checkpoint Saver
LangGraph-compatible checkpoint persistence for durable agent workflows
"""

import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Iterator, Sequence
from dataclasses import dataclass, field

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointTuple,
        CheckpointMetadata,
    )
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Define fallback types for development
    BaseCheckpointSaver = object
    Checkpoint = dict
    CheckpointTuple = tuple
    CheckpointMetadata = dict


logger = logging.getLogger(__name__)


@dataclass
class CheckpointRecord:
    """Internal representation of a checkpoint."""
    thread_id: str
    checkpoint_id: str
    parent_checkpoint_id: Optional[str]
    checkpoint_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class PostgresCheckpointSaver(BaseCheckpointSaver if LANGGRAPH_AVAILABLE else object):
    """
    PostgreSQL-based checkpoint saver for LangGraph workflows.

    Provides durable state persistence for:
    - Long-running proposal generation workflows
    - Resume/pause capability for agent sessions
    - State rollback and branching
    - Cross-session state recovery
    """

    def __init__(
        self,
        db_pool: Optional[Any] = None,
        connection_string: Optional[str] = None,
        table_name: str = "checkpoints",
    ):
        """
        Initialize PostgreSQL checkpoint saver.

        Args:
            db_pool: Existing asyncpg pool (preferred)
            connection_string: PostgreSQL connection string (fallback)
            table_name: Name of checkpoints table
        """
        if LANGGRAPH_AVAILABLE:
            super().__init__()

        self.db_pool = db_pool
        self.connection_string = connection_string
        self.table_name = table_name
        self._connection = None
        self._is_setup = False

    async def _get_connection(self):
        """Get database connection."""
        if self.db_pool:
            return await self.db_pool.acquire()

        if self._connection is None and self.connection_string:
            import asyncpg
            self._connection = await asyncpg.connect(self.connection_string)

        return self._connection

    async def _release_connection(self, conn):
        """Release connection back to pool if using pool."""
        if self.db_pool and conn:
            await self.db_pool.release(conn)

    async def setup(self) -> None:
        """Ensure checkpoint table exists."""
        if self._is_setup:
            return

        conn = await self._get_connection()
        try:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    thread_id VARCHAR(255) NOT NULL,
                    checkpoint_id VARCHAR(255) NOT NULL,
                    parent_checkpoint_id VARCHAR(255),
                    checkpoint_data JSONB NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_thread_created
                ON {self.table_name}(thread_id, created_at DESC)
            """)

            self._is_setup = True
        finally:
            await self._release_connection(conn)

    # =========================================================================
    # LangGraph CheckpointSaver Interface
    # =========================================================================

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Synchronous wrapper for async get."""
        return asyncio.get_event_loop().run_until_complete(
            self.aget_tuple(config)
        )

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by config."""
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        if not thread_id:
            return None

        conn = await self._get_connection()
        try:
            if checkpoint_id:
                # Get specific checkpoint
                row = await conn.fetchrow(
                    f"""
                    SELECT thread_id, checkpoint_id, parent_checkpoint_id,
                           checkpoint_data, metadata, created_at
                    FROM {self.table_name}
                    WHERE thread_id = $1 AND checkpoint_id = $2
                    """,
                    thread_id, checkpoint_id
                )
            else:
                # Get latest checkpoint for thread
                row = await conn.fetchrow(
                    f"""
                    SELECT thread_id, checkpoint_id, parent_checkpoint_id,
                           checkpoint_data, metadata, created_at
                    FROM {self.table_name}
                    WHERE thread_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    thread_id
                )

            if not row:
                return None

            return self._row_to_tuple(row, config)
        finally:
            await self._release_connection(conn)

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Synchronous list of checkpoints."""
        async def _list():
            result = []
            async for item in self.alist(config, filter=filter, before=before, limit=limit):
                result.append(item)
            return result
        return iter(asyncio.get_event_loop().run_until_complete(_list()))

    async def alist(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ):
        """Async generator of checkpoints."""
        thread_id = config.get("configurable", {}).get("thread_id") if config else None

        conn = await self._get_connection()
        try:
            sql = f"""
                SELECT thread_id, checkpoint_id, parent_checkpoint_id,
                       checkpoint_data, metadata, created_at
                FROM {self.table_name}
                WHERE 1=1
            """
            params = []
            param_idx = 1

            if thread_id:
                sql += f" AND thread_id = ${param_idx}"
                params.append(thread_id)
                param_idx += 1

            if before:
                before_id = before.get("configurable", {}).get("checkpoint_id")
                if before_id:
                    sql += f" AND created_at < (SELECT created_at FROM {self.table_name} WHERE checkpoint_id = ${param_idx})"
                    params.append(before_id)
                    param_idx += 1

            sql += " ORDER BY created_at DESC"

            if limit:
                sql += f" LIMIT ${param_idx}"
                params.append(limit)

            rows = await conn.fetch(sql, *params)

            for row in rows:
                yield self._row_to_tuple(row, config or {})
        finally:
            await self._release_connection(conn)

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronous put."""
        return asyncio.get_event_loop().run_until_complete(
            self.aput(config, checkpoint, metadata, new_versions)
        )

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save a checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id")
        parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        if not thread_id:
            thread_id = str(uuid.uuid4())

        checkpoint_id = str(uuid.uuid4())

        conn = await self._get_connection()
        try:
            # Serialize checkpoint data
            checkpoint_data = self._serialize_checkpoint(checkpoint)
            metadata_dict = dict(metadata) if metadata else {}

            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (thread_id, checkpoint_id)
                DO UPDATE SET checkpoint_data = $4, metadata = $5
                """,
                thread_id, checkpoint_id, parent_checkpoint_id,
                json.dumps(checkpoint_data), json.dumps(metadata_dict)
            )

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }
        finally:
            await self._release_connection(conn)

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Put writes (for streaming updates)."""
        asyncio.get_event_loop().run_until_complete(
            self.aput_writes(config, writes, task_id)
        )

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async put writes for streaming updates."""
        # For now, writes are accumulated and saved with the checkpoint
        # In production, this could be optimized with a separate writes table
        pass

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _serialize_checkpoint(self, checkpoint: Any) -> Dict[str, Any]:
        """Serialize checkpoint to JSON-compatible dict."""
        if isinstance(checkpoint, dict):
            return checkpoint
        if hasattr(checkpoint, 'dict'):
            return checkpoint.dict()
        if hasattr(checkpoint, '__dict__'):
            return checkpoint.__dict__
        return {"data": str(checkpoint)}

    def _deserialize_checkpoint(self, data: Dict[str, Any]) -> Checkpoint:
        """Deserialize checkpoint data."""
        if LANGGRAPH_AVAILABLE:
            return data  # LangGraph handles reconstruction
        return data

    def _row_to_tuple(
        self,
        row: Any,
        config: Dict[str, Any],
    ) -> CheckpointTuple:
        """Convert database row to CheckpointTuple."""
        checkpoint_data = row["checkpoint_data"]
        if isinstance(checkpoint_data, str):
            checkpoint_data = json.loads(checkpoint_data)

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        new_config = {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_id": row["checkpoint_id"],
            }
        }

        parent_config = None
        if row["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }

        if LANGGRAPH_AVAILABLE:
            return CheckpointTuple(
                config=new_config,
                checkpoint=checkpoint_data,
                metadata=metadata,
                parent_config=parent_config,
            )
        else:
            return (new_config, checkpoint_data, metadata, parent_config)

    # =========================================================================
    # Extended API for PropelAI
    # =========================================================================

    async def get_thread_history(
        self,
        thread_id: str,
        limit: int = 20,
    ) -> List[CheckpointRecord]:
        """Get checkpoint history for a thread."""
        conn = await self._get_connection()
        try:
            rows = await conn.fetch(
                f"""
                SELECT thread_id, checkpoint_id, parent_checkpoint_id,
                       checkpoint_data, metadata, created_at
                FROM {self.table_name}
                WHERE thread_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                thread_id, limit
            )

            return [
                CheckpointRecord(
                    thread_id=row["thread_id"],
                    checkpoint_id=row["checkpoint_id"],
                    parent_checkpoint_id=row["parent_checkpoint_id"],
                    checkpoint_data=row["checkpoint_data"],
                    metadata=row["metadata"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        finally:
            await self._release_connection(conn)

    async def rollback_to_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> bool:
        """Rollback a thread to a specific checkpoint by deleting newer ones."""
        conn = await self._get_connection()
        try:
            # Get the checkpoint's timestamp
            row = await conn.fetchrow(
                f"SELECT created_at FROM {self.table_name} WHERE thread_id = $1 AND checkpoint_id = $2",
                thread_id, checkpoint_id
            )

            if not row:
                return False

            # Delete all checkpoints after this one
            await conn.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE thread_id = $1 AND created_at > $2
                """,
                thread_id, row["created_at"]
            )

            return True
        finally:
            await self._release_connection(conn)

    async def fork_thread(
        self,
        source_thread_id: str,
        new_thread_id: Optional[str] = None,
        from_checkpoint_id: Optional[str] = None,
    ) -> str:
        """Fork a thread to create a new branch."""
        new_thread_id = new_thread_id or str(uuid.uuid4())

        conn = await self._get_connection()
        try:
            # Get checkpoints to fork
            if from_checkpoint_id:
                sql = f"""
                    SELECT * FROM {self.table_name}
                    WHERE thread_id = $1 AND created_at <= (
                        SELECT created_at FROM {self.table_name}
                        WHERE thread_id = $1 AND checkpoint_id = $2
                    )
                    ORDER BY created_at
                """
                rows = await conn.fetch(sql, source_thread_id, from_checkpoint_id)
            else:
                sql = f"""
                    SELECT * FROM {self.table_name}
                    WHERE thread_id = $1
                    ORDER BY created_at
                """
                rows = await conn.fetch(sql, source_thread_id)

            # Create new thread with copied checkpoints
            for row in rows:
                new_checkpoint_id = str(uuid.uuid4())
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name}
                    (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    new_thread_id, new_checkpoint_id, None,
                    row["checkpoint_data"], row["metadata"]
                )

            return new_thread_id
        finally:
            await self._release_connection(conn)

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoints for a thread."""
        conn = await self._get_connection()
        try:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE thread_id = $1",
                thread_id
            )
            return "DELETE" in result
        finally:
            await self._release_connection(conn)

    async def get_active_threads(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get list of active threads with their latest checkpoint info."""
        conn = await self._get_connection()
        try:
            sql = f"""
                SELECT DISTINCT ON (thread_id)
                    thread_id,
                    checkpoint_id,
                    metadata,
                    created_at
                FROM {self.table_name}
            """

            params = []
            if tenant_id:
                sql += " WHERE metadata->>'tenant_id' = $1"
                params.append(tenant_id)

            sql += f" ORDER BY thread_id, created_at DESC LIMIT ${len(params) + 1}"
            params.append(limit)

            rows = await conn.fetch(sql, *params)

            return [
                {
                    "thread_id": row["thread_id"],
                    "checkpoint_id": row["checkpoint_id"],
                    "last_updated": row["created_at"].isoformat(),
                    "metadata": row["metadata"],
                }
                for row in rows
            ]
        finally:
            await self._release_connection(conn)


# Singleton instance
_checkpoint_saver: Optional[PostgresCheckpointSaver] = None


def get_checkpoint_saver(
    db_pool: Optional[Any] = None,
    connection_string: Optional[str] = None,
) -> PostgresCheckpointSaver:
    """Get or create checkpoint saver singleton."""
    global _checkpoint_saver
    if _checkpoint_saver is None:
        _checkpoint_saver = PostgresCheckpointSaver(
            db_pool=db_pool,
            connection_string=connection_string,
        )
    return _checkpoint_saver
