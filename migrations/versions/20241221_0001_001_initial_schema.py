"""Initial schema - PropelAI v4.1

Revision ID: 001
Revises: None
Create Date: 2024-12-21

This migration creates the initial database schema for PropelAI v4.1,
including all tables for RFPs, users, teams, authentication, and vector search.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""

    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # =========================================================================
    # Users table
    # =========================================================================
    op.create_table(
        "users",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("email_verified", sa.Boolean(), default=False),
        sa.Column("email_verification_token", sa.String(255), nullable=True),
        sa.Column("email_verification_sent_at", sa.DateTime(), nullable=True),
        sa.Column("totp_secret", sa.String(32), nullable=True),
        sa.Column("totp_enabled", sa.Boolean(), default=False),
        sa.Column("backup_codes", postgresql.JSONB(), nullable=True),
        sa.Column("failed_login_attempts", sa.Integer(), default=0),
        sa.Column("locked_until", sa.DateTime(), nullable=True),
        sa.Column("password_reset_token", sa.String(255), nullable=True),
        sa.Column("password_reset_expires", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("idx_users_email", "users", ["email"])

    # =========================================================================
    # Teams table
    # =========================================================================
    op.create_table(
        "teams",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("settings", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("idx_teams_slug", "teams", ["slug"])

    # =========================================================================
    # Team Memberships table
    # =========================================================================
    op.create_table(
        "team_memberships",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("team_id", sa.String(50), sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(50), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, default="viewer"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("team_id", "user_id", name="uq_team_user"),
    )
    op.create_index("idx_memberships_team", "team_memberships", ["team_id"])
    op.create_index("idx_memberships_user", "team_memberships", ["user_id"])

    # =========================================================================
    # User Sessions table
    # =========================================================================
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("user_id", sa.String(50), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(255), nullable=False),
        sa.Column("device_info", sa.String(255), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("last_active", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_sessions_user", "user_sessions", ["user_id"])
    op.create_index("idx_sessions_token", "user_sessions", ["token_hash"])

    # =========================================================================
    # API Keys table
    # =========================================================================
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("team_id", sa.String(50), sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by", sa.String(50), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("key_hash", sa.String(255), nullable=False),
        sa.Column("key_prefix", sa.String(10), nullable=False),
        sa.Column("scopes", postgresql.JSONB(), default=[]),
        sa.Column("last_used", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_api_keys_team", "api_keys", ["team_id"])

    # =========================================================================
    # Team Invitations table
    # =========================================================================
    op.create_table(
        "team_invitations",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("team_id", sa.String(50), sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, default="viewer"),
        sa.Column("token", sa.String(255), unique=True, nullable=False),
        sa.Column("invited_by", sa.String(50), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_invitations_token", "team_invitations", ["token"])
    op.create_index("idx_invitations_team", "team_invitations", ["team_id"])

    # =========================================================================
    # RFPs table
    # =========================================================================
    op.create_table(
        "rfps",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("solicitation_number", sa.String(100), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, default="created"),
        sa.Column("progress", sa.Integer(), default=0),
        sa.Column("outline", postgresql.JSONB(), default={}),
        sa.Column("stats", postgresql.JSONB(), default={}),
        sa.Column("document_metadata", postgresql.JSONB(), default={}),
        sa.Column("is_deleted", sa.Boolean(), default=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column("deleted_by", sa.String(50), nullable=True),
        sa.Column("delete_reason", sa.Text(), nullable=True),
        sa.Column("permanent_delete_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("idx_rfps_status", "rfps", ["status"])
    op.create_index("idx_rfps_deleted", "rfps", ["is_deleted"])

    # =========================================================================
    # Requirements table
    # =========================================================================
    op.create_table(
        "requirements",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("rfp_id", sa.String(50), sa.ForeignKey("rfps.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("section", sa.String(100), nullable=True),
        sa.Column("priority", sa.String(20), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("source_page", sa.Integer(), nullable=True),
        sa.Column("source_doc", sa.String(255), nullable=True),
        sa.Column("coordinates", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_requirements_rfp", "requirements", ["rfp_id"])
    op.create_index("idx_requirements_section", "requirements", ["section"])

    # =========================================================================
    # Amendments table
    # =========================================================================
    op.create_table(
        "amendments",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("rfp_id", sa.String(50), sa.ForeignKey("rfps.id", ondelete="CASCADE"), nullable=False),
        sa.Column("amendment_number", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("changes", postgresql.JSONB(), default=[]),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("processed_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_amendments_rfp", "amendments", ["rfp_id"])

    # =========================================================================
    # Activity Log table
    # =========================================================================
    op.create_table(
        "activity_log",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("team_id", sa.String(50), sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=True),
        sa.Column("user_id", sa.String(50), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=True),
        sa.Column("resource_id", sa.String(50), nullable=True),
        sa.Column("details", postgresql.JSONB(), default={}),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(500), nullable=True),
        sa.Column("request_id", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_activity_team", "activity_log", ["team_id"])
    op.create_index("idx_activity_user", "activity_log", ["user_id"])
    op.create_index("idx_activity_action", "activity_log", ["action"])
    op.create_index("idx_activity_created", "activity_log", ["created_at"])

    # =========================================================================
    # Webhooks table
    # =========================================================================
    op.create_table(
        "webhooks",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("team_id", sa.String(50), sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("url", sa.String(2000), nullable=False),
        sa.Column("secret", sa.String(255), nullable=True),
        sa.Column("events", postgresql.JSONB(), default=[]),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("retry_count", sa.Integer(), default=3),
        sa.Column("timeout_seconds", sa.Integer(), default=30),
        sa.Column("success_count", sa.Integer(), default=0),
        sa.Column("failure_count", sa.Integer(), default=0),
        sa.Column("last_triggered", sa.DateTime(), nullable=True),
        sa.Column("last_success", sa.DateTime(), nullable=True),
        sa.Column("last_failure", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("idx_webhooks_team", "webhooks", ["team_id"])

    # =========================================================================
    # Webhook Deliveries table
    # =========================================================================
    op.create_table(
        "webhook_deliveries",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("webhook_id", sa.String(50), sa.ForeignKey("webhooks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), default=1),
        sa.Column("delivered_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_deliveries_webhook", "webhook_deliveries", ["webhook_id"])
    op.create_index("idx_deliveries_delivered", "webhook_deliveries", ["delivered_at"])

    # =========================================================================
    # LangGraph Checkpoints table (for workflow state persistence)
    # =========================================================================
    op.create_table(
        "checkpoints",
        sa.Column("thread_id", sa.String(255), primary_key=True),
        sa.Column("thread_ts", sa.String(255), primary_key=True),
        sa.Column("parent_ts", sa.String(255), nullable=True),
        sa.Column("checkpoint", postgresql.JSONB(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), default={}),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # =========================================================================
    # Company Library - Vector Search Tables
    # =========================================================================

    # Company Profiles
    op.create_table(
        "company_profiles",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("team_id", sa.String(50), sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Capabilities (with vector embeddings)
    op.execute("""
        CREATE TABLE capabilities (
            id VARCHAR(50) PRIMARY KEY,
            profile_id VARCHAR(50) REFERENCES company_profiles(id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            category VARCHAR(100),
            evidence TEXT,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_capabilities_embedding ON capabilities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

    # Past Performance (with vector embeddings)
    op.execute("""
        CREATE TABLE past_performances (
            id VARCHAR(50) PRIMARY KEY,
            profile_id VARCHAR(50) REFERENCES company_profiles(id) ON DELETE CASCADE,
            project_name VARCHAR(255) NOT NULL,
            client VARCHAR(255),
            contract_value DECIMAL(15, 2),
            start_date DATE,
            end_date DATE,
            description TEXT NOT NULL,
            outcomes TEXT,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_past_performance_embedding ON past_performances USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

    # Key Personnel (with vector embeddings)
    op.execute("""
        CREATE TABLE key_personnel (
            id VARCHAR(50) PRIMARY KEY,
            profile_id VARCHAR(50) REFERENCES company_profiles(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            title VARCHAR(255),
            qualifications TEXT NOT NULL,
            experience_years INTEGER,
            certifications TEXT,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_key_personnel_embedding ON key_personnel USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

    # Differentiators (with vector embeddings)
    op.execute("""
        CREATE TABLE differentiators (
            id VARCHAR(50) PRIMARY KEY,
            profile_id VARCHAR(50) REFERENCES company_profiles(id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            proof_points TEXT,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_differentiators_embedding ON differentiators USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

    # =========================================================================
    # Automatic updated_at trigger
    # =========================================================================
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    # Apply trigger to tables with updated_at
    for table in ["users", "teams", "rfps", "webhooks", "company_profiles"]:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)


def downgrade() -> None:
    """Drop all tables."""
    # Drop triggers first
    for table in ["users", "teams", "rfps", "webhooks", "company_profiles"]:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")

    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop tables in reverse dependency order
    op.drop_table("differentiators")
    op.drop_table("key_personnel")
    op.drop_table("past_performances")
    op.drop_table("capabilities")
    op.drop_table("company_profiles")
    op.drop_table("checkpoints")
    op.drop_table("webhook_deliveries")
    op.drop_table("webhooks")
    op.drop_table("activity_log")
    op.drop_table("amendments")
    op.drop_table("requirements")
    op.drop_table("rfps")
    op.drop_table("team_invitations")
    op.drop_table("api_keys")
    op.drop_table("user_sessions")
    op.drop_table("team_memberships")
    op.drop_table("teams")
    op.drop_table("users")

    # Drop extension
    op.execute("DROP EXTENSION IF EXISTS vector")
