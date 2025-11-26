"""
PropelAI Configuration
Environment-based configuration for the APOS system
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class Environment(str, Enum):
    """Deployment environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Available LLM providers"""
    GOOGLE = "google"           # Gemini - Primary (2M context)
    ANTHROPIC = "anthropic"     # Claude - Complex reasoning
    OPENAI = "openai"           # GPT-4 - Fallback


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider = LLMProvider.GOOGLE
    
    # Google/Gemini settings
    google_api_key: Optional[str] = None
    google_project_id: Optional[str] = None
    gemini_flash_model: str = "gemini-1.5-flash"      # For extraction (cheap/fast)
    gemini_pro_model: str = "gemini-1.5-pro"          # For reasoning (powerful)
    
    # Anthropic/Claude settings
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-3-5-sonnet-20241022"
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    
    # Model cascading thresholds
    use_flash_for_extraction: bool = True
    use_pro_for_reasoning: bool = True
    
    # Token limits
    max_context_tokens: int = 1_000_000  # Gemini 1.5 Pro limit
    max_output_tokens: int = 8192
    
    def __post_init__(self):
        """Load from environment variables"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY", self.google_api_key)
        self.google_project_id = os.getenv("GOOGLE_PROJECT_ID", self.google_project_id)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)


@dataclass
class DatabaseConfig:
    """Configuration for database (checkpointing)"""
    # PostgreSQL for LangGraph checkpointing
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "propelai"
    postgres_user: str = "propelai"
    postgres_password: str = ""
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    
    def __post_init__(self):
        """Load from environment variables"""
        self.postgres_host = os.getenv("POSTGRES_HOST", self.postgres_host)
        self.postgres_port = int(os.getenv("POSTGRES_PORT", self.postgres_port))
        self.postgres_db = os.getenv("POSTGRES_DB", self.postgres_db)
        self.postgres_user = os.getenv("POSTGRES_USER", self.postgres_user)
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", self.postgres_password)
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


@dataclass
class VectorStoreConfig:
    """Configuration for vector database"""
    provider: str = "chroma"  # "chroma", "pinecone", "qdrant"
    
    # Chroma settings (default for development)
    chroma_persist_directory: str = "./data/chroma"
    
    # Pinecone settings (for production)
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "propelai-proposals"
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    def __post_init__(self):
        """Load from environment variables"""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", self.pinecone_api_key)


@dataclass
class SecurityConfig:
    """Security configuration for GovCon compliance"""
    # FedRAMP compliance settings
    fedramp_mode: bool = False
    data_encryption_at_rest: bool = True
    data_encryption_in_transit: bool = True
    
    # Audit logging
    enable_audit_log: bool = True
    audit_log_retention_days: int = 365
    
    # Data sovereignty
    allowed_regions: list = field(default_factory=lambda: ["us-east-1", "us-west-2"])
    
    # API security
    api_key_required: bool = True
    rate_limit_per_minute: int = 100


@dataclass
class AgentConfig:
    """Configuration for agent behavior"""
    # Compliance Agent
    compliance_confidence_threshold: float = 0.75
    max_requirements_per_batch: int = 100
    
    # Strategy Agent
    max_win_themes: int = 7
    enable_competitor_ghosting: bool = True
    
    # Drafting Agent
    default_voice_style: str = "formal"
    citation_required: bool = True
    max_uncited_claims: int = 0  # Zero tolerance
    
    # Red Team Agent
    auto_remediation_suggestions: bool = True
    score_threshold_for_submission: float = 70.0  # Minimum score to proceed
    
    # General
    max_retries: int = 3
    timeout_seconds: int = 300
    enable_human_in_the_loop: bool = True


@dataclass
class APOSConfig:
    """Master configuration for PropelAI APOS"""
    environment: Environment = Environment.DEVELOPMENT
    
    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # File storage
    upload_directory: str = "./data/uploads"
    output_directory: str = "./data/outputs"
    max_upload_size_mb: int = 100
    
    def __post_init__(self):
        """Load environment from env var"""
        env_str = os.getenv("PROPELAI_ENV", "development")
        try:
            self.environment = Environment(env_str.lower())
        except ValueError:
            self.environment = Environment.DEVELOPMENT
        
        self.api_host = os.getenv("API_HOST", self.api_host)
        self.api_port = int(os.getenv("API_PORT", self.api_port))
    
    @classmethod
    def from_env(cls) -> "APOSConfig":
        """Create configuration from environment variables"""
        return cls(
            llm=LLMConfig(),
            database=DatabaseConfig(),
            vector_store=VectorStoreConfig(),
            security=SecurityConfig(),
            agents=AgentConfig()
        )
    
    def validate(self) -> list:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check LLM credentials
        if self.llm.provider == LLMProvider.GOOGLE and not self.llm.google_api_key:
            issues.append("GOOGLE_API_KEY not set")
        if self.llm.provider == LLMProvider.ANTHROPIC and not self.llm.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY not set")
        if self.llm.provider == LLMProvider.OPENAI and not self.llm.openai_api_key:
            issues.append("OPENAI_API_KEY not set")
        
        # Check database in production
        if self.environment == Environment.PRODUCTION:
            if not self.database.postgres_password:
                issues.append("POSTGRES_PASSWORD not set for production")
        
        return issues


# Global configuration instance
_config: Optional[APOSConfig] = None


def get_config() -> APOSConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = APOSConfig.from_env()
    return _config


def set_config(config: APOSConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config
