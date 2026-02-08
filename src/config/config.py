"""
Configuration for RAG Backend.

Field names match .env variable names (UPPERCASE).
Lowercase property aliases for code convenience.

Example .env:
    OPENAI_API_KEY=sk-...
    LLM_MODEL=gpt-4o-mini
    EMBEDDING_MODEL_NAME=text-embedding-3-small
    QDRANT_URL=http://localhost:6333
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# EMBEDDING
# =============================================================================
class EmbeddingConfig(BaseSettings):
    """OpenAI embedding model settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-3-small")
    EMBEDDING_MODEL_PROVIDER: str = Field(default="openai")
    EMBEDDING_DIMENSIONS: int = Field(default=1536)
    EMBEDDING_BATCH_SIZE: int = Field(default=100)
    EMBEDDING_MAX_RETRIES: int = Field(default=3)

    @property
    def model_name(self) -> str:
        return self.EMBEDDING_MODEL_NAME

    @property
    def model_provider(self) -> str:
        return self.EMBEDDING_MODEL_PROVIDER

    @property
    def dimensions(self) -> int:
        return self.EMBEDDING_DIMENSIONS

    @property
    def batch_size(self) -> int:
        return self.EMBEDDING_BATCH_SIZE

    @property
    def max_retries(self) -> int:
        return self.EMBEDDING_MAX_RETRIES


# =============================================================================
# CHUNKING
# =============================================================================
class ChunkingConfig(BaseSettings):
    """Text chunking settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    CHUNK_STRATEGY: str = Field(default="recursive")
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=50)

    @property
    def strategy(self) -> str:
        return self.CHUNK_STRATEGY

    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def chunk_overlap(self) -> int:
        return self.CHUNK_OVERLAP


# =============================================================================
# LLM
# =============================================================================
class LLMConfig(BaseSettings):
    """LLM provider settings.

    Works with any OpenAI-compatible API (OpenAI, OpenRouter, local).
    Set LLM_BASE_URL to switch providers.

    .env examples:
        # OpenAI (default)
        OPENAI_API_KEY=sk-...
        LLM_MODEL=gpt-4o-mini

        # OpenRouter
        OPENAI_API_KEY=sk-or-...
        LLM_BASE_URL=https://openrouter.ai/api/v1
        LLM_MODEL=openai/gpt-4o-mini

        # Local (Ollama, LM Studio, etc.)
        LLM_BASE_URL=http://localhost:11434/v1
        LLM_MODEL=llama3
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: Optional[str] = Field(default=None)
    LLM_BASE_URL: Optional[str] = Field(default=None)
    LLM_MODEL: str = Field(default="gpt-4o-mini")
    LLM_MAX_TOKENS: int = Field(default=1024)
    LLM_TEMPERATURE: float = Field(default=0.7)
    LLM_TIMEOUT: float = Field(default=60.0)
    LLM_MAX_RETRIES: int = Field(default=3)

    @property
    def api_key(self) -> Optional[str]:
        return self.OPENAI_API_KEY

    @property
    def base_url(self) -> Optional[str]:
        return self.LLM_BASE_URL

    @property
    def model(self) -> str:
        return self.LLM_MODEL

    @property
    def max_tokens(self) -> int:
        return self.LLM_MAX_TOKENS

    @property
    def temperature(self) -> float:
        return self.LLM_TEMPERATURE

    @property
    def request_timeout(self) -> float:
        return self.LLM_TIMEOUT

    @property
    def max_retries(self) -> int:
        return self.LLM_MAX_RETRIES


# =============================================================================
# QDRANT
# =============================================================================
class QdrantConfig(BaseSettings):
    """Qdrant vector store settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    QDRANT_COLLECTION: str = Field(default="documents")

    @property
    def qdrant_url(self) -> str:
        return self.QDRANT_URL

    @property
    def qdrant_api_key(self) -> Optional[str]:
        return self.QDRANT_API_KEY

    @property
    def qdrant_collection(self) -> str:
        return self.QDRANT_COLLECTION


# =============================================================================
# CACHE
# =============================================================================
class CacheConfig(BaseSettings):
    """Embedding cache toggle."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    CACHE_EMBEDDING_ENABLED: bool = Field(default=True)

    @property
    def embedding_cache_enabled(self) -> bool:
        return self.CACHE_EMBEDDING_ENABLED


# =============================================================================
# MAIN SETTINGS
# =============================================================================
class Settings(BaseSettings):
    """Application settings — the single source of truth."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    APP_NAME: str = Field(default="RAG Knowledge Assistant")
    APP_VERSION: str = Field(default="1.0.0")
    APP_ENV: str = Field(default="development")
    DEBUG: bool = Field(default=False)

    # API server
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default="/api/v1")

    # CORS (comma-separated origins, or "*")
    CORS_ORIGINS: str = Field(default="*")

    # Sub-configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Optional[Path] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"

    # Lowercase aliases
    @property
    def app_name(self) -> str:
        return self.APP_NAME

    @property
    def app_version(self) -> str:
        return self.APP_VERSION

    @property
    def debug(self) -> bool:
        return self.DEBUG

    @property
    def api_host(self) -> str:
        return self.API_HOST

    @property
    def api_port(self) -> int:
        return self.API_PORT

    @property
    def api_prefix(self) -> str:
        return self.API_PREFIX

    @property
    def cors_origins(self) -> list:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
