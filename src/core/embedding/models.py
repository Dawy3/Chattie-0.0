"""
Embedding Model Configuration.

FOCUS: Domain-matched model selection
MUST: Verify max tokens handle your chunks

Using OpenAI embedding models:
- text-embedding-3-small: Efficient, good balance of quality and cost
- text-embedding-3-large: Highest quality
- text-embedding-ada-002: Legacy model

CRITICAL: Index and query embeddings MUST use the same model.
Changing models requires full reindexing.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingModel:
    """
    Embedding model specification.

    Contains all configuration needed to use an embedding model
    consistently across indexing and querying.
    """

    # Model identification
    name: str
    version: str = "v1"

    # Model specifications
    dimensions: int = 1536
    max_tokens: int = 8191  # MUST handle your chunk size

    # Performance characteristics
    batch_size: int = 100  # Recommended batch size
    max_batch_size: int = 500  # Maximum safe batch size

    # API configuration
    api_key_env: str = "OPENAI_API_KEY"

    # Cost tracking (per 1M tokens)
    cost_per_million_tokens: float = 0.0

    # Additional model parameters
    normalize_embeddings: bool = True
    truncation_strategy: str = "end"  # end, start, middle

    # Metadata
    description: str = ""

    @property
    def model_id(self) -> str:
        """Unique model identifier for tracking."""
        return f"openai/{self.name}/{self.version}"

    def validate_chunk_size(self, chunk_tokens: int) -> bool:
        """
        Validate that chunk size fits within model's max tokens.

        MUST: Verify max tokens handle your chunks.
        """
        return chunk_tokens <= self.max_tokens

    def get_recommended_batch_size(self, avg_chunk_tokens: int) -> int:
        """
        Calculate recommended batch size based on chunk size.

        Larger chunks = smaller batches to avoid API limits.
        """
        # Estimate based on total tokens per batch
        # Most APIs have ~100k token limit per batch
        tokens_per_batch = 100_000
        estimated_batch = tokens_per_batch // max(avg_chunk_tokens, 100)

        return min(estimated_batch, self.max_batch_size)


SUPPORTED_MODELS: dict[str, EmbeddingModel] = {
    "openai/text-embedding-3-small": EmbeddingModel(
        name="text-embedding-3-small",
        version="v1",
        dimensions=1536,
        max_tokens=8191,
        batch_size=100,
        max_batch_size=500,
        cost_per_million_tokens=0.02,
        description="OpenAI's efficient embedding model. Good balance of quality and cost.",
    ),
    "openai/text-embedding-3-large": EmbeddingModel(
        name="text-embedding-3-large",
        version="v1",
        dimensions=3072,
        max_tokens=8191,
        batch_size=100,
        max_batch_size=500,
        cost_per_million_tokens=0.13,
        description="OpenAI's highest quality embedding model.",
    ),
    "openai/text-embedding-ada-002": EmbeddingModel(
        name="text-embedding-ada-002",
        version="v2",
        dimensions=1536,
        max_tokens=8191,
        batch_size=100,
        max_batch_size=500,
        cost_per_million_tokens=0.10,
        description="Legacy OpenAI model. Consider text-embedding-3-small instead.",
    ),
    "huggingface/minilm-l6-v2": EmbeddingModel(
    name="sentence-transformers/all-MiniLM-L6-v2",
    version="v1",
    dimensions=384,
    max_tokens=256,
    batch_size=64,
    max_batch_size=128,
    api_key_env="HF_TOKEN",
    cost_per_million_tokens=0.0,
    description="Lightweight, fast, low-quality baseline. Suitable for demos and small datasets only.",
    ),
}

# Default model
DEFAULT_MODEL = "openai/text-embedding-3-small"


def get_embedding_model(
    model_key: Optional[str] = None,
    custom_config: Optional[dict] = None,
) -> EmbeddingModel:
    """
    Get embedding model by key with optional customization.

    Args:
        model_key: Model identifier (e.g., "openai/text-embedding-3-small")
                   If None, returns the default model
        custom_config: Optional dict to override model settings

    Returns:
        Configured EmbeddingModel instance

    Example:
        >>> model = get_embedding_model("openai/text-embedding-3-small")
        >>> model.validate_chunk_size(512)
        True
    """
    if model_key is None:
        model_key = DEFAULT_MODEL

    if model_key not in SUPPORTED_MODELS:
        available = list(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Unknown model: {model_key}. Available models: {available}"
        )

    model = SUPPORTED_MODELS[model_key]

    # Apply custom configuration if provided
    if custom_config:
        model_dict = {
            "name": model.name,
            "version": model.version,
            "dimensions": model.dimensions,
            "max_tokens": model.max_tokens,
            "batch_size": model.batch_size,
            "max_batch_size": model.max_batch_size,
            "api_key_env": model.api_key_env,
            "cost_per_million_tokens": model.cost_per_million_tokens,
            "normalize_embeddings": model.normalize_embeddings,
            "truncation_strategy": model.truncation_strategy,
            "description": model.description,
        }
        model_dict.update(custom_config)
        model = EmbeddingModel(**model_dict)

    return model


def list_available_models() -> list[str]:
    """List all available model keys."""
    return list(SUPPORTED_MODELS.keys())


def validate_model_compatibility(
    index_model_id: str,
    query_model_id: str,
) -> bool:
    """
    CRITICAL: Validate that index and query models match.

    Mismatched models will produce garbage results.
    """
    return index_model_id == query_model_id
