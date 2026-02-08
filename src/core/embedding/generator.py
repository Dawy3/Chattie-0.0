"""
Embedding Generation for RAG Pipeline.

FOCUS: Batch embeddings (100-500 per call)
CRITICAL: NEVER mix models (index + query must match)
MUST: Cache embeddings for unchanged docs

Implementation follows best practices:
1. Batch texts for efficient API usage
2. Cache embeddings keyed by content hash + model ID
3. Retry with exponential backoff
4. Track model version for consistency

Uses OpenAI embeddings exclusively.
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from src.config import settings
from ..caching.embedding_cache import EmbeddingCache
from .models import EmbeddingModel, get_embedding_model, DEFAULT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embeddings: list[list[float]]
    model_id: str
    dimensions: int

    # Metadata
    texts_count: int = 0
    total_tokens: int = 0
    cached_count: int = 0
    generated_count: int = 0

    # Performance
    latency_ms: float = 0.0

    # Cost tracking
    estimated_cost: float = 0.0


class OpenAIEmbeddingClient:
    """OpenAI embedding API client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[AsyncOpenAI] = None

    async def _get_client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModel,
    ) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        client = await self._get_client()

        response = await client.embeddings.create(
            model=model.name,
            input=texts,
            encoding_format="float",
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


class EmbeddingGenerator:
    """
    Main embedding generator with batching, caching, and model consistency.

    FOCUS: Batch embeddings (100-500 per call)
    CRITICAL: NEVER mix models (index + query must match)
    MUST: Cache embeddings for unchanged docs

    Usage:
        generator = EmbeddingGenerator(model_key="openai/text-embedding-3-small")
        result = await generator.embed_texts(texts)
    """

    def __init__(
        self,
        model_key: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        enable_cache: Optional[bool] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 1.0,
    ):
        """
        Initialize embedding generator.

        Uses unified EmbeddingCache from core.caching.embedding_cache.
        Supports both Redis (persistent) and in-memory caching.

        Args:
            model_key: Model identifier from supported models (defaults to config)
            cache: Optional EmbeddingCache instance (Redis or in-memory)
                   If None and caching enabled, creates in-memory cache
            enable_cache: Whether to use caching (defaults to config)
            batch_size: Override default batch size (defaults to config)
            max_retries: Maximum retry attempts (defaults to config)
            retry_delay: Initial retry delay in seconds

        Example:
            # In-memory cache for initial ingestion
            generator = EmbeddingGenerator()  # Auto-creates in-memory cache

            # Redis cache for production
            from core.caching.embedding_cache import EmbeddingCache
            import redis
            redis_client = redis.Redis(host='localhost', port=6379)
            cache = EmbeddingCache(redis_client=redis_client)
            generator = EmbeddingGenerator(cache=cache)
        """
        # Use config defaults if not specified
        if model_key is None:
            model_key = getattr(settings.embedding, 'model_key', None)
            if model_key is None:
                provider = getattr(settings.embedding, 'model_provider', 'openai')
                name = getattr(settings.embedding, 'model_name', 'text-embedding-3-small')
                model_key = f"{provider}/{name}"

        if enable_cache is None:
            enable_cache = getattr(settings.cache, 'embedding_cache_enabled', True)
        if batch_size is None:
            batch_size = getattr(settings.embedding, 'batch_size', 100)
        if max_retries is None:
            max_retries = getattr(settings.embedding, 'max_retries', 3)

        self.model = get_embedding_model(model_key)
        self.enable_cache = enable_cache
        self.batch_size = batch_size or self.model.batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize unified EmbeddingCache
        self._cache = cache if enable_cache else None
        if enable_cache and cache is None:
            # Create in-memory cache as default
            # For production, pass Redis-backed cache explicitly
            self._cache = EmbeddingCache(
                redis_client=None,  # In-memory only
                max_memory_size=10000,
                prefix="emb_cache"
            )
            logger.info("Using in-memory embedding cache (no Redis)")

        # Initialize OpenAI client
        self._client = OpenAIEmbeddingClient()

        logger.info(
            f"Initialized EmbeddingGenerator: model={self.model.model_id}, "
            f"cache={'enabled' if enable_cache else 'disabled'}, "
            f"batch_size={self.batch_size}"
        )

    @property
    def model_id(self) -> str:
        """Get current model identifier."""
        return self.model.model_id

    def _compute_content_hash(self, text: str) -> str:
        """Compute hash of text content for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key combining content hash and model ID.

        CRITICAL: Model ID is included to prevent mixing embeddings
        from different models.

        Note: Unified cache handles key prefixing automatically.
        """
        content_hash = self._compute_content_hash(text)
        # Include model ID to ensure cache invalidation on model change
        model_hash = hashlib.sha256(self.model_id.encode()).hexdigest()[:8]
        return f"{model_hash}:{content_hash}"

    async def embed_texts(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        FOCUS: Batch embeddings (100-500 per call)

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache for this request

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_id=self.model_id,
                dimensions=self.model.dimensions,
            )

        # Try to get from unified cache
        texts_to_embed: list[tuple[int, str]] = []  # (index, text)
        cached_count = 0
        final_embeddings: list[Optional[list[float]]] = [None] * len(texts)

        if use_cache and self._cache:
            # Use unified cache batch API
            cached_embeddings, missing_indices = self._cache.get_batch(
                contents=texts,
                model_id=self.model_id
            )

            # Fill in cached embeddings
            for i, embedding in enumerate(cached_embeddings):
                if embedding is not None:
                    final_embeddings[i] = embedding
                    cached_count += 1

            # Track texts that need embedding
            texts_to_embed = [(i, texts[i]) for i in missing_indices]
        else:
            # No cache, embed all
            texts_to_embed = list(enumerate(texts))

        # Generate embeddings for uncached texts
        new_embeddings: dict[int, list[float]] = {}

        if texts_to_embed:
            # Process in batches
            new_embeddings = await self._embed_in_batches(texts_to_embed)

            # Cache new embeddings using unified cache
            if use_cache and self._cache:
                new_contents = [texts[idx] for idx in new_embeddings.keys()]
                new_embedding_list = list(new_embeddings.values())
                self._cache.set_batch(
                    contents=new_contents,
                    model_id=self.model_id,
                    embeddings=new_embedding_list
                )

            # Fill in new embeddings
            for idx, embedding in new_embeddings.items():
                final_embeddings[idx] = embedding

        latency_ms = (time.time() - start_time) * 1000

        # Estimate cost
        total_tokens = sum(len(text.split()) for text in texts)  # Approximate
        estimated_cost = (total_tokens / 1_000_000) * self.model.cost_per_million_tokens

        return EmbeddingResult(
            embeddings=final_embeddings,  # type: ignore
            model_id=self.model_id,
            dimensions=self.model.dimensions,
            texts_count=len(texts),
            total_tokens=total_tokens,
            cached_count=cached_count,
            generated_count=len(new_embeddings),
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
        )

    async def _embed_in_batches(
        self,
        texts_with_indices: list[tuple[int, str]],
    ) -> dict[int, list[float]]:
        """
        Embed texts in batches with retries.

        FOCUS: Batch embeddings (100-500 per call)
        """
        results: dict[int, list[float]] = {}
        # Sort by index for consistent batching
        texts_with_indices = sorted(texts_with_indices, key=lambda x: x[0])

        # Process batches
        for batch_start in range(0, len(texts_with_indices), self.batch_size):
            batch = texts_with_indices[batch_start:batch_start + self.batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_texts = [text for _, text in batch]

            # Embed with retries
            embeddings = await self._embed_with_retry(batch_texts)

            # Map back to original indices
            for idx, embedding in zip(batch_indices, embeddings):
                results[idx] = embedding

        return results

    async def _embed_with_retry(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed texts with exponential backoff retry."""
        last_error = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                return await self._client.embed(texts, self.model)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        raise RuntimeError(
            f"Failed to generate embeddings after {self.max_retries} attempts: {last_error}"
        )

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query text.

        Convenience method for search queries.
        """
        result = await self.embed_texts([query], use_cache=False)
        return result.embeddings[0]

    async def embed_documents(
        self,
        documents: list[str],
        document_ids: Optional[list[str]] = None,
    ) -> EmbeddingResult:
        """
        Embed documents with caching.

        MUST: Cache embeddings for unchanged docs
        """
        return await self.embed_texts(documents, use_cache=True)

    def validate_model_consistency(self, other_model_id: str) -> bool:
        """
        CRITICAL: Validate that models match.

        Index and query embeddings MUST use the same model.
        """
        is_consistent = self.model_id == other_model_id

        if not is_consistent:
            logger.error(
                f"Model mismatch! Current: {self.model_id}, "
                f"Other: {other_model_id}. This will produce garbage results."
            )

        return is_consistent


def create_embedding_generator(
    model_key: Optional[str] = None,
    **kwargs,
) -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator.

    Args:
        model_key: Model identifier (defaults to config settings)
        **kwargs: Additional arguments for EmbeddingGenerator

    Returns:
        Configured EmbeddingGenerator instance

    Example:
        >>> generator = create_embedding_generator()  # Uses config
        >>> result = await generator.embed_texts(["Hello world"])
    """
    return EmbeddingGenerator(model_key=model_key, **kwargs)
