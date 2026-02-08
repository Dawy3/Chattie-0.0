"""
Semantic Cache for RAG Pipeline.

2-Layer Cache:
- Layer 1: Exact match (hash lookup - fast, simple)
- Layer 2: Semantic similarity (embed query, search cache, threshold > 0.9)
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached query-response pair."""
    query: str
    response: str
    embedding: np.ndarray
    created_at: float


@dataclass
class CacheResult:
    """Result from cache lookup."""
    hit: bool
    response: Optional[str] = None
    layer: Optional[str] = None  # "exact" or "semantic"
    similarity: float = 0.0


class SemanticCache:
    """
    2-Layer semantic cache.

    Layer 1: Exact match - hash lookup (fast, simple)
    Layer 2: Semantic similarity - embed query, search cache, threshold > 0.9

    Usage:
        cache = SemanticCache(embed_func)

        result = cache.get(query)
        if result.hit:
            return result.response

        response = generate(query)
        cache.set(query, response)
    """

    def __init__(
        self,
        embed_func: Callable[[str], list[float]],
        similarity_threshold: float = 0.9,
        max_size: int = 10000,
    ):
        """
        Args:
            embed_func: Function to embed queries
            similarity_threshold: Min similarity for Layer 2 (default: 0.9)
            max_size: Max cache entries
        """
        self.embed_func = embed_func
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size

        self._cache: dict[str, CacheEntry] = {}
        self._stats = {"exact": 0, "semantic": 0, "miss": 0}

    def get(self, query: str) -> CacheResult:
        """
        Look up query in cache.

        Layer 1: Exact match (hash lookup)
        Layer 2: Semantic similarity (if Layer 1 misses)
        """
        # Layer 1: Exact match
        query_hash = self._hash(query)
        if query_hash in self._cache:
            self._stats["exact"] += 1
            return CacheResult(
                hit=True,
                response=self._cache[query_hash].response,
                layer="exact",
                similarity=1.0,
            )

        # Layer 2: Semantic similarity
        query_embedding = np.array(self.embed_func(query))
        match = self._semantic_search(query_embedding)

        if match:
            self._stats["semantic"] += 1
            return CacheResult(
                hit=True,
                response=match["response"],
                layer="semantic",
                similarity=match["similarity"],
            )

        self._stats["miss"] += 1
        return CacheResult(hit=False)

    def set(self, query: str, response: str) -> None:
        """Cache query-response pair."""
        if not response:
            return

        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        query_hash = self._hash(query)
        embedding = np.array(self.embed_func(query))

        self._cache[query_hash] = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            created_at=time.time(),
        )

    def _hash(self, query: str) -> str:
        """Hash for exact match."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    def _semantic_search(self, query_embedding: np.ndarray) -> Optional[dict]:
        """Search cache by semantic similarity."""
        best_sim = 0.0
        best_entry = None

        for entry in self._cache.values():
            sim = self._cosine_sim(query_embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        # Check threshold
        if best_sim >= self.similarity_threshold and best_entry:
            return {"response": best_entry.response, "similarity": best_sim}

        return None

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity."""
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _evict_oldest(self) -> None:
        """Remove oldest entry."""
        oldest = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[oldest]

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._stats = {"exact": 0, "semantic": 0, "miss": 0}

    def stats(self) -> dict:
        """Cache statistics."""
        total = sum(self._stats.values())
        hits = self._stats["exact"] + self._stats["semantic"]
        return {
            **self._stats,
            "hit_rate": hits / total if total > 0 else 0,
            "size": len(self._cache),
        }
