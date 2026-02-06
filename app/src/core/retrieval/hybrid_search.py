"""
Hybrid Search for RAG Pipeline.

Combines semantic (vector) + keyword (BM25) using Reciprocal Rank Fusion (RRF).

Pipeline:
1. Retrieve top 100 from vector search
2. Retrieve top 100 from BM25 search
3. Fuse using RRF + recency boost
4. Return top 10
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .vector_search import VectorSearch, SearchResult as VectorResult
from .bm25_search import BM25Search, SearchResult as BM25Result

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Single hybrid search result."""
    chunk_id: str
    score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    recency_score: float = 0.0
    content: str = ""
    metadata: dict = field(default_factory=dict)
    document_id: str = ""
    from_vector: bool = False
    from_bm25: bool = False


class HybridSearch:
    """
    Hybrid search using RRF (Reciprocal Rank Fusion).

    RRF Score = 1/(k + vector_rank) + 1/(k + bm25_rank) + (recency * weight)

    Pipeline:
    - Get top 100 from vector search
    - Get top 100 from BM25 search
    - Fuse with RRF + recency
    - Return top 10

    Usage:
        hybrid = HybridSearch(vector_search, bm25_search)
        results = hybrid.search(query, query_embedding, top_k=10)
    """

    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
        rrf_k: int = 60,
        recency_weight: float = 0.2,
        recency_field: str = "created_at",
        recency_decay_days: float = 30.0,
        retrieval_k: int = 100,
    ):
        """
        Args:
            vector_search: Vector search instance
            bm25_search: BM25 search instance
            rrf_k: RRF constant (default: 60)
            recency_weight: Weight for recency boost (default: 0.2)
            recency_field: Metadata field for timestamp
            recency_decay_days: Half-life in days for recency decay
            retrieval_k: Candidates from each search (default: 100)
        """
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.rrf_k = rrf_k
        self.recency_weight = recency_weight
        self.recency_field = recency_field
        self.recency_decay_days = recency_decay_days
        self.retrieval_k = retrieval_k

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        filter_dict: Optional[dict] = None,
    ) -> list[HybridResult]:
        """
        Perform hybrid search with RRF.

        Args:
            query: Search query text
            query_embedding: Query embedding vector
            top_k: Number of final results (default: 10)
            filter_dict: Optional metadata filter

        Returns:
            List of HybridResult sorted by RRF score
        """
        # Step 1: Get top 100 from vector search
        vector_results = self.vector_search.search(
            query_embedding=query_embedding,
            top_k=self.retrieval_k,
            filter_dict=filter_dict,
        )

        # Step 2: Get top 100 from BM25 search
        bm25_results = self.bm25_search.search(
            query=query,
            top_k=self.retrieval_k,
        )

        # Step 3: RRF fusion
        fused = self._rrf_fusion(vector_results, bm25_results)

        # Step 4: Sort and return top_k
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]

    def _rrf_fusion(
        self,
        vector_results: list[VectorResult],
        bm25_results: list[BM25Result],
    ) -> list[HybridResult]:
        """
        Reciprocal Rank Fusion + Recency.

        RRF(d) = 1/(k + rank_vector) + 1/(k + rank_bm25) + (recency * weight)
        """
        # Build rank dictionaries (0-indexed)
        vector_ranks = {r.chunk_id: i for i, r in enumerate(vector_results)}
        bm25_ranks = {r.chunk_id: i for i, r in enumerate(bm25_results)}

        # Build content/metadata lookup
        content_lookup: dict[str, tuple] = {}
        for r in vector_results:
            content_lookup[r.chunk_id] = (r.content, r.metadata, r.document_id)
        for r in bm25_results:
            if r.chunk_id not in content_lookup:
                content_lookup[r.chunk_id] = (r.content, r.metadata, r.document_id)

        # Calculate RRF scores for all unique chunks
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        results = []

        for chunk_id in all_ids:
            rrf_score = 0.0
            v_rank = None
            b_rank = None

            # RRF from vector rank
            if chunk_id in vector_ranks:
                v_rank = vector_ranks[chunk_id]
                rrf_score += 1.0 / (self.rrf_k + v_rank + 1)

            # RRF from BM25 rank
            if chunk_id in bm25_ranks:
                b_rank = bm25_ranks[chunk_id]
                rrf_score += 1.0 / (self.rrf_k + b_rank + 1)

            content, metadata, doc_id = content_lookup[chunk_id]

            # Add recency boost
            recency_score = self._calculate_recency(metadata)
            final_score = rrf_score + (recency_score * self.recency_weight)

            results.append(HybridResult(
                chunk_id=chunk_id,
                score=final_score,
                vector_rank=v_rank,
                bm25_rank=b_rank,
                recency_score=recency_score,
                content=content,
                metadata=metadata,
                document_id=doc_id,
                from_vector=v_rank is not None,
                from_bm25=b_rank is not None,
            ))

        return results

    def _calculate_recency(self, metadata: dict) -> float:
        """
        Calculate recency score using exponential decay.

        score = exp(-days_old * decay_rate)
        Recent docs → closer to 1.0, older docs → closer to 0.0
        """
        timestamp = metadata.get(self.recency_field)
        if not timestamp:
            return 0.0

        try:
            if isinstance(timestamp, str):
                doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                doc_time = datetime.fromtimestamp(timestamp)
            else:
                return 0.0

            days_old = max(0, (datetime.now() - doc_time.replace(tzinfo=None)).days)

            # Exponential decay with half-life
            decay_rate = math.log(2) / self.recency_decay_days
            return math.exp(-decay_rate * days_old)

        except Exception:
            return 0.0


def create_hybrid_search(
    vector_search: VectorSearch,
    bm25_search: BM25Search,
    rrf_k: int = 60,
    recency_weight: float = 0.2,
) -> HybridSearch:
    """
    Create hybrid search with RRF fusion.

    Pipeline:
    - Retrieve 100 from vector search
    - Retrieve 100 from BM25 search
    - Fuse with RRF + recency boost
    - Return top 10

    Example:
        vector = VectorSearch(collection_name="docs")
        bm25 = BM25Search()

        vector.index(chunks, embeddings)
        bm25.index(chunks)

        hybrid = create_hybrid_search(vector, bm25)
        results = hybrid.search(query, embedding, top_k=10)
    """
    return HybridSearch(
        vector_search=vector_search,
        bm25_search=bm25_search,
        rrf_k=rrf_k,
        recency_weight=recency_weight,
        retrieval_k=100,
    )
