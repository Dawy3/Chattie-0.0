"""Direct search endpoints (hybrid and vector-only)."""

import logging

from fastapi import APIRouter, Depends

from src.core.embedding.generator import EmbeddingGenerator
from src.services.vector_store.qdrant import QdrantStore
from src.core.retrieval.hybrid_search import HybridSearch

from api.v1.schemas import SearchRequest, SearchResultItem, SearchResponse
from api.v1.dependencies import (
    get_embedding_generator,
    get_qdrant_store,
    get_hybrid_search,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(
    request: SearchRequest,
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
    hybrid: HybridSearch = Depends(get_hybrid_search),
):
    """Hybrid search: embed query, then RRF fusion of vector + BM25."""
    query_embedding = await embedding_gen.embed_query(request.query)

    results = hybrid.search(
        query=request.query,
        query_embedding=query_embedding,
        top_k=request.top_k,
        filter_dict=request.filter,
    )

    items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            score=r.score,
            content=r.content,
            document_id=r.document_id,
            metadata=r.metadata,
        )
        for r in results
    ]

    return SearchResponse(results=items, total=len(items), query=request.query)


@router.post("/search/vector", response_model=SearchResponse)
async def vector_search(
    request: SearchRequest,
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
):
    """Vector-only search via QdrantStore."""
    query_embedding = await embedding_gen.embed_query(request.query)

    results = await qdrant_store.search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        filter=request.filter,
    )

    items = [
        SearchResultItem(
            chunk_id=r.get("id", ""),
            score=r.get("score", 0.0),
            content=r.get("metadata", {}).get("content", ""),
            document_id=r.get("metadata", {}).get("document_id", ""),
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]

    return SearchResponse(results=items, total=len(items), query=request.query)
