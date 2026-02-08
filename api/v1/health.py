"""Health check and stats endpoints."""

import logging

from fastapi import APIRouter, Depends

from src.config import settings
from src.services.vector_store.qdrant import QdrantStore
from src.core.retrieval.bm25_search import BM25Search
from src.core.memory.conversation import ConversationMemory

from api.v1.schemas import HealthResponse, StatsResponse
from api.v1.dependencies import (
    get_qdrant_store,
    get_bm25_search,
    get_conversation_memory,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.APP_ENV,
    )


@router.get("/health/stats", response_model=StatsResponse)
async def health_stats(
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    bm25_search: BM25Search = Depends(get_bm25_search),
    memory: ConversationMemory = Depends(get_conversation_memory),
):
    """Detailed system statistics."""
    qdrant_stats = qdrant_store.stats()
    return StatsResponse(
        qdrant_points=qdrant_stats.get("points_count", 0),
        qdrant_status=qdrant_stats.get("status", "unknown"),
        bm25_index_size=bm25_search.size,
        active_sessions=memory.get_session_count(),
    )
