"""API v1 router aggregation."""

from fastapi import APIRouter

from api.v1.health import router as health_router
from api.v1.documents import router as documents_router
from api.v1.query import router as query_router
from api.v1.search import router as search_router

v1_router = APIRouter()

v1_router.include_router(health_router)
v1_router.include_router(documents_router, prefix="/documents")
v1_router.include_router(query_router, prefix="/query")
v1_router.include_router(search_router, prefix="/search")
