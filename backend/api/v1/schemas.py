"""Pydantic v2 request/response models for API v1."""

from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Error
# =============================================================================

class ErrorResponse(BaseModel):
    detail: str
    status_code: int = 500


# =============================================================================
# Health
# =============================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    environment: str


class StatsResponse(BaseModel):
    qdrant_points: int
    qdrant_status: str
    bm25_index_size: int
    active_sessions: int


# =============================================================================
# Documents
# =============================================================================

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    strategy: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    message: str = "Document uploaded and indexed successfully"


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks_count: int


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    chunks: list[DocumentChunk]


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    message: str = "Document deleted successfully"


# =============================================================================
# Query
# =============================================================================

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=50)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    use_history: bool = True
    filter: Optional[dict] = None


class QuerySource(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    route: str
    sources: list[QuerySource] = Field(default_factory=list)
    model: str = ""
    usage: dict = Field(default_factory=dict)
    latency_ms: float = 0.0
    session_id: str = ""


class ClassifyRequest(BaseModel):
    query: str


class ClassifyResponse(BaseModel):
    query: str
    route: str
    reason: str
    follow_up_question: Optional[str] = None


# =============================================================================
# Search
# =============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3 , ge=1, le=50)
    filter: Optional[dict] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    score: float
    content: str
    document_id: str
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int
    query: str
