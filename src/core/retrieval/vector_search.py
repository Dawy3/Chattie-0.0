"""
Vector (Semantic) Search for RAG Pipeline.

Uses Qdrant with HNSW index for fast approximate nearest neighbor search.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result."""
    chunk_id: str
    score: float
    content: str = ""
    metadata: dict = field(default_factory=dict)
    document_id: str = ""


class VectorSearch:
    """
    Vector search using Qdrant.

    Usage:
        search = VectorSearch(collection_name="documents")
        await search.index(chunks, embeddings)
        results = await search.search(query_embedding, top_k=10)
    """

    def __init__(
        self,
        collection_name: str = "documents",
        dimensions: int = 1536,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            collection_name: Qdrant collection name
            dimensions: Embedding dimensions
            host: Qdrant host (for local)
            port: Qdrant port (for local)
            url: Qdrant Cloud URL (if using cloud)
            api_key: Qdrant API key (if using cloud)
        """
        self.collection_name = collection_name
        self.dimensions = dimensions

        # Connect to Qdrant
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(host=host, port=port)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")

    def index(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """
        Index chunks with embeddings.

        Args:
            chunks: List of {"chunk_id", "content", "document_id", "metadata"}
            embeddings: Corresponding embeddings

        Returns:
            Number of chunks indexed
        """
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "content": chunk.get("content", ""),
                "document_id": chunk.get("document_id", ""),
                **chunk.get("metadata", {}),
            }
            points.append(PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload=payload,
            ))

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Indexed {len(points)} chunks")
        return len(points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_dict: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Optional metadata filter {"field": "value"}

        Returns:
            List of SearchResult
        """
        # Build filter if provided
        qdrant_filter = None
        if filter_dict:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            qdrant_filter = Filter(must=conditions)

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            payload = hit.payload or {}
            results.append(SearchResult(
                chunk_id=str(hit.id),
                score=hit.score,
                content=payload.get("content", ""),
                metadata=payload,
                document_id=payload.get("document_id", ""),
            ))

        return results

    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=chunk_ids,
        )
        return len(chunk_ids)
