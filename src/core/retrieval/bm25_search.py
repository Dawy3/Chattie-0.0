"""
BM25 Keyword Search for RAG Pipeline.

Exact term matching for IDs, codes, names, technical terms.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result."""
    chunk_id: str
    score: float
    content: str = ""
    metadata: dict = field(default_factory=dict)
    document_id: str = ""


class BM25Search:
    """
    BM25 keyword search.

    Usage:
        bm25 = BM25Search()
        bm25.index(chunks)
        results = bm25.search("error code ERR-404", top_k=10)
    """

    # Simple tokenizer pattern
    TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9\u0600-\u06FF]+(?:[-_][a-zA-Z0-9\u0600-\u06FF]+)*')

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._documents: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        if not text:
            return []
        return [t.lower() for t in self.TOKEN_PATTERN.findall(text)]

    def index(self, chunks: list[dict]) -> int:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of {"chunk_id", "content", "document_id", "metadata"}

        Returns:
            Number of chunks indexed
        """
        self._documents = chunks
        corpus = [self._tokenize(c.get("content", "")) for c in chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

        logger.info(f"Built BM25 index with {len(chunks)} chunks")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Search for matching chunks.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of SearchResult
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built")
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                doc = self._documents[idx]
                results.append(SearchResult(
                    chunk_id=doc["chunk_id"],
                    score=score,
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    document_id=doc.get("document_id", ""),
                ))

        return results

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)
