"""
Context Builder for RAG Pipeline.

Builds context from retrieved chunks within token limits.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """Built context with metadata."""
    text: str
    chunks_used: int
    estimated_tokens: int
    sources: list[dict]


class ContextBuilder:
    """
    Builds context from retrieved chunks.

    Usage:
        builder = ContextBuilder(max_tokens=2000)
        context = builder.build(chunks)
        # context.text -> formatted context string
        # context.sources -> source references
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        max_chunks: int = 10,
        chars_per_token: int = 4,
    ):
        """
        Args:
            max_tokens: Maximum tokens for context
            max_chunks: Maximum chunks to include
            chars_per_token: Approximate chars per token (for estimation)
        """
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.chars_per_token = chars_per_token

    def build(
        self,
        chunks: list[dict],
        include_sources: bool = True,
    ) -> Context:
        """
        Build context from chunks.

        Args:
            chunks: List of {content, document_id, chunk_id, score, ...}
            include_sources: Include source references in output

        Returns:
            Context with formatted text and metadata
        """
        if not chunks:
            return Context(text="", chunks_used=0, estimated_tokens=0, sources=[])

        max_chars = self.max_tokens * self.chars_per_token
        context_parts = []
        sources = []
        total_chars = 0

        for i, chunk in enumerate(chunks[:self.max_chunks]):
            content = chunk.get("content", "")
            chunk_chars = len(content)

            # Check if adding this chunk exceeds limit
            if total_chars + chunk_chars > max_chars:
                # Add truncated version if worth it
                remaining = max_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                    chunk_chars = len(content)
                else:
                    break

            # Add to context
            if include_sources:
                context_parts.append(f"[{i+1}] {content}")
            else:
                context_parts.append(content)

            total_chars += chunk_chars

            # Track source
            sources.append({
                "index": i + 1,
                "document_id": chunk.get("document_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("score", 0),
            })

        # Format final context
        text = "\n\n".join(context_parts)
        estimated_tokens = total_chars // self.chars_per_token

        logger.debug(f"Built context: {len(sources)} chunks, ~{estimated_tokens} tokens")

        return Context(
            text=text,
            chunks_used=len(sources),
            estimated_tokens=estimated_tokens,
            sources=sources,
        )


def create_context_builder(
    max_tokens: int = 2000,
    max_chunks: int = 10,
) -> ContextBuilder:
    """
    Create context builder.

    Example:
        builder = create_context_builder(max_tokens=2000)

        # After retrieval
        context = builder.build(retrieved_chunks)

        # Use in prompt
        prompt = f'''Answer based on this context:
        {context.text}

        Question: {query}'''
    """
    return ContextBuilder(max_tokens=max_tokens, max_chunks=max_chunks)
