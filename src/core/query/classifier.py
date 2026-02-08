"""
Rule-Based Query Classification for RAG Pipeline.

Not every query should go to RAG:
- RETRIEVAL: Factual questions answerable from your docs → use RAG
- GENERATION: Creative, summarization, analysis tasks → LLM only
- CLARIFICATION: Ambiguous → ask follow-up question
- REJECTION: Out of scope → decline politely

Cost: $0 | Latency: <1ms
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    """Query routing categories."""
    RETRIEVAL = "retrieval"        # Factual questions → use RAG
    GENERATION = "generation"      # Creative/analysis → LLM only
    CLARIFICATION = "clarification"  # Ambiguous → ask follow-up
    REJECTION = "rejection"        # Out of scope → decline


@dataclass
class ClassificationResult:
    """Result of query classification."""
    query: str
    route: QueryRoute
    reason: str = ""
    follow_up_question: Optional[str] = None

    @property
    def needs_rag(self) -> bool:
        """Whether this query should go through RAG pipeline."""
        return self.route == QueryRoute.RETRIEVAL

    @property
    def needs_clarification(self) -> bool:
        """Whether we should ask a follow-up question."""
        return self.route == QueryRoute.CLARIFICATION


class QueryClassifier:
    """
    Rule-based query classifier.

    Fast classification using patterns - no LLM calls.
    Cost: $0 | Latency: <1ms

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("What is the return policy?")

        if result.needs_rag:
            # Run RAG pipeline
        elif result.needs_clarification:
            # Ask follow-up
    """

    # Patterns for GENERATION (creative/analysis - no RAG needed)
    GENERATION_PATTERNS = [
        r"^(write|create|generate|compose|draft)\s",
        r"^(summarize|summarise)\s",
        r"^(translate|convert)\s",
        r"^(explain|describe)\s+(to me|like|as if)",
        r"(in your (own )?words|in simple terms)$",
        r"^(what do you think|give me your opinion)",
    ]

    # Patterns for REJECTION (out of scope)
    REJECTION_KEYWORDS = [
        # Harmful
        "hack", "crack", "exploit", "attack", "malware", "virus",
        "steal", "illegal", "bypass security",
        # Off-topic (customize based on your domain)
        "weather", "stock price", "sports score", "celebrity",
        "recipe", "joke", "sing", "play game",
    ]

    # Clarification triggers
    MIN_QUERY_WORDS = 2
    VAGUE_QUERIES = [
        "help", "help me", "i need help",
        "question", "i have a question",
        "hi", "hello", "hey",
        "yes", "no", "ok", "okay",
        "what", "how", "why",  # Single word questions
        "tell me", "show me", "explain",
    ]

    # Arabic vague patterns
    ARABIC_VAGUE = [
        "مساعدة", "سؤال", "مرحبا", "اهلا",
        "نعم", "لا", "ماذا", "كيف", "لماذا",
    ]

    def __init__(
        self,
        rejection_keywords: Optional[list[str]] = None,
        generation_patterns: Optional[list[str]] = None,
        min_query_words: int = 2,
    ):
        """
        Args:
            rejection_keywords: Additional keywords to reject
            generation_patterns: Additional patterns for generation route
            min_query_words: Minimum words for a valid query
        """
        self.rejection_keywords = self.REJECTION_KEYWORDS.copy()
        if rejection_keywords:
            self.rejection_keywords.extend(rejection_keywords)

        self.generation_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.GENERATION_PATTERNS
        ]
        if generation_patterns:
            self.generation_patterns.extend([
                re.compile(p, re.IGNORECASE) for p in generation_patterns
            ])

        self.min_query_words = min_query_words

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify query using rules.

        Args:
            query: User's query

        Returns:
            ClassificationResult with route
        """
        query = query.strip()
        query_lower = query.lower()
        words = query.split()

        # Empty query
        if not query:
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="empty query",
                follow_up_question="How can I help you today?",
            )

        # Check REJECTION first (harmful/off-topic)
        for keyword in self.rejection_keywords:
            if keyword in query_lower:
                return ClassificationResult(
                    query=query,
                    route=QueryRoute.REJECTION,
                    reason=f"contains out-of-scope keyword: {keyword}",
                )

        # Check CLARIFICATION (too vague)
        if len(words) < self.min_query_words:
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="query too short",
                follow_up_question="Could you please provide more details about what you're looking for?",
            )

        if query_lower in self.VAGUE_QUERIES or query in self.ARABIC_VAGUE:
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="vague query",
                follow_up_question="Could you please be more specific about what you need help with?",
            )

        # Check GENERATION (creative tasks - no RAG needed)
        for pattern in self.generation_patterns:
            if pattern.search(query_lower):
                return ClassificationResult(
                    query=query,
                    route=QueryRoute.GENERATION,
                    reason="creative/generation task",
                )

        # Default: RETRIEVAL (use RAG)
        return ClassificationResult(
            query=query,
            route=QueryRoute.RETRIEVAL,
            reason="factual query - use RAG",
        )


def create_classifier(
    rejection_keywords: Optional[list[str]] = None,
    min_query_words: int = 2,
) -> QueryClassifier:
    """
    Factory function to create query classifier.

    Args:
        rejection_keywords: Additional keywords to reject
        min_query_words: Minimum words for valid query

    Example:
        classifier = create_classifier(
            rejection_keywords=["competitor_name", "unrelated_topic"]
        )
        result = classifier.classify("What's the return policy?")

        if result.needs_rag:
            context = await retriever.search(query)
            response = await llm.generate(query, context)
        elif result.needs_clarification:
            return result.follow_up_question
        elif result.route == QueryRoute.REJECTION:
            return "I can only help with questions about our products."
        else:  # GENERATION
            response = await llm.generate(query)  # No RAG needed
    """
    return QueryClassifier(
        rejection_keywords=rejection_keywords,
        min_query_words=min_query_words,
    )
