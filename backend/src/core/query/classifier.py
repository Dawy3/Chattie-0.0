"""
Query Classifier for Multilingual RAG Pipeline (Arabic + English).

Purpose:
- Decide whether a query needs RAG, LLM-only, clarification, or rejection.
- Rule-based for zero cost and ultra-low latency.

Cost: $0 | Latency: <1ms
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    RETRIEVAL = "retrieval"       # Needs RAG
    GENERATION = "generation"     # LLM-only (creative, chat, explanation)
    CLARIFICATION = "clarification"
    REJECTION = "rejection"       # Explicitly unsupported or unsafe


@dataclass
class ClassificationResult:
    query: str
    route: QueryRoute
    reason: str = ""
    follow_up_question: Optional[str] = None
    detected_language: Optional[str] = None  # "ar" | "en"

    @property
    def needs_rag(self) -> bool:
        return self.route == QueryRoute.RETRIEVAL

    @property
    def needs_clarification(self) -> bool:
        return self.route == QueryRoute.CLARIFICATION


class QueryClassifier:
    """
    Rule-based multilingual query classifier.

    Optimized for:
    - RAG cost control
    - Predictable routing
    - Arabic & English support
    """

    # -------------------------
    # Language detection
    # -------------------------
    ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")

    # -------------------------
    # Greetings / small talk
    # -------------------------
    GREETING_PATTERNS = [
        r"^(مرحبا|مرحباً|اهلا|أهلا|السلام عليكم|صباح الخير|مساء الخير).*$",
        r"^(hi|hello|hey|good morning|good evening|how are you).*$",
    ]

    # -------------------------
    # Generation (no RAG)
    # -------------------------
    GENERATION_PATTERNS = [
        # Arabic
        r"^(اكتب|أنشئ|لخص|اختصر|ترجم|اشرح|صف|حلل)\s",
        r"(بكلماتك|بأسلوبك|ببساطة)$",
        r"^(ما رأيك|اعطني رأيك)",
        # English
        r"^(write|summarize|translate|explain|describe|analyze)\s",
        r"(in your own words|simply)$",
        r"^(what do you think)",
    ]

    # -------------------------
    # Explicitly unsafe / disallowed
    # -------------------------
    HARD_REJECTION_KEYWORDS = [
        # Arabic
        "اختراق", "تهكير", "تجسس", "سرقة",
        # English
        "hack", "crack", "exploit", "malware",
    ]

    # -------------------------
    # Vague / low-signal queries
    # -------------------------
    VAGUE_PATTERNS = [
        r"^(مساعدة|ساعدني|سؤال|عندي سؤال)$",
        r"^(help|question)$",
        r"^(نعم|لا|اوكي|تمام|ok|yes|no)$",
        r"^(كيف|لماذا|ماذا)$",
    ]

    MIN_QUERY_WORDS = 2

    def __init__(self, min_query_words: int = 2):
        self.min_query_words = min_query_words

        self.greeting_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS
        ]
        self.generation_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.GENERATION_PATTERNS
        ]
        self.vague_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.VAGUE_PATTERNS
        ]

    # -------------------------
    # Helpers
    # -------------------------
    def _detect_language(self, query: str) -> str:
        return "ar" if self.ARABIC_CHAR_PATTERN.search(query) else "en"

    def _matches_any(self, patterns, text: str) -> bool:
        return any(p.search(text) for p in patterns)

    # -------------------------
    # Main classification
    # -------------------------
    def classify(self, query: str) -> ClassificationResult:
        query = query.strip()
        query_lower = query.lower()
        language = self._detect_language(query)

        # Empty query
        if not query:
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="Empty query",
                follow_up_question="كيف يمكنني مساعدتك؟" if language == "ar" else "How can I help you?",
                detected_language=language,
            )

        # Hard rejection (unsafe)
        for keyword in self.HARD_REJECTION_KEYWORDS:
            if keyword in query_lower:
                return ClassificationResult(
                    query=query,
                    route=QueryRoute.REJECTION,
                    reason=f"Unsafe keyword detected: {keyword}",
                    detected_language=language,
                )

        # Greetings / chit-chat
        if self._matches_any(self.greeting_patterns, query):
            return ClassificationResult(
                query=query,
                route=QueryRoute.GENERATION,
                reason="Greeting or small talk",
                detected_language=language,
            )

        # Too short / vague
        if len(query.split()) < self.min_query_words or self._matches_any(self.vague_patterns, query):
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="Low information query",
                follow_up_question=(
                    "هل يمكنك توضيح سؤالك أكثر؟"
                    if language == "ar"
                    else "Could you please clarify your question?"
                ),
                detected_language=language,
            )

        # Creative / generative
        if self._matches_any(self.generation_patterns, query):
            return ClassificationResult(
                query=query,
                route=QueryRoute.GENERATION,
                reason="Creative or explanatory task",
                detected_language=language,
            )

        # Default → Retrieval
        return ClassificationResult(
            query=query,
            route=QueryRoute.RETRIEVAL,
            reason="Factual or knowledge-based query",
            detected_language=language,
        )