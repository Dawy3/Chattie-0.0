"""
Prompt Manager for RAG Pipeline.

Simple prompt building for RAG queries.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant. Answer based on the provided context. If the context doesn't contain the answer, say so."""

RAG_TEMPLATE = """Context:
{context}

Question: {query}

Answer based on the context above. Be concise and accurate."""

RAG_WITH_HISTORY_TEMPLATE = """Context:
{context}

Previous conversation:
{history}

Question: {query}

Answer based on context. Be concise and accurate."""

NO_CONTEXT_TEMPLATE = """No relevant information found for this question.

Question: {query}

Explain that you don't have information on this topic."""


class PromptManager:
    """
    Simple prompt manager for RAG.

    Usage:
        pm = PromptManager()
        system, user = pm.build(query, context)
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Args:
            system_prompt: Custom system prompt (optional)
        """
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def build(
        self,
        query: str,
        context: str,
        history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        """
        Build RAG prompt.

        Args:
            query: User question
            context: Retrieved context string
            history: Optional conversation history [{role, content}, ...]

        Returns:
            (system_prompt, user_prompt)
        """
        if not context:
            user_prompt = NO_CONTEXT_TEMPLATE.format(query=query)
            return self.system_prompt, user_prompt

        if history:
            history_str = self._format_history(history)
            user_prompt = RAG_WITH_HISTORY_TEMPLATE.format(
                context=context,
                history=history_str,
                query=query,
            )
        else:
            user_prompt = RAG_TEMPLATE.format(context=context, query=query)

        return self.system_prompt, user_prompt

    def _format_history(self, history: list[dict]) -> str:
        """Format conversation history from ConversationMemory.get().

        Memory already handles windowing (last 3 full) and summarization
        (older messages capped at ~150 tokens). Just format as text here.
        """
        lines = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
