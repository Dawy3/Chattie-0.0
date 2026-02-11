"""
Prompt Manager for RAG Pipeline.

Production-grade prompt building for multilingual RAG queries.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a professional AI assistant integrated with a Retrieval-Augmented Generation (RAG) system.

Language behavior:
- Automatically detect the user's language.
- Respond in the SAME language as the user.
- Do NOT switch languages unless the user explicitly asks.
- Avoid generic apology or refusal phrases.

Knowledge behavior:
- Prefer answers grounded in the provided context.
- If the context is incomplete or missing:
  - Do NOT apologize.
  - Do NOT say you "don't have information".
  - Answer using general knowledge when it is safe.
  - Clearly indicate when the answer is based on general knowledge rather than retrieved context.

Response style:
- Be clear, concise, and helpful.
- Sound natural and confident.
- Avoid mentioning internal system limitations.
""".strip()


RAG_TEMPLATE = """
Context:
{context}

User question:
{query}

Instructions:
- Answer using ONLY the information from the context.
- If the context does not fully answer the question, say what is missing and provide a best-effort answer when possible.
- Do not apologize.
- Respond in the user's language.
""".strip()


RAG_WITH_HISTORY_TEMPLATE = """
Context:
{context}

Previous conversation:
{history}

User question:
{query}

Instructions:
- Use the context as the primary source.
- Use conversation history only for clarification.
- Do not repeat previous answers unnecessarily.
- If the context is insufficient, provide a best-effort answer and clearly state assumptions.
- Do not apologize.
- Respond in the user's language.
""".strip()


NO_CONTEXT_TEMPLATE = """
User question:
{query}

Instructions:
- No relevant documents were retrieved.
- Answer using general knowledge if possible.
- Clearly state that the answer is based on general knowledge, not retrieved documents.
- Ask one short follow-up question only if it helps clarify the request.
- Do not apologize.
- Respond in the user's language.
""".strip()


class PromptManager:
    """
    Prompt manager for RAG pipelines.

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
            user_prompt = RAG_TEMPLATE.format(
                context=context,
                query=query,
            )

        return self.system_prompt, user_prompt

    def _format_history(self, history: list[dict]) -> str:
        """Format conversation history into readable text."""
        lines = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)