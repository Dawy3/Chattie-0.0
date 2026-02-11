"""Prompt building for RAG queries."""


SYSTEM_PROMPT = """You are a helpful AI assistant. Answer in the same language the user writes in. Be direct and concise."""


def build_prompt(query: str, context: str, history: list[dict] | None = None) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the LLM."""

    if not context:
        return SYSTEM_PROMPT, f"{query}\n\n(No documents found — use your general knowledge.)"

    parts = [f"Context:\n{context}"]

    if history:
        lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history]
        parts.append(f"Conversation:\n" + "\n".join(lines))

    parts.append(f"Question:\n{query}")

    user_prompt = "\n\n".join(parts)
    return SYSTEM_PROMPT, user_prompt
