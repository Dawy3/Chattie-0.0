"""
Conversation History Manager.

Sliding window: last 3 messages full, older messages summarized.
Summary is truncated to ~150 tokens (600 chars) to control context size.
"""


class ConversationMemory:
    """
    In-memory conversation storage with summarization.

    - Last N messages: kept in full
    - Older messages: truncated into a short summary block
    - get() returns [summary_msg (if any), ...last_N_messages]

    Usage:
        memory = ConversationMemory(window_size=3, summary_max_chars=600)
        memory.add("s1", "user", "What is RAG?")
        memory.add("s1", "assistant", "RAG is...")
        history = memory.get("s1")
    """

    def __init__(self, window_size: int = 3, summary_max_chars: int = 600):
        """
        Args:
            window_size: Number of recent messages to keep in full
            summary_max_chars: Max chars for the older-messages summary (~150 tokens)
        """
        self.window_size = window_size
        self.summary_max_chars = summary_max_chars
        self._store: dict[str, list[dict]] = {}

    def add(self, session_id: str, role: str, content: str) -> None:
        """Add a message. Stores everything — trimming happens on get()."""
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({"role": role, "content": content})

    def get(self, session_id: str) -> list[dict]:
        """
        Get conversation history with sliding window + summary.

        Returns:
            If <= window_size messages: all messages as-is
            If > window_size: [summary_of_older, ...last_N_messages]
        """
        messages = self._store.get(session_id, [])
        if len(messages) <= self.window_size:
            return list(messages)

        older = messages[:-self.window_size]
        recent = messages[-self.window_size:]

        summary = self._summarize(older)
        return [{"role": "system", "content": summary}] + list(recent)

    def _summarize(self, messages: list[dict]) -> str:
        """Truncate older messages into a compact summary block."""
        lines = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            # Take first 80 chars of each old message
            short = content[:80] + "..." if len(content) > 80 else content
            lines.append(f"{role}: {short}")

        text = "Previous conversation summary:\n" + "\n".join(lines)
        if len(text) > self.summary_max_chars:
            text = text[:self.summary_max_chars] + "..."
        return text

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._store.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all conversations."""
        self._store.clear()

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._store)


conversation_memory = ConversationMemory(window_size=3, summary_max_chars=600)
