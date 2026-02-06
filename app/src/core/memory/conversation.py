"""
Conversation History Manager (In-Memory).

Stores conversation history in RAM per session.
Applies sliding window (last N messages).

Note: History is lost on server restart.
For production with multiple servers, consider Redis or PostgreSQL.
"""

from typing import Optional


class ConversationMemory:
    """
    In-memory conversation storage.

    Stores history per session_id in RAM.

    Usage:
        memory = ConversationMemory(window_size=3)

        # Add messages
        memory.add("session_123", "user", "What is RAG?")
        memory.add("session_123", "assistant", "RAG is...")

        # Get history (with sliding window applied)
        history = memory.get("session_123")
    """

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Number of conversation turns to keep (default: 3)
        """
        self.window_size = window_size
        self._store: dict[str, list[dict]] = {}  # session_id -> messages

    def add(self, session_id: str, role: str, content: str) -> None:
        """
        Add message to conversation.

        Args:
            session_id: Unique session/conversation identifier
            role: "user" or "assistant"
            content: Message content
        """
        if session_id not in self._store:
            self._store[session_id] = []

        self._store[session_id].append({
            "role": role,
            "content": content,
        })

        # Apply sliding window - keep last N turns (N * 2 messages)
        max_messages = self.window_size * 2
        if len(self._store[session_id]) > max_messages:
            self._store[session_id] = self._store[session_id][-max_messages:]

    def get(self, session_id: str) -> list[dict]:
        """
        Get conversation history.

        Args:
            session_id: Session identifier

        Returns:
            List of messages [{"role": "user/assistant", "content": "..."}]
        """
        return self._store.get(session_id, [])

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._store.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all conversations."""
        self._store.clear()

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._store)


# Global instance - use this in your API
conversation_memory = ConversationMemory(window_size=3)
