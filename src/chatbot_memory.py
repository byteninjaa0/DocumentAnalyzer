"""
Conversation memory for the chatbot using LangChain's ConversationBufferMemory.
Enables the assistant to remember previous interactions in the session.
"""

from typing import List, Optional

# LangChain memory
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.runnables.history import RunnableWithMessageHistory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class ConversationBufferMemory:
    """
    Simple conversation buffer that stores (human, assistant) message pairs.
    Used when LangChain's full memory is not required, or as a fallback.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._history: List[tuple] = []  # [(human, assistant), ...]

    def add_user_message(self, content: str) -> None:
        self._history.append((content, None))

    def add_ai_message(self, content: str) -> None:
        if self._history and self._history[-1][1] is None:
            self._history[-1] = (self._history[-1][0], content)
        else:
            self._history.append((None, content))

    def get_history_as_string(self, last_n: int = 10) -> str:
        """Format recent history as a string for context."""
        lines = []
        for h, a in self._history[-last_n:]:
            if h:
                lines.append(f"User: {h}")
            if a:
                lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def get_full_history(self) -> List[tuple]:
        return self._history.copy()

    def clear(self) -> None:
        self._history.clear()


def get_langchain_memory():
    """
    Return a LangChain ConversationBufferMemory if available.
    Otherwise return None (caller can use our simple ConversationBufferMemory).
    """
    if not LANGCHAIN_AVAILABLE:
        return None
    try:
        from langchain.memory import ConversationBufferMemory as LCConversationBufferMemory
        return LCConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    except Exception:
        return None
