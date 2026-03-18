"""
Central graph state for the LangGraph-based AI Knowledge Assistant.
All nodes read from and write to this shared state schema.
"""

from operator import add
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    """State schema for the assistant graph. All keys are optional for partial updates."""

    # User's raw input (command + payload, e.g. "ask What is X?" or "summary")
    user_input: str
    # Resolved task: "qa" | "summarization" | "sentiment" | "history" | "unknown"
    task: str
    # Current document context (for QA and summarization)
    context: str
    # Model/formatted response to show and to store in memory
    response: str
    # Chat history: list of (human_message, assistant_message) tuples; reducer appends
    chat_history: Annotated[list, add]
