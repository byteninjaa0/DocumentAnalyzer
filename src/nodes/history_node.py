"""
History node: formats chat_history for display and sets response (no model call).
"""

from src.state import GraphState


def history_node(state: GraphState) -> dict:
    """
    Format recent chat history as a string and set response. No LLM or ML call.
    """
    history = state.get("chat_history") or []
    last_n = 20
    lines = []
    for h, a in history[-last_n:]:
        if h:
            lines.append(f"User: {h}")
        if a:
            lines.append(f"Assistant: {a}")
    response = "\n".join(lines) if lines else "No history yet."
    return {"response": response}
