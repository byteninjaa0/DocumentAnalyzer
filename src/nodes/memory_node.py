"""
Memory node: appends the current turn (user_input, response) to chat_history.
Chat history is stored in GraphState and reduced with operator.add (append).
"""

from src.state import GraphState


def memory_node(state: GraphState) -> dict:
    """
    Append (user_input, response) to chat_history. Returns the single new pair
    so the state reducer (add) appends it to the list.
    """
    user_input = state.get("user_input") or ""
    response = state.get("response") or ""
    return {"chat_history": [(user_input, response)]}
