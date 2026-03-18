"""
Router node: analyzes user_input and sets the task for conditional routing.
Task is one of: qa | summarization | sentiment | history | unknown.
"""

from src.state import GraphState


def router_node(state: GraphState) -> dict:
    """
    Inspect user input and set the task. Does not call any model.
    Returns a state update with the chosen task.
    """
    raw = (state.get("user_input") or "").strip().lower()
    task = "unknown"

    if raw.startswith("summary") or raw == "summary":
        task = "summarization"
    elif raw.startswith("ask "):
        task = "qa"
    elif raw.startswith("sentiment "):
        task = "sentiment"
    elif raw == "sentiment":
        # Sentiment with no extra text: will use context (document) in sentiment_node
        task = "sentiment"
    elif raw == "history":
        task = "history"
    else:
        # Treat as a question if it looks like one (e.g. free-form question when document is set)
        if raw and not any(
            raw.startswith(cmd)
            for cmd in ("document", "clear", "quit", "exit", "q")
        ):
            task = "qa"

    return {"task": task}
