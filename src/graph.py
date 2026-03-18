"""
LangGraph workflow: stateful assistant with router and task nodes.
Flow: START → router → (qa | summarize | sentiment | history) → memory → END.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.state import GraphState
from src.nodes import (
    router_node,
    qa_node,
    summarize_node,
    sentiment_node,
    memory_node,
    history_node,
)
from src.nodes.qa_node import set_qa_chain_getter
from src.nodes.summarize_node import set_summarize_chain_getter
from src.nodes.sentiment_node import set_sentiment_model


def _route_after_router(state: GraphState) -> Literal["qa", "summarize", "sentiment", "history"]:
    """Conditional edge: route to the node indicated by state['task']."""
    task = (state.get("task") or "unknown").strip().lower()
    if task == "summarization":
        return "summarize"
    if task == "sentiment":
        return "sentiment"
    if task == "history":
        return "history"
    # "qa" and "unknown" (treat unknown as QA attempt)
    return "qa"


def build_graph(
    get_qa_chain=None,
    get_summary_chain=None,
    vectorizer=None,
    classifier=None,
):
    """
    Build and compile the assistant graph. Injects dependencies into nodes.
    Call with get_qa_chain, get_summary_chain (callables), and vectorizer/classifier for sentiment.
    """
    # Inject dependencies into nodes (they use module-level getters)
    if get_qa_chain is not None:
        set_qa_chain_getter(get_qa_chain)
    if get_summary_chain is not None:
        set_summarize_chain_getter(get_summary_chain)
    if vectorizer is not None and classifier is not None:
        set_sentiment_model(vectorizer, classifier)

    builder = StateGraph(GraphState)

    # Add all nodes
    builder.add_node("router", router_node)
    builder.add_node("qa", qa_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("history", history_node)
    builder.add_node("memory", memory_node)

    # Entry: START → router
    builder.add_edge(START, "router")

    # Conditional: router → task node based on task
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        path_map={
            "qa": "qa",
            "summarize": "summarize",
            "sentiment": "sentiment",
            "history": "history",
        },
    )

    # Task nodes → memory
    builder.add_edge("qa", "memory")
    builder.add_edge("summarize", "memory")
    builder.add_edge("sentiment", "memory")
    builder.add_edge("history", "memory")

    # memory → END
    builder.add_edge("memory", END)

    return builder.compile()
