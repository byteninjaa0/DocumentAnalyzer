"""
Summarize node: generates a summary of the current document using HuggingFace (T5/BART).
Reuses the existing summarization chain.
"""

from src.state import GraphState

_get_summary_chain = None


def set_summarize_chain_getter(getter):
    global _get_summary_chain
    _get_summary_chain = getter


def summarize_node(state: GraphState) -> dict:
    """
    Summarize state["context"] and set state["response"].
    """
    context = state.get("context") or ""
    if not context:
        return {"response": "Paste a document first (command: document)."}

    if _get_summary_chain is None:
        return {"response": "Summarization model not loaded."}

    try:
        from src.summarization_chain import summarize_document
        chain = _get_summary_chain()
        summary = summarize_document(context, chain=chain)
        return {"response": summary.strip() if summary else "No summary generated."}
    except Exception as e:
        return {"response": f"Summary failed: {e}"}
