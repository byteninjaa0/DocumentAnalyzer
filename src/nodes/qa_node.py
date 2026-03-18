"""
QA node: answers questions over the current document using the HuggingFace QA chain.
Reuses the existing LangChain-style QA pipeline (prompt + T5).
"""

from src.state import GraphState

# Will be injected when building the graph (closure)
_get_qa_chain = None


def set_qa_chain_getter(getter):
    global _get_qa_chain
    _get_qa_chain = getter


def qa_node(state: GraphState) -> dict:
    """
    Run question-answering over state["context"] for state["user_input"].
    Sets state["response"] with the model answer.
    """
    context = state.get("context") or ""
    user_input = (state.get("user_input") or "").strip()
    # Strip "ask " prefix if present so we pass only the question to the chain
    question = user_input[4:].strip() if user_input.startswith("ask ") else user_input
    if not question:
        return {"response": "No question provided."}
    if not context:
        return {"response": "Paste a document first (command: document)."}

    if _get_qa_chain is None:
        return {"response": "QA model not loaded."}

    try:
        from src.qa_chain import answer_question
        chain = _get_qa_chain()
        answer = answer_question(context, question, chain=chain)
        return {"response": answer.strip() if answer else "No answer generated."}
    except Exception as e:
        return {"response": f"QA failed: {e}"}
