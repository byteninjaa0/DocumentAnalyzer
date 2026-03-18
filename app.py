"""
AI Knowledge Assistant - Web UI (Streamlit).
Run: streamlit run app.py

Uses LangGraph for QA, summarization, and sentiment. Session-based chat history.
"""

import sys
from pathlib import Path

# Ensure project root is on path (deployment-friendly: no hardcoded paths)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utils.file_utils import extract_text_from_upload

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Constants
DEFAULT_HF_MODEL = "google-t5/t5-small"
MODEL_DIR = PROJECT_ROOT / "models"


def _ensure_sentiment_model():
    """Load or train sentiment model; store in session state."""
    if "sentiment_vectorizer" in st.session_state and "sentiment_classifier" in st.session_state:
        return st.session_state["sentiment_vectorizer"], st.session_state["sentiment_classifier"]
    from src.sentiment_model import (
        train_sentiment_model,
        evaluate_model,
        save_model,
        load_model,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (MODEL_DIR / "sentiment_lr.pkl").exists():
        v, c = load_model(str(MODEL_DIR))
    else:
        with st.spinner("Training sentiment model (first run)..."):
            v, c, X_test, y_test, grid = train_sentiment_model(
                data_path=None, test_size=0.2, n_jobs=1
            )
            save_model(v, c, str(MODEL_DIR))
            metrics = evaluate_model(c, X_test, y_test)
            st.session_state["_sentiment_metrics"] = metrics
        v, c = load_model(str(MODEL_DIR))
    st.session_state["sentiment_vectorizer"] = v
    st.session_state["sentiment_classifier"] = c
    return v, c


def _get_qa_chain():
    """Lazy-load QA chain; cache in session state."""
    if "qa_chain" not in st.session_state:
        with st.spinner("Loading QA model..."):
            from src.qa_chain import create_qa_chain
            st.session_state["qa_chain"] = create_qa_chain(
                model_id=DEFAULT_HF_MODEL,
                max_new_tokens=100,
            )
    return st.session_state["qa_chain"]


def _get_summary_chain():
    """Lazy-load summarization chain; cache in session state."""
    if "summary_chain" not in st.session_state:
        with st.spinner("Loading summarization model..."):
            from src.summarization_chain import create_summarization_chain
            st.session_state["summary_chain"] = create_summarization_chain(
                model_id=DEFAULT_HF_MODEL,
                max_new_tokens=150,
            )
    return st.session_state["summary_chain"]


def _get_graph():
    """Build and cache compiled LangGraph in session state."""
    if "graph" in st.session_state:
        return st.session_state["graph"]
    vectorizer, classifier = _ensure_sentiment_model()
    from src.graph import build_graph
    graph = build_graph(
        get_qa_chain=_get_qa_chain,
        get_summary_chain=_get_summary_chain,
        vectorizer=vectorizer,
        classifier=classifier,
    )
    st.session_state["graph"] = graph
    return graph


def _run_graph(user_input: str, context: str, chat_history: list) -> tuple[str, list]:
    """Invoke LangGraph and return (response, updated chat_history)."""
    initial = {
        "user_input": user_input,
        "context": context,
        "chat_history": chat_history,
    }
    result = _get_graph().invoke(initial)
    response = result.get("response") or ""
    new_history = result.get("chat_history") or []
    return response, new_history


def _build_user_input(action: str, query: str) -> str:
    """Map UI action + query to the router-style user_input."""
    if action == "Ask a question about the document":
        return f"ask {query}" if query else "ask "
    if action == "Summarize document":
        return "summary"
    if action == "Sentiment analysis":
        return f"sentiment {query}" if query else "sentiment"
    if action == "Show chat history":
        return "history"
    return query or ""


# ----- Session state init -----
if "document_text" not in st.session_state:
    st.session_state["document_text"] = ""
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ----- UI -----
st.title("AI Knowledge Assistant")
st.caption("Summarize documents, ask questions, and analyze sentiment using local AI.")

# Sidebar: document upload
with st.sidebar:
    st.subheader("Document")
    uploaded = st.file_uploader(
        "Upload PDF or text file",
        type=["pdf", "txt", "md", "csv"],
        help="Provide a document to summarize, ask questions about, or analyze sentiment.",
    )
    if uploaded is not None:
        try:
            text = extract_text_from_upload(uploaded)
            st.session_state["document_text"] = text
            st.success(f"Loaded {len(text)} characters from {uploaded.name}")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    if st.session_state["document_text"]:
        st.info(f"Current document: **{len(st.session_state['document_text'])}** characters")
    else:
        st.warning("No document loaded. Upload a file to use summarize / QA.")

# Main: action + query
action = st.selectbox(
    "What do you want to do?",
    [
        "Ask a question about the document",
        "Summarize document",
        "Sentiment analysis",
        "Show chat history",
    ],
    index=0,
)

query_placeholder = "e.g. What is the main idea? (or leave blank for sentiment on document)"
if action == "Show chat history":
    query_placeholder = "Not needed for history"
query = st.text_input("Query (optional for summarize/history)", placeholder=query_placeholder)

col1, col2, _ = st.columns([1, 1, 3])
with col1:
    submit = st.button("Submit")
with col2:
    clear_chat = st.button("Clear chat")

if clear_chat:
    st.session_state["chat_history"] = []
    st.session_state.pop("last_response", None)
    st.session_state.pop("last_action", None)
    st.rerun()

if submit:
    user_input = _build_user_input(action, query or "")
    # Validation
    if not user_input.strip():
        st.warning("Please enter a query or choose an action.")
    elif action == "Ask a question about the document" and not (query or "").strip():
        st.warning("Please enter a question.")
    elif action in ("Summarize document", "Ask a question about the document") and not st.session_state["document_text"]:
        st.warning("Upload a document first (sidebar).")
    elif action == "Sentiment analysis" and not (query or "").strip() and not st.session_state["document_text"]:
        st.warning("Enter text to analyze or upload a document first.")
    else:
        with st.spinner("Processing..."):
            try:
                response, new_history = _run_graph(
                    user_input,
                    st.session_state["document_text"],
                    st.session_state["chat_history"],
                )
                st.session_state["chat_history"] = new_history
                st.session_state["last_response"] = response
                st.session_state["last_action"] = action
            except Exception as e:
                st.error(f"Error: {e}")
                response = str(e)
        st.rerun()

# Show last response after submit
if "last_response" in st.session_state:
    st.divider()
    st.subheader("Response")
    if st.session_state.get("last_action") == "Show chat history":
        st.text_area("Chat history", value=st.session_state["last_response"], height=200, disabled=True)
    else:
        st.write(st.session_state["last_response"])

# Chat history (recent turns)
if st.session_state["chat_history"]:
    st.divider()
    st.subheader("Recent conversation")
    for i, (human, assistant) in enumerate(st.session_state["chat_history"][-10:]):
        if human:
            st.markdown(f"**You:** {human[:200]}{'…' if len(human) > 200 else ''}")
        if assistant:
            st.markdown(f"**Assistant:** {assistant[:300]}{'…' if len(assistant) > 300 else ''}")
        st.markdown("---")
