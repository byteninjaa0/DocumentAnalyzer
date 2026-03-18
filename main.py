"""
AI Knowledge Assistant - CLI Chatbot (LangGraph).
Run from project root: python main.py

Features:
- Paste document
- Generate summary (LangGraph + HuggingFace)
- Ask questions about the document
- Sentiment analysis on text
- Conversation memory stored in graph state
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_for_display
from src.sentiment_model import (
    train_sentiment_model,
    evaluate_model,
    save_model,
    load_model,
    load_or_create_training_data,
)
from src.graph import build_graph


DEFAULT_HF_MODEL = "google-t5/t5-small"


def ensure_sentiment_model():
    """Train and save sentiment model if not already present."""
    model_dir = PROJECT_ROOT / "models"
    if (model_dir / "sentiment_lr.pkl").exists():
        return load_model(str(model_dir))
    print("Training sentiment model (first run)...")
    vectorizer, classifier, X_test, y_test, grid_search = train_sentiment_model(
        data_path=None,
        test_size=0.2,
        n_jobs=1,
    )
    save_model(vectorizer, classifier, str(model_dir))
    metrics = evaluate_model(classifier, X_test, y_test)
    print(
        f"  Accuracy: {metrics['accuracy']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"Best params: {grid_search.best_params_}"
    )
    return vectorizer, classifier


def main():
    print("=" * 60)
    print("  AI Knowledge Assistant (LangGraph)")
    print("=" * 60)
    print()
    print("Commands: document [file.txt] | summary | ask | sentiment | history | clear | quit")
    print()

    current_document = ""
    chat_history = []  # Replaces ConversationBufferMemory; stored in graph state
    vectorizer, classifier = ensure_sentiment_model()

    summarization_chain = None
    qa_chain = None

    def get_summary_chain():
        nonlocal summarization_chain
        if summarization_chain is None:
            print("Loading summarization model (one-time)...")
            from src.summarization_chain import create_summarization_chain
            summarization_chain = create_summarization_chain(
                model_id=DEFAULT_HF_MODEL,
                max_new_tokens=150,
            )
        return summarization_chain

    def get_qa_chain():
        nonlocal qa_chain
        if qa_chain is None:
            print("Loading QA model (one-time)...")
            from src.qa_chain import create_qa_chain
            qa_chain = create_qa_chain(
                model_id=DEFAULT_HF_MODEL,
                max_new_tokens=100,
            )
        return qa_chain

    # Build LangGraph once with dependency getters
    graph = build_graph(
        get_qa_chain=get_qa_chain,
        get_summary_chain=get_summary_chain,
        vectorizer=vectorizer,
        classifier=classifier,
    )

    def run_graph(user_input: str) -> tuple[str, list]:
        """Invoke the graph and return (response, updated_chat_history)."""
        initial = {
            "user_input": user_input,
            "context": current_document,
            "chat_history": chat_history,
        }
        result = graph.invoke(initial)
        response = result.get("response") or ""
        new_history = result.get("chat_history") or []
        return response, new_history

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower().split()[0] if user_input else ""

        # ----- document: paste or load from file -----
        if cmd == "document":
            rest = user_input[9:].strip()
            if rest:
                filepath = Path(rest)
                if filepath.exists():
                    try:
                        current_document = preprocess_for_display(
                            filepath.read_text(encoding="utf-8", errors="replace")
                        )
                        chat_history.append(
                            (
                                "[User loaded a document from file]",
                                f"Document loaded ({len(current_document)} characters). "
                                "You can now use 'summary', 'ask', or 'sentiment'.",
                            )
                        )
                        print(f"\nAssistant: Document loaded from file ({len(current_document)} characters).")
                    except Exception as e:
                        print(f"\nAssistant: Could not read file: {e}")
                else:
                    print(f"\nAssistant: File not found: {filepath}")
                continue
            print()
            print("  Paste your document below, then press Enter.")
            print("  If you paste multiple lines, type END on a new line and press Enter.")
            print("  Or load from file: document path/to/file.txt")
            print()
            lines = []
            first = True
            while True:
                try:
                    line = input("  doc> ")
                except (EOFError, KeyboardInterrupt):
                    break
                if first and "\n" in line:
                    lines = [ln for ln in line.split("\n") if ln.strip() != "END"]
                    break
                first = False
                if line.strip() == "END":
                    break
                lines.append(line)
            current_document = preprocess_for_display("\n".join(lines))
            chat_history.append(
                (
                    "[User pasted a document]",
                    f"Document stored ({len(current_document)} characters). "
                    "You can now use 'summary', 'ask', or 'sentiment'.",
                )
            )
            print(f"\nAssistant: Document stored ({len(current_document)} characters).")
            continue

        # ----- summary | ask | sentiment | history: run graph -----
        if cmd in ("summary", "ask", "sentiment", "history"):
            response, chat_history = run_graph(user_input)
            if cmd == "history":
                print("\nAssistant (recent conversation):")
                print(response)
            else:
                print(f"\nAssistant: {response}")
            continue

        # ----- clear: reset document and memory -----
        if cmd == "clear":
            current_document = ""
            chat_history = []
            print("Assistant: Document and conversation cleared.")
            continue

        # ----- quit -----
        if cmd in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # Unknown: treat as question if we have a document (graph routes to qa)
        if current_document:
            response, chat_history = run_graph(user_input)
            print(f"\nAssistant: {response}")
        else:
            print(
                "Assistant: Unknown command. Use: document | summary | ask | sentiment | history | clear | quit"
            )


if __name__ == "__main__":
    main()
