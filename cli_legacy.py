"""
AI Knowledge Assistant - Legacy CLI Chatbot (LangChain only, no LangGraph).
Run from project root: python cli_legacy.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_pipeline, preprocess_for_display
from src.sentiment_model import (
    train_sentiment_model,
    evaluate_model,
    save_model,
    load_model,
    predict_sentiment,
    load_or_create_training_data,
)
from src.summarization_chain import create_summarization_chain, summarize_document
from src.qa_chain import create_qa_chain, answer_question
from src.chatbot_memory import ConversationBufferMemory


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
    print("  AI Knowledge Assistant (GenAI Portfolio Project)")
    print("=" * 60)
    print()
    print("Commands: document [file.txt] | summary | ask | sentiment | history | clear | quit")
    print()

    current_document: str = ""
    memory = ConversationBufferMemory(max_turns=20)
    vectorizer, classifier = ensure_sentiment_model()

    summarization_chain = None
    qa_chain = None

    def get_summary_chain():
        nonlocal summarization_chain
        if summarization_chain is None:
            print("Loading summarization model (one-time)...")
            summarization_chain = create_summarization_chain(
                model_id=DEFAULT_HF_MODEL,
                max_new_tokens=150,
            )
        return summarization_chain

    def get_qa_chain():
        nonlocal qa_chain
        if qa_chain is None:
            print("Loading QA model (one-time)...")
            qa_chain = create_qa_chain(
                model_id=DEFAULT_HF_MODEL,
                max_new_tokens=100,
            )
        return qa_chain

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower().split()[0] if user_input else ""

        if cmd == "document":
            rest = user_input[9:].strip()
            if rest:
                filepath = Path(rest)
                if filepath.exists():
                    try:
                        current_document = preprocess_for_display(filepath.read_text(encoding="utf-8", errors="replace"))
                        memory.add_user_message("[User loaded a document from file]")
                        memory.add_ai_message(
                            f"Document loaded ({len(current_document)} characters). "
                            "You can now use 'summary', 'ask', or 'sentiment'."
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
            memory.add_user_message("[User pasted a document]")
            memory.add_ai_message(
                f"Document stored ({len(current_document)} characters). "
                "You can now use 'summary', 'ask', or 'sentiment'."
            )
            print(f"\nAssistant: Document stored ({len(current_document)} characters).")
            continue

        if cmd == "summary":
            if not current_document:
                print("Assistant: Paste a document first (command: document).")
                continue
            memory.add_user_message("Generate summary")
            try:
                chain = get_summary_chain()
                summary = summarize_document(current_document, chain=chain)
                memory.add_ai_message(summary)
                print(f"\nAssistant (summary):\n{summary}")
            except Exception as e:
                msg = f"Summary failed: {e}"
                memory.add_ai_message(msg)
                print(f"\nAssistant: {msg}")
            continue

        if cmd == "ask":
            if not current_document:
                print("Assistant: Paste a document first (command: document).")
                continue
            question = user_input[3:].strip() if len(user_input) > 3 else ""
            if not question:
                print("Ask what? Type: ask <your question>")
                continue
            memory.add_user_message(question)
            try:
                chain = get_qa_chain()
                answer = answer_question(current_document, question, chain=chain)
                memory.add_ai_message(answer)
                print(f"\nAssistant: {answer}")
            except Exception as e:
                msg = f"QA failed: {e}"
                memory.add_ai_message(msg)
                print(f"\nAssistant: {msg}")
            continue

        if cmd == "sentiment":
            text = user_input[9:].strip() if len(user_input) > 9 else current_document
            if not text:
                print("Assistant: Paste a document or type: sentiment <sentence>")
                continue
            memory.add_user_message(f"Sentiment: {text[:50]}...")
            try:
                label, confidence = predict_sentiment(
                    text, vectorizer=vectorizer, classifier=classifier,
                )
                sentiment = "positive" if label == 1 else "negative"
                memory.add_ai_message(sentiment)
                print(f"\nAssistant: Sentiment = {sentiment} (confidence: {confidence:.2f})")
            except Exception as e:
                print(f"\nAssistant: Sentiment failed: {e}")
            continue

        if cmd == "history":
            hist = memory.get_history_as_string(last_n=20)
            print("\nAssistant (recent conversation):")
            print(hist if hist else "No history yet.")
            continue

        if cmd == "clear":
            current_document = ""
            memory.clear()
            print("Assistant: Document and conversation cleared.")
            continue

        if cmd in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if current_document:
            memory.add_user_message(user_input)
            try:
                chain = get_qa_chain()
                answer = answer_question(current_document, user_input, chain=chain)
                memory.add_ai_message(answer)
                print(f"\nAssistant: {answer}")
            except Exception as e:
                print(
                    "\nAssistant: Unknown command. Use: document | summary | ask | sentiment | history | clear | quit"
                )
        else:
            print(
                "Assistant: Unknown command. Use: document | summary | ask | sentiment | history | clear | quit"
            )


if __name__ == "__main__":
    main()
