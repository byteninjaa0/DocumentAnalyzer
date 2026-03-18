"""
Sentiment node: runs TF-IDF + Logistic Regression sentiment on text.
Uses state["user_input"] (e.g. text after "sentiment ") or state["context"] as fallback.
"""

from src.state import GraphState

_vectorizer = None
_classifier = None


def set_sentiment_model(vectorizer, classifier):
    global _vectorizer, _classifier
    _vectorizer = vectorizer
    _classifier = classifier


def sentiment_node(state: GraphState) -> dict:
    """
    Predict sentiment for the given text. Sets state["response"] with result message.
    """
    user_input = (state.get("user_input") or "").strip()
    context = state.get("context") or ""
    # Use text after "sentiment " or the whole document if user said just "sentiment"
    if user_input == "sentiment":
        text = context
    elif user_input.startswith("sentiment "):
        text = user_input[9:].strip()
    else:
        text = user_input or context
    if not text:
        return {"response": "Paste a document or type: sentiment <sentence>"}

    if _vectorizer is None or _classifier is None:
        return {"response": "Sentiment model not loaded."}

    try:
        from src.sentiment_model import predict_sentiment
        label, confidence = predict_sentiment(
            text, vectorizer=_vectorizer, classifier=_classifier
        )
        sentiment = "positive" if label == 1 else "negative"
        return {"response": f"Sentiment = {sentiment} (confidence: {confidence:.2f})"}
    except Exception as e:
        return {"response": f"Sentiment failed: {e}"}
