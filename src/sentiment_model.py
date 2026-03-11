"""
Sentiment Analysis Model using TF-IDF vectorization and Logistic Regression.
Includes train/test split, evaluation metrics, and GridSearchCV hyperparameter tuning.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Default path for saving/loading the trained model and vectorizer
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _get_sample_data() -> pd.DataFrame:
    """
    Return sample sentiment data for training when no CSV is provided.
    Labels: 1 = positive, 0 = negative.
    """
    texts = [
        "I love this product it is amazing",
        "Terrible experience would not recommend",
        "Great service and fast delivery",
        "Waste of money very disappointed",
        "Excellent quality and value",
        "Poor quality broke immediately",
        "Best purchase I have ever made",
        "Not worth the price at all",
        "Highly recommend to everyone",
        "Complete waste of time",
        "Outstanding performance",
        "Very bad and frustrating",
        "Could not be happier",
        "Regret buying this",
        "Perfect in every way",
        "Worst product ever",
        "Fantastic and reliable",
        "Do not buy",
        "Super satisfied with result",
        "Absolutely horrible",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return pd.DataFrame({"text": texts, "label": labels})


def load_or_create_training_data(data_path: str = None) -> pd.DataFrame:
    """
    Load training data from CSV or use built-in sample data.
    CSV must have columns: text, label (0=negative, 1=positive).
    """
    if data_path and os.path.isfile(data_path):
        df = pd.read_csv(data_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns")
        return df
    return _get_sample_data()


def train_sentiment_model(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    n_jobs: int = -1,
) -> tuple:
    """
    Train TF-IDF + Logistic Regression with optional GridSearchCV.
    Returns (vectorizer, classifier, X_test, y_test) for evaluation.
    """
    df = load_or_create_training_data(data_path)
    X = df["text"].astype(str).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [500, 1000],
    }
    base_lr = LogisticRegression(random_state=random_state)
    grid_search = GridSearchCV(
        base_lr,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=n_jobs,
        verbose=0,
    )
    grid_search.fit(X_train_tfidf, y_train)
    classifier = grid_search.best_estimator_

    return vectorizer, classifier, X_test_tfidf, y_test, grid_search


def evaluate_model(
    classifier,
    X_test,
    y_test,
    target_names: list = None,
) -> dict:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix.
    """
    y_pred = classifier.predict(X_test)
    target_names = target_names or ["negative", "positive"]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        ),
        "y_pred": y_pred,
    }


def save_model(vectorizer, classifier, model_dir: str = None):
    """Save vectorizer and classifier to disk."""
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "sentiment_lr.pkl", "wb") as f:
        pickle.dump(classifier, f)


def load_model(model_dir: str = None) -> tuple:
    """Load vectorizer and classifier from disk."""
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    with open(model_dir / "tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "sentiment_lr.pkl", "rb") as f:
        classifier = pickle.load(f)
    return vectorizer, classifier


def predict_sentiment(
    text: str,
    vectorizer=None,
    classifier=None,
    model_dir: str = None,
) -> tuple:
    """
    Predict sentiment for a single text.
    Returns (label, confidence) where label is 0 (negative) or 1 (positive).
    """
    if vectorizer is None or classifier is None:
        vectorizer, classifier = load_model(model_dir)
    X = vectorizer.transform([text])
    label = int(classifier.predict(X)[0])
    proba = classifier.predict_proba(X)[0]
    confidence = float(proba[label])
    return label, confidence
