"""
Text preprocessing module for the AI Knowledge Assistant.
Provides: text cleaning, tokenization, and stopword removal.
"""

import re
import string
from typing import List

# Use NLTK-style stopwords; we avoid NLTK dependency by using a minimal set
# so the project runs with only the required stack (sklearn has no stopwords)
STOPWORDS_EN = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now",
}


def clean_text(text: str) -> str:
    """
    Clean raw text: lowercase, remove extra whitespace, remove numbers/punctuation.
    
    Example:
        >>> clean_text("  Hello World! 123  ")
        'hello world'
    """
    if not text or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower().strip()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove numbers (optional: keep if needed for sentiment)
    text = re.sub(r"\d+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces/newlines to single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize cleaned text into a list of words (whitespace tokenization).
    
    Example:
        >>> tokenize("hello world example")
        ['hello', 'world', 'example']
    """
    if not text:
        return []
    return text.split()


def remove_stopwords(tokens: List[str], stopwords: set = None) -> List[str]:
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens: List of token strings.
        stopwords: Set of stopwords (default: English stopwords).
    
    Example:
        >>> remove_stopwords(['this', 'is', 'a', 'test'])
        ['test']
    """
    stopwords = stopwords or STOPWORDS_EN
    return [t for t in tokens if t and t not in stopwords]


def preprocess_pipeline(text: str, remove_stopwords_flag: bool = True) -> List[str]:
    """
    Full pipeline: clean -> tokenize -> (optional) remove stopwords.
    Returns list of tokens suitable for TF-IDF or sentiment input.
    
    Example:
        >>> preprocess_pipeline("This is a sample document. It has multiple sentences!")
        ['sample', 'document', 'multiple', 'sentences']
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens)
    return tokens


def preprocess_for_display(text: str) -> str:
    """
    Clean text for display or for feeding to LLMs (keep structure, no tokenization).
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
