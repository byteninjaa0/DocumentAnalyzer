"""
Document summarization using LangChain and a local HuggingFace model (e.g. t5-small).
Implements: PromptTemplate → LLM → Output using RunnableSequence (LCEL).
"""

import os
from typing import Optional

# Truncate long documents to avoid OOM; T5 has limited context
MAX_INPUT_LENGTH = 512


def _get_llm(model_id: str = "google-t5/t5-small", max_new_tokens: int = 150):
    """
    Build a LangChain-compatible LLM from T5 (seq2seq) for summarization.
    Uses hf_t5_pipeline helper (transformers 5.x removed text2text-generation).
    """
    try:
        from langchain_community.llms import HuggingFacePipeline
        from .hf_t5_pipeline import get_t5_pipeline
    except ImportError:
        raise ImportError(
            "Install: pip install langchain-community transformers torch"
        )

    device_map = "auto" if _has_cuda() else None
    pipe = get_t5_pipeline(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        device_map=device_map,
    )
    return HuggingFacePipeline(pipeline=pipe)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _truncate_for_model(text: str, max_chars: int = MAX_INPUT_LENGTH * 4) -> str:
    """Truncate document to avoid exceeding model context."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

def create_summarization_chain(
    model_id: str = "google-t5/t5-small",
    max_new_tokens: int = 150,
):
    """
    Create a LangChain LCEL chain: PromptTemplate → LLM → Output (RunnableSequence).
    """
    from langchain_core.output_parsers import StrOutputParser
    from .prompts import SUMMARIZE_PROMPT

    llm = _get_llm(model_id=model_id, max_new_tokens=max_new_tokens)
    # RunnableSequence: prompt | llm | parser
    chain = SUMMARIZE_PROMPT | llm | StrOutputParser()
    return chain


def summarize_document(
    document: str,
    model_id: str = "google-t5/t5-small",
    max_new_tokens: int = 150,
    chain=None,
) -> str:
    """
    Summarize a document using the LangChain summarization chain.
    """
    document = _truncate_for_model(document)
    if chain is None:
        chain = create_summarization_chain(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    result = chain.invoke({"document": document})
    # LLM may return extra whitespace or newlines
    if isinstance(result, str):
        return result.strip()
    return str(result).strip()
