"""
Question answering using a prompt-based LLM chain (LangChain + local HuggingFace).
Users can ask questions about the provided context (e.g. document).
"""

from typing import Optional

# Reasonable context length for QA
MAX_CONTEXT_LENGTH = 1024


def _get_llm(model_id: str = "google-t5/t5-small", max_new_tokens: int = 100):
    """Build LangChain LLM from T5 (seq2seq) via compatibility helper (transformers 5.x removed text2text-generation)."""
    try:
        from langchain_community.llms import HuggingFacePipeline
        from .hf_t5_pipeline import get_t5_pipeline
    except ImportError:
        raise ImportError(
            "Install: pip install langchain-community transformers torch"
        )

    pipe = get_t5_pipeline(model_id=model_id, max_new_tokens=max_new_tokens)
    return HuggingFacePipeline(pipeline=pipe)


def _truncate_context(text: str, max_chars: int = MAX_CONTEXT_LENGTH * 4) -> str:
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def create_qa_chain(
    model_id: str = "google-t5/t5-small",
    max_new_tokens: int = 100,
):
    """
    Create LangChain RunnableSequence for QA: PromptTemplate → LLM → Output.
    """
    from langchain_core.output_parsers import StrOutputParser
    from .prompts import QA_PROMPT

    llm = _get_llm(model_id=model_id, max_new_tokens=max_new_tokens)
    chain = QA_PROMPT | llm | StrOutputParser()
    return chain


def answer_question(
    context: str,
    question: str,
    model_id: str = "google-t5/t5-small",
    max_new_tokens: int = 100,
    chain=None,
) -> str:
    """
    Answer a question given a context (e.g. the current document).
    """
    context = _truncate_context(context)
    if chain is None:
        chain = create_qa_chain(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    result = chain.invoke({"context": context, "question": question})
    if isinstance(result, str):
        return result.strip()
    return str(result).strip()
