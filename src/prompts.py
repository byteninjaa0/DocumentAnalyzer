"""
Prompt engineering: templates for document summarization and question answering.
"""

from langchain_core.prompts import PromptTemplate

# ----- Document Summarization -----
SUMMARIZE_TEMPLATE = """Summarize the following document in a clear and concise way. Keep the main ideas and key points. Do not add information that is not in the document.

Document:
{document}

Summary:"""

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["document"],
    template=SUMMARIZE_TEMPLATE,
)

# ----- Question Answering -----
QA_TEMPLATE = """Use the following context to answer the question. If the answer cannot be found in the context, say "I don't have enough information in the document to answer that."

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_TEMPLATE,
)

# ----- Chat-style QA (when we have conversation history) -----
QA_WITH_HISTORY_TEMPLATE = """You are a helpful assistant. Use the context below to answer the question. If the answer is not in the context, say so.

Context:
{context}

Current question: {question}

Answer:"""

QA_WITH_HISTORY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_WITH_HISTORY_TEMPLATE,
)
