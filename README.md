# AI Knowledge Assistant

A full end-to-end **Generative AI** system for your portfolio. It summarizes documents, answers questions about them, maintains conversation history, and performs sentiment analysis—using **only free local models** (HuggingFace). No OpenAI or other paid APIs.

---

## Project Overview

The AI Knowledge Assistant lets you:

1. **Paste a document** and store it in the session.
2. **Generate a summary** using a local HuggingFace model (e.g. T5-small) via LangChain.
3. **Ask questions** about the document with a prompt-based LLM chain.
4. **Analyze sentiment** of text using a TF-IDF + Logistic Regression classifier (trained locally).
5. **Keep conversation memory** so the chatbot remembers prior interactions in the session.

All processing runs **locally** (CPU or GPU). No data is sent to external APIs.

---

## System Architecture

### LangGraph (recommended: `main.py`)

The application is refactored to a **stateful LangGraph** workflow:

```
START → router_node → [qa_node | summarize_node | sentiment_node | history_node] → memory_node → END
```

- **State**: Central `GraphState` (`src/state.py`) with `user_input`, `task`, `context`, `response`, and `chat_history` (list with append reducer).
- **Router**: Analyzes `user_input` and sets `task`; `add_conditional_edges` routes to the correct task node.
- **Nodes**: Each major function is a node in `src/nodes/` (router, qa, summarize, sentiment, history, memory). QA and summarization reuse the existing HuggingFace chains; sentiment reuses TF-IDF + Logistic Regression.
- **Memory**: Stored inside `GraphState.chat_history`; `memory_node` appends each turn after a response. No separate `ConversationBufferMemory`.
- **Entry point**: `python main.py` builds the graph with dependency injection and invokes it per user input.

### Legacy chains (still used inside nodes)

```
User → Prompt → LangChain Chain → LLM (HuggingFace) → Output
```

- **Prompt engineering**: Templates in `src/prompts.py`.
- **Chains**: `src/qa_chain.py` and `src/summarization_chain.py` are used by the QA and summarization nodes.
- **LLM**: Local HuggingFace (e.g. `google-t5/t5-small`). **Sentiment**: TF-IDF + Logistic Regression.

---

## Technologies Used

| Area | Technology |
|------|------------|
| Language | Python |
| Orchestration | LangGraph (state graph, nodes, conditional edges) + LangChain (chains inside nodes) |
| LLM | HuggingFace Transformers (T5-small, local) |
| ML / NLP | scikit-learn (TF-IDF, Logistic Regression, GridSearchCV) |
| Data | NumPy, Pandas |
| Runtime | PyTorch (CPU or CUDA) |

---

## Project Structure

```
genai-document-assistant/
│
├── data/                 # Optional: CSV for sentiment training (text, label)
├── models/               # Saved sentiment model (TF-IDF + classifier)
│
├── src/
│   ├── state.py              # GraphState (TypedDict) for LangGraph
│   ├── graph.py              # LangGraph StateGraph, conditional edges, build_graph()
│   ├── nodes/                # LangGraph nodes
│   │   ├── router.py         # Routes by user_input → task
│   │   ├── qa_node.py        # Question answering over context
│   │   ├── summarize_node.py # Document summarization
│   │   ├── sentiment_node.py # Sentiment analysis (TF-IDF + LR)
│   │   ├── history_node.py   # Format chat history
│   │   └── memory_node.py    # Append turn to chat_history
│   ├── preprocessing.py     # Text cleaning, tokenization
│   ├── sentiment_model.py   # TF-IDF, Logistic Regression, metrics
│   ├── summarization_chain.py # LangChain + HuggingFace summarization
│   ├── qa_chain.py          # Prompt-based QA chain
│   ├── hf_t5_pipeline.py    # T5 pipeline for LangChain
│   ├── prompts.py           # Prompt templates
│   └── chatbot_memory.py    # (Legacy) conversation buffer; memory now in GraphState
│
├── main.py               # CLI entry point (LangGraph)
├── app.py                # Legacy CLI (direct chains + ConversationBufferMemory)
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Create a virtual environment (recommended)

```bash
cd genai-document-assistant
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the CLI chatbot

**LangGraph (recommended):**
```bash
python main.py
```

**Legacy (chains only):**
```bash
python app.py
```

### 4. Use the assistant

- **`document`** – Paste your text (then press Enter; if multi-line, type `END` on a new line and press Enter). Or load from file: **`document path/to/file.txt`**.
- **`summary`** – Summarize the current document.
- **`ask <question>`** – Ask a question about the document.
- **`sentiment`** or **`sentiment <sentence>`** – Sentiment of the document or the given sentence.
- **`history`** – Show recent conversation.
- **`clear`** – Clear document and conversation.
- **`quit`** – Exit.

**Example flow**

1. Type `document`, then paste a short article or paragraph, then type `END`.
2. Type `summary` to get a summary.
3. Type `ask What is the main point?` to ask about the document.
4. Type `sentiment` to get sentiment of the document (or `sentiment I love this` for a single sentence).
5. Type `history` to see the conversation so far.

---

## Sentiment model (optional custom data)

By default, a small built-in dataset is used to train the sentiment model on first run. To use your own data:

1. Add a CSV in `data/` with columns **`text`** and **`label`** (0 = negative, 1 = positive).
2. In `app.py` or when calling `train_sentiment_model()`, pass `data_path="data/your_file.csv"`.

The pipeline uses **TF-IDF** + **Logistic Regression** with **GridSearchCV** and reports **accuracy**, **precision**, **recall**, **F1**, and **confusion matrix**.

---

## License

MIT. Use and adapt for your portfolio and learning.
