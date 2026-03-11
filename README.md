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

```
User → Prompt → LangChain Chain → LLM (HuggingFace) → Output
```

- **Prompt engineering**: Templates for summarization and question-answering live in `src/prompts.py`.
- **LangChain workflow**: Implemented as **RunnableSequence** (LCEL): `PromptTemplate | LLM | StrOutputParser`.
- **LLM**: Local HuggingFace pipeline (e.g. `google-t5/t5-small`) for text-to-text generation.
- **Sentiment**: Separate pipeline: text → preprocessing → TF-IDF → Logistic Regression (with GridSearchCV) → label + confidence.
- **Memory**: Conversation buffer stores (user, assistant) turns for the session.

---

## Technologies Used

| Area | Technology |
|------|------------|
| Language | Python |
| Orchestration | LangChain (chains, prompts, LCEL) |
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
│   ├── preprocessing.py      # Text cleaning, tokenization, stopword removal
│   ├── sentiment_model.py    # TF-IDF, Logistic Regression, metrics, GridSearchCV
│   ├── summarization_chain.py # LangChain + HuggingFace document summarization
│   ├── qa_chain.py           # Prompt-based question-answering chain
│   ├── chatbot_memory.py     # Conversation buffer memory
│   └── prompts.py            # Prompt templates (summarization, QA)
│
├── app.py                # CLI chatbot entry point
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
