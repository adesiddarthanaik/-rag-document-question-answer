# RAG Document QA

A modular **Retrieval-Augmented Generation** pipeline that answers questions over your own PDF documents. Built while following Krish Naik's RAG tutorial, then restructured from notebooks into a clean, runnable Python package.

Ask a question → relevant chunks are retrieved from a vector store → a Groq-hosted LLM answers using only that context, with source citations.

## Architecture

```
PDFs ──► ingestion ──► chunks ──► embeddings ──► ChromaDB (vector store)
                                                      │
question ──► embed query ──► retrieve top-k chunks ◄──┘
                                  │
                                  ▼
                         Groq LLM ──► grounded answer + sources
```

| Component | File | Tech |
|-----------|------|------|
| Loading & chunking | `src/ingestion.py` | PyPDFLoader, RecursiveCharacterTextSplitter |
| Embeddings | `src/embeddings.py` | SentenceTransformer (`all-MiniLM-L6-v2`) |
| Vector store | `src/vectorstore.py` | ChromaDB (persistent) |
| Retrieval | `src/retriever.py` | cosine similarity |
| Generation | `src/llm.py` | Groq (`gemma2-9b-it`) |
| Orchestration | `src/pipeline.py` | `RAGPipeline` |
| CLI | `app.py` | argparse |
| Evaluation (optional) | `evaluate.py` | LangSmith, LLM-as-judge |

## Setup

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env   # then add your GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com.

## Usage

```bash
# 1. Drop your PDFs into data/pdf/, then index them
python app.py ingest --pdf-dir data/pdf

# 2. Ask a one-off question
python app.py ask "What is the attention mechanism?"

# 3. Or start an interactive chat
python app.py chat
```

## Evaluation (optional)

`evaluate.py` shows how to grade answers with LangSmith using correctness and
concision metrics. Install the extras and add the LangSmith / OpenAI keys to `.env`:

```bash
pip install langsmith openai
python evaluate.py
```

## Web UI

A Streamlit interface (`app_ui.py`) reuses the same `src/` modules: upload PDFs,
build an in-memory index, and ask questions in the browser. Each visitor gets an
isolated session — documents are never shared or written to disk.

```bash
pip install -r requirements.txt
# uses your GROQ_API_KEY from .env (or a key box appears if none is set)
streamlit run app_ui.py
```

## Deploy (so anyone can use it via a URL)

The deployed app uses **one** Groq key stored as a deployment secret, so visitors
just upload and ask — no key needed from them.

**Streamlit Community Cloud (simplest):**
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app → pick this repo, main file `app_ui.py`.
3. Advanced settings → Secrets → add:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   ```
4. Deploy. Share the URL.

**HuggingFace Spaces (alternative):** create a Streamlit Space, push the same files,
add `GROQ_API_KEY` under the Space's Settings → Secrets.

> Note: free hosting tiers are CPU-only with limited RAM. This app runs there because
> the heavy LLM work happens on Groq's servers via the API, not on the host.

## Possible extensions

- Swap the CLI for a Streamlit UI
- Add hybrid search (BM25 + dense)
- Support `.txt`, `.docx`, and web URLs as sources
- Add a re-ranker before generation

