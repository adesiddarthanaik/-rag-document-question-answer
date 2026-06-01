"""Streamlit web UI for the RAG pipeline (level 2 / level 3).

Run locally:   streamlit run app_ui.py
Reuses the same src/ modules as the CLI. Each visitor uses their own Groq key
and gets an isolated, in-memory vector store (nothing is shared or saved to disk).
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.embeddings import EmbeddingManager
from src.ingestion import load_pdfs, split_documents
from src.llm import GroqLLM
from src.retriever import Retriever
from src.vectorstore import VectorStore

st.set_page_config(page_title="RAG Document QA", page_icon="📄")


def configured_groq_key() -> str | None:
    """Read the Groq key from Streamlit secrets or env (set on the deployment)."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:  # no secrets file present (e.g. local run)
        pass
    return os.environ.get("GROQ_API_KEY")


# The embedding model is document-independent, so load it once and share it
# across reruns/sessions. This avoids re-downloading ~80MB on every question.
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder() -> EmbeddingManager:
    return EmbeddingManager("all-MiniLM-L6-v2")


def build_index(files, embedder: EmbeddingManager) -> VectorStore:
    """Write uploaded PDFs to a temp dir, chunk, embed, and index in memory."""
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            Path(tmp, f.name).write_bytes(f.getbuffer())
        docs = load_pdfs(tmp)
    chunks = split_documents(docs)
    store = VectorStore(persist_directory=None)  # in-memory, per session
    store.add(chunks, embedder.embed([c.page_content for c in chunks]))
    return store


st.title("📄 RAG Document QA")
st.caption("Upload PDFs, ask questions, get answers grounded in your documents.")

with st.sidebar:
    st.header("Setup")
    server_key = configured_groq_key()
    if server_key:
        groq_key = server_key
    else:
        groq_key = st.text_input(
            "Groq API key", type="password",
            help="Set GROQ_API_KEY in the deployment secrets to hide this. "
                 "Free key: https://console.groq.com",
        )
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if st.button("Build index", disabled=not files):
        st.session_state.store = build_index(files, get_embedder())
        st.session_state.n_files = len(files)
        st.success(f"Indexed {len(files)} file(s). Ask away.")

if "store" in st.session_state:
    st.info(f"{st.session_state.n_files} document(s) indexed.")
    question = st.text_input("Your question")
    if question:
        if not groq_key:
            st.warning("No Groq API key configured. Add one in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                retriever = Retriever(st.session_state.store, get_embedder())
                llm = GroqLLM(api_key=groq_key)
                results = retriever.retrieve(question, top_k=4)
                if not results:
                    st.write("No relevant context found.")
                else:
                    answer = llm.answer(question, [r["content"] for r in results])
                    st.markdown("### Answer")
                    st.write(answer)
                    with st.expander("Sources"):
                        for r in results:
                            m = r["metadata"]
                            st.write(
                                f"- **{m.get('source_file', '?')}** "
                                f"(page {m.get('page', '?')}, "
                                f"score {r['similarity_score']:.3f})"
                            )
else:
    st.write("Upload PDFs and click **Build index** in the sidebar to begin.")
