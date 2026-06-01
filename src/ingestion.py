"""Data ingestion: load documents from disk and split into chunks."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdfs(pdf_dir: str) -> List[Document]:
    """Load every PDF under `pdf_dir` (recursively) into LangChain Documents."""
    docs: List[Document] = []
    pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s) in {pdf_dir}")

    for pdf_file in pdf_files:
        try:
            loaded = PyPDFLoader(str(pdf_file)).load()
            for d in loaded:
                d.metadata["source_file"] = pdf_file.name
                d.metadata["file_type"] = "pdf"
            docs.extend(loaded)
            print(f"  loaded {len(loaded):>3} pages  <- {pdf_file.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR loading {pdf_file.name}: {e}")

    print(f"Total pages loaded: {len(docs)}")
    return docs


def load_texts(text_dir: str) -> List[Document]:
    """Load every .txt file under `text_dir`."""
    loader = DirectoryLoader(
        text_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} text document(s) from {text_dir}")
    return docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s)")
    return chunks
