from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
import os

# Global vector store — held in memory
vector_store = None


def load_document(file_path: str):
    """Load a PDF or text document."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")
    return loader.load()


def chunk_documents(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vector_store(chunks):
    """Embed chunks and build FAISS index."""
    global vector_store
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def ingest_document(file_path: str) -> int:
    """
    Full ingestion pipeline:
    Load → Chunk → Embed → Index
    Returns number of chunks created.
    """
    documents = load_document(file_path)
    chunks = chunk_documents(documents)
    build_vector_store(chunks)
    return len(chunks)


def is_index_loaded() -> bool:
    """Check if vector store is ready."""
    return vector_store is not None