import os
from dotenv import load_dotenv

load_dotenv()

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

# App
APP_TITLE = "Document Q&A Agent"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Production-grade agentic document Q&A using LangGraph and RAG"