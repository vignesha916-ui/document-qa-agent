# Document Q&A Agent

> Production-grade agentic document Q&A system built with LangGraph, RAG, and FastAPI

## What It Does

Upload any PDF or text document and ask natural language questions about it. 
The system retrieves relevant context, checks confidence to prevent hallucination, 
and generates grounded answers using Claude with full source attribution.

## Architecture
User Question
↓
FastAPI Endpoint (/ask)
↓
LangGraph Agent Orchestrator
├── Node 1: FAISS Vector Retrieval
│         (HuggingFace sentence-transformers)
├── Node 2: Confidence Scoring
│         (hallucination prevention)
└── Node 3: Claude Answer Synthesis
(grounded response + source attribution)
↓
JSON Response with answer, sources, confidence score

## Key Design Decisions

- **LangGraph over simple chains** — stateful multi-node agent with conditional 
  routing enables robust error handling and extensibility
- **Confidence scoring** — similarity threshold prevents Claude from hallucinating 
  on out-of-scope queries; returns explicit "cannot answer" instead
- **Sliding window chunking** — overlapping chunks preserve context across 
  chunk boundaries for better retrieval quality
- **Source attribution** — every answer includes the retrieved chunks that 
  grounded it, making responses verifiable

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph |
| RAG retrieval | FAISS + sentence-transformers |
| LLM | Anthropic Claude (claude-haiku-4-5) |
| API layer | FastAPI |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Containerisation | Docker |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health + index status |
| POST | `/ingest` | Upload and embed a document |
| POST | `/ask` | Ask a question, get grounded answer |

## Quick Start

```bash
# Clone
git clone https://github.com/vignesha916-ui/document-qa-agent.git
cd document-qa-agent

# Install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Run
uvicorn app.main:app --reload

# Open API docs
# http://127.0.0.1:8000/docs
```

## Example Response

```json
{
  "question": "Where is Vignesh based and what is his visa status?",
  "answer": "Vignesh Ajith is based in Dublin, Ireland. He holds a 
             Stamp 1G Graduate Permission with full right to work in Ireland.",
  "sources": [
    {
      "content": "Vignesh Ajith is an RPA Developer and AI Engineer 
                  based in Dublin, Ireland...",
      "score": 0.4727,
      "chunk_index": 0
    }
  ],
  "confidence": 0.4727,
  "grounded": true
}
```

## Project Structure

document-qa-agent/
├── app/
│   ├── main.py          # FastAPI app
│   ├── agent.py         # LangGraph state machine
│   ├── config.py        # Environment config
│   ├── models/
│   │   └── schemas.py   # Pydantic models
│   └── tools/
│       ├── embedder.py  # Document ingestion + FAISS
│       ├── retriever.py # Semantic search
│       └── synthesiser.py # Claude integration
├── tests/
├── Dockerfile
└── requirements.txt

## Author

**Vignesh Ajith** — RPA Developer & AI Engineer, Dublin Ireland  
MSc Artificial Intelligence, Dublin City University (2025)  
[LinkedIn](https://linkedin.com/in/vignesh-ajith-04377512a)