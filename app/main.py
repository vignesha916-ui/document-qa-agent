from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from app.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    ClassifyRequest,
    ClassifyResponse
)
from app.tools.embedder import ingest_document, is_index_loaded
from app.agent import run_agent

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "data/sample_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Routes ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the service is running and index is loaded."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        index_loaded=is_index_loaded()
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF or TXT).
    Chunks and embeds it into the FAISS vector store.
    """
    # Validate file type
    allowed = [".pdf", ".txt", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use: {allowed}"
        )

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest
    try:
        chunks_created = ingest_document(file_path)
        return IngestResponse(
            success=True,
            message=f"Successfully ingested '{file.filename}'",
            chunks_created=chunks_created
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Ask a question about the ingested document.
    Runs the full LangGraph agent pipeline:
    retrieve → confidence check → synthesise answer
    """
    if not is_index_loaded():
        raise HTTPException(
            status_code=400,
            detail="No document ingested yet. POST a file to /ingest first."
        )

    try:
        result = run_agent(request.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["chunks"],
            confidence=result["confidence"],
            grounded=result["grounded"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )

@app.post("/classify", response_model=ClassifyResponse)
def classify_document(request: ClassifyRequest):
    """
    Classify and extract key information from document text.
    Designed for UiPath integration — accepts extracted text,
    returns structured JSON for bot consumption.
    """
    if not is_index_loaded():
        raise HTTPException(
            status_code=400,
            detail="No document ingested. POST to /ingest first."
        )

    try:
        # Ask three structured questions via the agent
        doc_type_result = run_agent(
            "What type of document is this? "
            "Reply with one of: Invoice, Contract, Report, "
            "Email, Form, Letter, or Other."
        )

        summary_result = run_agent(
            "Summarise this document in exactly two sentences."
        )

        fields_result = run_agent(
            "Extract key fields as a list. Include: "
            "dates, amounts, parties, reference numbers, "
            "and any other important values."
        )

        # Determine if human review needed
        avg_confidence = (
            doc_type_result["confidence"] +
            summary_result["confidence"] +
            fields_result["confidence"]
        ) / 3

        return ClassifyResponse(
            document_type=doc_type_result["answer"],
            summary=summary_result["answer"],
            key_fields={"extracted": fields_result["answer"]},
            confidence=round(avg_confidence, 4),
            requires_human_review=avg_confidence < 0.4
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )