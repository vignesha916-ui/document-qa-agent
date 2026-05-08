from pydantic import BaseModel, Field
from typing import Optional, List


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: str = Field(..., description="Path to the document to ingest")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    message: str
    chunks_created: int


class QueryRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask about the document"
    )


class SourceChunk(BaseModel):
    """A retrieved source chunk with metadata."""
    content: str
    score: float
    chunk_index: int


class QueryResponse(BaseModel):
    """Response model for Q&A."""
    question: str
    answer: str
    sources: List[SourceChunk]
    confidence: float
    grounded: bool = Field(
        description="True if answer is grounded in retrieved context"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    index_loaded: bool