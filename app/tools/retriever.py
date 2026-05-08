import app.tools.embedder as embedder_module
from app.config import TOP_K_RESULTS, CONFIDENCE_THRESHOLD
from app.models.schemas import SourceChunk
from typing import List, Tuple


def retrieve_chunks(question: str) -> Tuple[List[SourceChunk], float]:
    """
    Retrieve top-K chunks from FAISS index.
    Returns chunks and average confidence score.
    """
    if embedder_module.vector_store is None:
        raise RuntimeError(
            "Vector store not initialised. "
            "Please ingest a document first via POST /ingest"
        )

    # Similarity search with scores
    results = embedder_module.vector_store.similarity_search_with_score(
        question,
        k=TOP_K_RESULTS
    )

    chunks = []
    scores = []

    for i, (doc, score) in enumerate(results):
        # FAISS returns L2 distance — convert to similarity
        similarity = 1 / (1 + score)
        scores.append(similarity)

        chunks.append(SourceChunk(
            content=doc.page_content,
            score=round(similarity, 4),
            chunk_index=i
        ))

    avg_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0
    return chunks, avg_confidence


def is_grounded(confidence: float) -> bool:
    """
    Determine if retrieved context is strong enough
    to ground an answer — prevents hallucination.
    """
    return confidence >= CONFIDENCE_THRESHOLD