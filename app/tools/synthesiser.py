from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import ANTHROPIC_API_KEY
from app.models.schemas import SourceChunk
from typing import List


# Initialise Claude
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=1024,
    temperature=0.1
)

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Your job is to answer questions based ONLY on the provided context.
Rules:
- Only use information from the context below
- If the context does not contain enough information, say so clearly
- Always be concise and factual
- Never invent or extrapolate information not present in the context
- Cite which part of the context supports your answer"""

OUT_OF_SCOPE_RESPONSE = (
    "I cannot find sufficient information in the provided document "
    "to answer this question confidently. Please try rephrasing "
    "your question or ensure the document contains relevant information."
)


def synthesise_answer(
    question: str,
    chunks: List[SourceChunk],
    grounded: bool
) -> str:
    """
    Generate a grounded answer using Claude.
    If confidence is too low, return out-of-scope message.
    """
    if not grounded:
        return OUT_OF_SCOPE_RESPONSE

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([
        f"[Chunk {c.chunk_index + 1}] {c.content}"
        for c in chunks
    ])

    user_message = f"""Context from document:
{context}

Question: {question}

Answer based strictly on the context above:"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    return response.content.strip()