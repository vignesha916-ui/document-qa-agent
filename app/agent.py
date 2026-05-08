from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from app.tools.retriever import retrieve_chunks, is_grounded
from app.tools.synthesiser import synthesise_answer
from app.models.schemas import SourceChunk


# ── State definition ──────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    chunks: List[SourceChunk]
    confidence: float
    grounded: bool
    answer: str
    error: Optional[str]


# ── Node 1: Retrieve ──────────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant chunks from FAISS vector store.
    Updates state with chunks and confidence score.
    """
    try:
        chunks, confidence = retrieve_chunks(state["question"])
        return {
            **state,
            "chunks": chunks,
            "confidence": confidence,
            "error": None
        }
    except RuntimeError as e:
        return {
            **state,
            "chunks": [],
            "confidence": 0.0,
            "error": str(e)
        }


# ── Node 2: Confidence Check ──────────────────────────────────
def confidence_node(state: AgentState) -> AgentState:
    """
    Check if retrieved context is strong enough to ground an answer.
    This is the hallucination prevention layer.
    """
    grounded = is_grounded(state["confidence"])
    return {
        **state,
        "grounded": grounded
    }


# ── Node 3: Synthesise ────────────────────────────────────────
def synthesise_node(state: AgentState) -> AgentState:
    """
    Generate answer using Claude.
    If not grounded, returns out-of-scope message.
    """
    answer = synthesise_answer(
        question=state["question"],
        chunks=state["chunks"],
        grounded=state["grounded"]
    )
    return {
        **state,
        "answer": answer
    }


# ── Error Node ────────────────────────────────────────────────
def error_node(state: AgentState) -> AgentState:
    """Handle errors gracefully."""
    return {
        **state,
        "answer": f"An error occurred: {state.get('error', 'Unknown error')}",
        "grounded": False
    }


# ── Routing Logic ─────────────────────────────────────────────
def route_after_retrieval(state: AgentState) -> str:
    """
    Route to error node if retrieval failed,
    otherwise proceed to confidence check.
    """
    if state.get("error"):
        return "error"
    return "confidence_check"


# ── Build LangGraph ───────────────────────────────────────────
def build_agent():
    """
    Construct the LangGraph state machine.

    Flow:
    retrieve → [error | confidence_check] → synthesise → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("confidence_check", confidence_node)
    graph.add_node("synthesise", synthesise_node)
    graph.add_node("error", error_node)

    # Entry point
    graph.set_entry_point("retrieve")

    # Conditional routing after retrieval
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "error": "error",
            "confidence_check": "confidence_check"
        }
    )

    # Linear edges
    graph.add_edge("confidence_check", "synthesise")
    graph.add_edge("synthesise", END)
    graph.add_edge("error", END)

    return graph.compile()


# ── Compiled agent (singleton) ────────────────────────────────
agent = build_agent()


# ── Public interface ──────────────────────────────────────────
def run_agent(question: str) -> AgentState:
    """
    Run the full agent pipeline for a given question.
    Returns the final state with answer and metadata.
    """
    initial_state: AgentState = {
        "question": question,
        "chunks": [],
        "confidence": 0.0,
        "grounded": False,
        "answer": "",
        "error": None
    }
    result = agent.invoke(initial_state)
    return result