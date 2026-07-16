"""Part 1 — Baseline RAG endpoints."""

from fastapi import APIRouter, Request

from api.schemas import AskRequest, AskResponse, BackendInfo

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a natural-language question over job postings",
    description=(
        "Retrieves relevant posting chunks by embedding similarity and generates a grounded "
        "answer with citations (job_id, title, company, location, excerpt). Falls back to a "
        "deterministic mock answer if no Anthropic credentials are configured."
    ),
)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    rag_index = request.app.state.rag_index
    result = rag_index.answer_question(
        payload.question,
        backend=payload.backend,
        k_chunks=payload.k_chunks,
        k_postings=payload.k_postings,
    )
    return AskResponse(**result)


@router.get(
    "/backends",
    response_model=list[BackendInfo],
    summary="List available vector-search backends and index stats",
)
def backends(request: Request) -> list[BackendInfo]:
    rag_index = request.app.state.rag_index
    dim = int(rag_index.embeddings.shape[1])
    return [
        BackendInfo(name=name, num_chunks=len(rag_index.corpus), embedding_dim=dim)
        for name in rag_index.vector_stores
    ]
