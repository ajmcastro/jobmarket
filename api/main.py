"""FastAPI app exposing Part 1 (Baseline RAG) and Part 2 (Knowledge Graph).

Building the RAG index and the knowledge graph is expensive (a few minutes on a laptop CPU for
the full corpus), so both are built once at process startup and reused for every request. See
jobmarket/rag/pipeline.py and jobmarket/kg/pipeline.py for the on-disk cache that makes restarts
fast after the first run — delete .cache/ to force a rebuild (e.g. after changing the source CSVs).

Run with: uv run uvicorn api.main:app --reload
Docs at:  http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.routers import kg as kg_router
from api.routers import rag as rag_router
from api.schemas import HealthResponse
from jobmarket.data import load_companies, load_postings
from jobmarket.kg.pipeline import build_or_load_graph
from jobmarket.rag.pipeline import RagIndex


@asynccontextmanager
async def lifespan(app: FastAPI):
    postings = load_postings()
    companies = load_companies()

    app.state.rag_index = RagIndex.build(postings)
    app.state.kg = build_or_load_graph(postings, companies)

    yield

    app.state.rag_index = None
    app.state.kg = None


app = FastAPI(
    title="Job-Market Intelligence Assistant API",
    description=(
        "HTTP API over the take-home task's Baseline RAG (Part 1) and Knowledge Graph (Part 2) "
        "implementations, built from a sample of LinkedIn job postings. See the notebook "
        "(mleng_take_home_task.ipynb) and README for the full design write-up; both share the "
        "same underlying jobmarket/ package this API calls into."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(rag_router.router)
app.include_router(kg_router.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["health"], summary="Liveness/readiness check")
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        rag_ready=getattr(app.state, "rag_index", None) is not None,
        kg_ready=getattr(app.state, "kg", None) is not None,
    )
