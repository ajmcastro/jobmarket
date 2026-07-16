"""Pydantic request/response models — these drive the Swagger schema shown at /docs."""

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    job_id: int
    title: str
    company: str
    location: str
    excerpt: str
    score: float = Field(description="Cosine similarity of the best-matching chunk, in [-1, 1]")


class AskRequest(BaseModel):
    question: str = Field(
        examples=["Which postings mention retrieval-augmented generation, vector databases, or LLM application development?"],
    )
    backend: Literal["faiss", "turbovec"] = Field(
        default="faiss", description="Vector-search backend: exact (faiss) or 4-bit quantized (turbovec)."
    )
    k_chunks: int = Field(default=8, ge=1, le=50, description="Chunks retrieved from the vector store before de-duplication.")
    k_postings: int = Field(default=5, ge=1, le=20, description="Distinct postings kept as evidence/citations after de-duplication.")


class AskResponse(BaseModel):
    query: str
    answer: str = Field(description="LLM-generated answer, or a '[Mock mode]' extractive answer if no Anthropic credentials are configured.")
    backend: str
    citations: list[Citation]


class BackendInfo(BaseModel):
    name: str
    num_chunks: int
    embedding_dim: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    rag_ready: bool
    kg_ready: bool


class GraphStats(BaseModel):
    num_nodes: int
    num_edges: int
    nodes_by_kind: dict[str, int]
    edges_by_type: dict[str, int]


class RoleInfo(BaseModel):
    role: str
    num_postings: int


class SkillFrequency(BaseModel):
    skill: str
    count: int


class BridgingCompany(BaseModel):
    company: str
    postings_a: list[int]
    postings_b: list[int]
    shared_skills: list[str]


class BridgeResponse(BaseModel):
    role_a: str
    role_b: str
    num_companies_role_a: int
    num_companies_role_b: int
    bridging_companies: list[BridgingCompany]
    shared_skills_overall: list[str]
