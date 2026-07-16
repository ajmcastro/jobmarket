"""Part 2 — Knowledge Graph endpoints."""

from fastapi import APIRouter, HTTPException, Query, Request

from api.schemas import BridgeResponse, GraphStats, RoleInfo, SkillFrequency
from jobmarket.kg.build import graph_stats
from jobmarket.kg.queries import companies_bridging_roles, list_roles, skills_for_role

router = APIRouter(prefix="/kg", tags=["knowledge-graph"])


@router.get("/stats", response_model=GraphStats, summary="Node/edge counts by type")
def stats(request: Request) -> GraphStats:
    return GraphStats(**graph_stats(request.app.state.kg))


@router.get("/roles", response_model=list[RoleInfo], summary="Canonical roles and their posting counts")
def roles(request: Request) -> list[RoleInfo]:
    return [RoleInfo(**r) for r in list_roles(request.app.state.kg)]


@router.get(
    "/roles/{role}/skills",
    response_model=list[SkillFrequency],
    summary="Top skills mentioned in postings for a role",
)
def role_skills(role: str, request: Request, top: int = Query(default=15, ge=1, le=50)) -> list[SkillFrequency]:
    kg = request.app.state.kg
    known_roles = {r["role"] for r in list_roles(kg)}
    if role not in known_roles:
        raise HTTPException(status_code=404, detail=f"Unknown role '{role}'. See GET /kg/roles for valid values.")
    counts = skills_for_role(kg, role)
    return [SkillFrequency(skill=skill, count=count) for skill, count in counts.most_common(top)]


@router.get(
    "/bridge",
    response_model=BridgeResponse,
    summary="Companies hiring for both of two roles, and the skills connecting them",
    description=(
        "The required knowledge-graph demo query, e.g. role_a='Data Engineer', "
        "role_b='Machine Learning Engineer'. Answered by traversal + set intersection over the graph."
    ),
)
def bridge(
    request: Request,
    role_a: str = Query(..., description="e.g. 'Data Engineer'"),
    role_b: str = Query(..., description="e.g. 'Machine Learning Engineer'"),
    top_skills: int = Query(default=10, ge=1, le=50),
) -> BridgeResponse:
    kg = request.app.state.kg
    known_roles = {r["role"] for r in list_roles(kg)}
    for role in (role_a, role_b):
        if role not in known_roles:
            raise HTTPException(status_code=404, detail=f"Unknown role '{role}'. See GET /kg/roles for valid values.")
    result = companies_bridging_roles(kg, role_a, role_b, top_skills=top_skills)
    return BridgeResponse(**result)
