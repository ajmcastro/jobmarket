"""Traversal helpers over the knowledge graph, composed to answer questions like
'which companies hire for both role A and role B, and what skills connect them'.
Traversal here is “manual” graph edge traversal / neighbor lookup, not a BFS or DFS search.
"""

from collections import Counter

import networkx as nx


def postings_for_role(kg: nx.DiGraph, role: str) -> list[str]:
    role_key = f"role:{role}"
    if not kg.has_node(role_key):
        return []
    return [u for u, _, d in kg.in_edges(role_key, data=True) if d["type"] == "HAS_ROLE"]


def company_for_posting(kg: nx.DiGraph, posting_node: str) -> str | None:
    for _, v, d in kg.out_edges(posting_node, data=True):
        if d["type"] == "POSTED_BY":
            return v
    return None


def skills_for_posting(kg: nx.DiGraph, posting_node: str) -> set[str]:
    return {kg.nodes[v]["name"] for _, v, d in kg.out_edges(posting_node, data=True) if d["type"] == "MENTIONS_SKILL"}


def postings_by_company_for_role(kg: nx.DiGraph, role: str) -> dict[str, list[str]]:
    """Map company node -> postings with this role at that company."""
    result: dict[str, list[str]] = {}
    for posting_node in postings_for_role(kg, role):
        company_node = company_for_posting(kg, posting_node)
        if company_node:
            result.setdefault(company_node, []).append(posting_node)
    return result


def skills_for_role(kg: nx.DiGraph, role: str) -> Counter:
    counts: Counter = Counter()
    for posting_node in postings_for_role(kg, role):
        counts.update(skills_for_posting(kg, posting_node))
    return counts


def list_roles(kg: nx.DiGraph) -> list[dict]:
    roles = []
    for node, data in kg.nodes(data=True):
        if data.get("kind") == "Role":
            roles.append({"role": data["name"], "num_postings": len(postings_for_role(kg, data["name"]))})
    return sorted(roles, key=lambda r: r["num_postings"], reverse=True)


def companies_bridging_roles(kg: nx.DiGraph, role_a: str, role_b: str, top_skills: int = 10) -> dict:
    by_company_a = postings_by_company_for_role(kg, role_a)
    by_company_b = postings_by_company_for_role(kg, role_b)
    bridging_companies = sorted(set(by_company_a) & set(by_company_b))

    skills_a_overall = skills_for_role(kg, role_a)
    skills_b_overall = skills_for_role(kg, role_b)
    shared_overall = sorted(
        set(skills_a_overall) & set(skills_b_overall),
        key=lambda s: skills_a_overall[s] + skills_b_overall[s],
        reverse=True,
    )

    per_company = []
    for company_node in bridging_companies:
        skills_a = Counter()
        for p in by_company_a[company_node]:
            skills_a.update(skills_for_posting(kg, p))
        skills_b = Counter()
        for p in by_company_b[company_node]:
            skills_b.update(skills_for_posting(kg, p))
        per_company.append({
            "company": kg.nodes[company_node]["name"],
            "postings_a": [kg.nodes[p]["job_id"] for p in by_company_a[company_node]],
            "postings_b": [kg.nodes[p]["job_id"] for p in by_company_b[company_node]],
            "shared_skills": sorted(set(skills_a) & set(skills_b)),
        })

    return {
        "role_a": role_a,
        "role_b": role_b,
        "num_companies_role_a": len(by_company_a),
        "num_companies_role_b": len(by_company_b),
        "bridging_companies": per_company,
        "shared_skills_overall": shared_overall[:top_skills],
    }
