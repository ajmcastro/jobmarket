"""Builds the NetworkX knowledge graph from postings (+ companies, when joinable)."""

import re
from collections import Counter

import networkx as nx
import pandas as pd

from jobmarket.data import safe_str
from jobmarket.kg.taxonomy import classify_role, extract_skills


def normalize_company_key(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def build_graph(postings: pd.DataFrame, companies: pd.DataFrame | None) -> nx.DiGraph:
    companies_by_id = None
    if companies is not None and "company_id" in companies.columns:
        companies_by_id = companies.dropna(subset=["company_id"]).copy()
        companies_by_id["company_id"] = companies_by_id["company_id"].astype("int64")
        companies_by_id = companies_by_id.set_index("company_id")

    kg = nx.DiGraph()

    for row in postings.itertuples(index=False):
        posting_node = f"posting:{row.job_id}"
        title = safe_str(row.title) or "Unknown title"
        company_name = safe_str(row.company_name)
        location = safe_str(row.location)
        description = safe_str(row.description)
        skills_desc = safe_str(row.skills_desc)

        kg.add_node(posting_node, kind="JobPosting", job_id=row.job_id, title=title, company=company_name or None, location=location or None)

        # Skip the POSTED_BY edge entirely when company_name is missing (~1% of postings), rather
        # than bucketing them under one shared "Unknown company" node — that would falsely merge
        # distinct anonymous employers into a single node that could then look like a bridging company.
        if company_name:
            company_key = f"company:{normalize_company_key(company_name)}"
            if not kg.has_node(company_key):
                attrs = {"kind": "Company", "name": company_name}
                if companies_by_id is not None and pd.notna(row.company_id):
                    cid = int(row.company_id)
                    if cid in companies_by_id.index:
                        crow = companies_by_id.loc[cid]
                        attrs.update({
                            "city": safe_str(crow.get("city")) or None,
                            "state": safe_str(crow.get("state")) or None,
                            "country": safe_str(crow.get("country")) or None,
                            "company_size": crow.get("company_size") if pd.notna(crow.get("company_size")) else None,
                        })
                kg.add_node(company_key, **attrs)
            kg.add_edge(posting_node, company_key, type="POSTED_BY")

        role = classify_role(title)
        role_key = f"role:{role}"
        if not kg.has_node(role_key):
            kg.add_node(role_key, kind="Role", name=role)
        kg.add_edge(posting_node, role_key, type="HAS_ROLE")

        if location:
            location_key = f"location:{location.lower()}"
            if not kg.has_node(location_key):
                kg.add_node(location_key, kind="Location", name=location)
            kg.add_edge(posting_node, location_key, type="LOCATED_IN")

        text = " ".join([title, description, skills_desc])
        for skill in extract_skills(text):
            skill_key = f"skill:{skill}"
            if not kg.has_node(skill_key):
                kg.add_node(skill_key, kind="Skill", name=skill)
            kg.add_edge(posting_node, skill_key, type="MENTIONS_SKILL")

    return kg


def graph_stats(kg: nx.DiGraph) -> dict:
    kinds = Counter(nx.get_node_attributes(kg, "kind").values())
    edge_types = Counter(d["type"] for _, _, d in kg.edges(data=True))
    return {
        "num_nodes": kg.number_of_nodes(),
        "num_edges": kg.number_of_edges(),
        "nodes_by_kind": dict(kinds),
        "edges_by_type": dict(edge_types),
    }
