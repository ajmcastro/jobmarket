"""Builds (or loads a cached copy of) the knowledge graph.

Building is fast relative to the RAG embedding step, but still non-trivial at 4K postings, so it
gets the same disk-cache treatment: pickle the finished nx.DiGraph under CACHE_DIR and reload it
on the next build() call. Delete CACHE_DIR to force a rebuild.
"""

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from jobmarket.config import CACHE_DIR
from jobmarket.kg.build import build_graph

GRAPH_CACHE_FILE = "kg_graph.pkl"


def build_or_load_graph(
    postings: pd.DataFrame,
    companies: pd.DataFrame | None,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
) -> nx.DiGraph:
    graph_path = cache_dir / GRAPH_CACHE_FILE

    if use_cache and graph_path.exists():
        with graph_path.open("rb") as f:
            return pickle.load(f)

    kg = build_graph(postings, companies)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with graph_path.open("wb") as f:
            pickle.dump(kg, f)

    return kg
