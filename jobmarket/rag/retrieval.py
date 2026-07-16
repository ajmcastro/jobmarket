"""Query embedding + vector search + de-duplication by posting."""

import pandas as pd
from sentence_transformers import SentenceTransformer

from jobmarket.config import TOP_K_CHUNKS, TOP_K_POSTINGS
from jobmarket.rag.vector_store import VectorStore


def retrieve(
    query: str,
    corpus: pd.DataFrame,
    vector_stores: dict[str, VectorStore],
    embedder: SentenceTransformer,
    k_chunks: int = TOP_K_CHUNKS,
    k_postings: int = TOP_K_POSTINGS,
    backend: str = "faiss",
) -> list[dict]:
    store = vector_stores[backend]
    query_vector = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
    hits = store.search(query_vector, k_chunks)

    best_per_posting: dict = {}
    for chunk_idx, score in hits:
        row = corpus.iloc[chunk_idx]
        job_id = row["job_id"]
        if job_id not in best_per_posting or score > best_per_posting[job_id]["score"]:
            best_per_posting[job_id] = {
                "job_id": job_id,
                "title": row["title"],
                "company": row["company"],
                "location": row["location"],
                "excerpt": row["text"],
                "score": float(score),
            }

    ranked = sorted(best_per_posting.values(), key=lambda e: e["score"], reverse=True)
    return ranked[:k_postings]
