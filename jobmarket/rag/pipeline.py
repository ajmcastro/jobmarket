"""Ties chunking + embedding + vector stores + retrieval + generation into one object.

Building the index (embedding ~16K chunks) takes a few minutes on a laptop CPU, so this supports
a simple disk cache: the corpus and embeddings are pickled/saved under CACHE_DIR, and reloaded on
the next build() call instead of being recomputed. The embedder and vector-store objects
themselves are cheap to rebuild from cached embeddings (no re-encoding), so only the corpus
DataFrame and the embedding matrix need to survive on disk. Delete CACHE_DIR to force a rebuild
(e.g. after changing chunk size, the embedding model, or the source CSVs).
"""

from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from jobmarket.config import (
    ANTHROPIC_MODEL,
    CACHE_DIR,
    EMBEDDING_MODEL_NAME,
    TOP_K_CHUNKS,
    TOP_K_POSTINGS,
)
from jobmarket.rag.chunking import build_corpus
from jobmarket.rag.generation import generate_answer
from jobmarket.rag.retrieval import retrieve
from jobmarket.rag.vector_store import VectorStore, build_vector_stores

CORPUS_CACHE_FILE = "rag_corpus.pkl"
EMBEDDINGS_CACHE_FILE = "rag_embeddings.npy"


@dataclass
class RagIndex:
    corpus: pd.DataFrame
    embeddings: np.ndarray
    embedder: SentenceTransformer
    vector_stores: dict[str, VectorStore]
    anthropic_client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)

    @classmethod
    def build(
        cls,
        postings: pd.DataFrame,
        cache_dir: Path = CACHE_DIR,
        use_cache: bool = True,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> "RagIndex":
        embedder = SentenceTransformer(embedding_model_name)

        corpus_path = cache_dir / CORPUS_CACHE_FILE
        embeddings_path = cache_dir / EMBEDDINGS_CACHE_FILE

        if use_cache and corpus_path.exists() and embeddings_path.exists():
            corpus = pd.read_pickle(corpus_path)
            embeddings = np.load(embeddings_path)
        else:
            corpus = build_corpus(postings)
            embeddings = embedder.encode(
                corpus["embedding_text"].tolist(),
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            if use_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                corpus.to_pickle(corpus_path)
                np.save(embeddings_path, embeddings)

        vector_stores = build_vector_stores(embeddings)
        return cls(corpus=corpus, embeddings=embeddings, embedder=embedder, vector_stores=vector_stores)

    def retrieve(
        self,
        query: str,
        k_chunks: int = TOP_K_CHUNKS,
        k_postings: int = TOP_K_POSTINGS,
        backend: str = "faiss",
    ) -> list[dict]:
        return retrieve(
            query,
            corpus=self.corpus,
            vector_stores=self.vector_stores,
            embedder=self.embedder,
            k_chunks=k_chunks,
            k_postings=k_postings,
            backend=backend,
        )

    def answer_question(
        self,
        query: str,
        backend: str = "faiss",
        k_chunks: int = TOP_K_CHUNKS,
        k_postings: int = TOP_K_POSTINGS,
        model: str = ANTHROPIC_MODEL,
        verbose: bool = False,
    ) -> dict:
        evidence = self.retrieve(query, k_chunks=k_chunks, k_postings=k_postings, backend=backend)
        answer = generate_answer(query, evidence, anthropic_client=self.anthropic_client, model=model)

        if verbose:
            print(f"Q: {query}\n(backend={backend})\n")
            print(answer)
            print("\nCitations:")
            for i, e in enumerate(evidence, start=1):
                print(f"  [{i}] job_id={e['job_id']} | {e['title']} | {e['company']} | {e['location']} | score={e['score']:.3f}")
                print(f"      \"{e['excerpt'][:160].strip()}...\"")

        return {"query": query, "answer": answer, "citations": evidence, "backend": backend}
