"""
Vector-search backends sitting behind a common interface: exact FAISS and quantized turbovec.

A framework like LangChain provides several vector store integration options.
URL: https://docs.langchain.com/oss/python/integrations/vectorstores
"""

import numpy as np
import faiss
from turbovec import IdMapIndex


class VectorStore:
    """Common interface over the FAISS and turbovec backends."""

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        raise NotImplementedError


class FaissVectorStore(VectorStore):
    def __init__(self, vectors: np.ndarray):
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int):
        scores, indices = self.index.search(query_vector[None, :], k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))


class TurboVecVectorStore(VectorStore):
    def __init__(self, vectors: np.ndarray, bit_width: int = 4):
        self.index = IdMapIndex(dim=vectors.shape[1], bit_width=bit_width)
        self.index.add_with_ids(vectors, np.arange(vectors.shape[0], dtype=np.uint64))

    def search(self, query_vector: np.ndarray, k: int):
        scores, ids = self.index.search(query_vector[None, :], k)
        return list(zip(ids[0].tolist(), scores[0].tolist()))


def build_vector_stores(embeddings: np.ndarray) -> dict[str, VectorStore]:
    return {
        "faiss": FaissVectorStore(embeddings),
        "turbovec": TurboVecVectorStore(embeddings),
    }
