"""Shared configuration for the RAG pipeline and knowledge graph. """

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / ".cache"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # local, no API key required
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40
TOP_K_CHUNKS = 8      # chunks retrieved from the vector store
TOP_K_POSTINGS = 5    # distinct postings kept as evidence/citations after de-duplication
ANTHROPIC_MODEL = "claude-haiku-4-5"
