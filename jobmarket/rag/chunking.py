"""Splitting posting text into overlapping word chunks for embedding."""

import pandas as pd
import re

from jobmarket.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from jobmarket.data import safe_str


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of words."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def improved_split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks using sentence-aware boundaries."""
    if not text or not text.strip():
        return []

    sentences = [s.strip() for s in re.split(_SENTENCE_SPLIT_RE, text) if s.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    def flush_chunk():
        if current_words:
            chunks.append(" ".join(current_words))

    def tail_words(words: list[str]) -> list[str]:
        return words[-overlap:] if overlap > 0 else []

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) <= chunk_size:
            current_words.extend(sentence_words)
            continue

        # close current chunk, then preserve overlap
        flush_chunk()
        current_words = tail_words(current_words)

        if len(sentence_words) > chunk_size:
            # sentence itself is too long; split it with a word-window fallback
            start = 0
            while start < len(sentence_words):
                end = start + chunk_size
                chunks.append(" ".join(sentence_words[start:end]))
                if end >= len(sentence_words):
                    break
                start = end - overlap
            current_words = []
        else:
            current_words.extend(sentence_words)

    flush_chunk()
    return chunks

def build_corpus(
    postings: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> pd.DataFrame:
    """One row per chunk, with a back-reference to its parent posting for citations."""
    records = []
    for row in postings.itertuples(index=False):
        body = " ".join(t for t in (safe_str(row.description), safe_str(row.skills_desc)) if t)
        if not body:
            continue
        title = safe_str(row.title) or "Unknown title"
        company = safe_str(row.company_name) or "Unknown company"
        location = safe_str(row.location) or "Unknown location"
        for i, chunk in enumerate(split_into_chunks(body, chunk_size, overlap)):
            records.append({
                "chunk_id": f"{row.job_id}_{i}",
                "job_id": row.job_id,
                "title": title,
                "company": company,
                "location": location,
                "text": chunk,
                "embedding_text": f"Job title: {title}\nCompany: {company}\nLocation: {location}\n\n{chunk}",
            })
    return pd.DataFrame(records)
