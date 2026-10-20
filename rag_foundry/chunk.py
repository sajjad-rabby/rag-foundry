"""Text chunking strategies for RAG ingestion."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Chunk:
    """A chunk of a document, identified by source and position."""

    id: str
    doc_source: str
    text: str
    index: int


def _chunk_id(doc_source: str, index: int) -> str:
    digest = hashlib.md5(doc_source.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{index}"


def _fixed_chunks(text: str, size: int, overlap: int) -> List[str]:
    """Window text into ``size``-char pieces stepping by ``size - overlap``."""
    if overlap >= size:
        overlap = max(0, size - 1)
    step = size - overlap
    pieces: List[str] = []
    start = 0
    while True:
        piece = text[start:start + size]
        if not piece:
            break
        pieces.append(piece)
        if start + size >= len(text):
            break
        start += step
    return pieces


def _hard_split(sentence: str, size: int, overlap: int) -> List[str]:
    """Split an over-long single sentence using the fixed-window logic."""
    return _fixed_chunks(sentence, size, overlap)


def _sentence_chunks(text: str, size: int, overlap: int) -> List[str]:
    """Pack sentences greedily up to ``size`` chars, carrying trailing overlap."""
    sentences: List[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(text):
        part = raw.strip()
        if part:
            sentences.append(part)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_hard_split(sentence, size, overlap))
            continue
        if current and current_len + 1 + len(sentence) > size:
            chunks.append(" ".join(current))
            if overlap > 0:
                tail: List[str] = []
                tail_len = 0
                for prev in reversed(current):
                    extra = len(prev) + (1 if tail else 0)
                    if tail_len + extra > overlap or tail_len + extra >= size:
                        break
                    tail.insert(0, prev)
                    tail_len += extra
                current, current_len = tail, tail_len
            else:
                current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + (1 if current_len > 0 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(
    text: str,
    doc_source: str,
    strategy: str = "fixed",
    size: int = 800,
    overlap: int = 120,
) -> List[Chunk]:
    """Split text into overlapping chunks.

Two modes are available: token-window and sentence-aware, where

    ``strategy`` is ``"fixed"`` (character windows) or ``"sentence"``
    (greedy sentence packing with character-based overlap). Over-long
    single sentences are hard-split in both modes.
    """
    if strategy == "fixed":
        pieces = _fixed_chunks(text, size, overlap)
    elif strategy == "sentence":
        pieces = _sentence_chunks(text, size, overlap)
    else:
        raise ValueError(f"unknown chunk strategy: {strategy!r}")

    return [
        Chunk(id=_chunk_id(doc_source, i), doc_source=doc_source, text=piece, index=i)
        for i, piece in enumerate(pieces)
    ]
