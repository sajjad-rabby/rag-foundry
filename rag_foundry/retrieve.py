"""Hybrid retrieval fusing BM25 and vector rankings with reciprocal rank fusion."""
from __future__ import annotations

from typing import List, Tuple

from .chunk import Chunk
from .index import InMemoryIndex

#: Number of independent rankings fused by :func:`hybrid_search`.
_N_SIGNALS = 2


def hybrid_search(
    index: InMemoryIndex,
    query: str,
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[Tuple[Chunk, float]]:
    """Search the BM25 and vector indexes, then fuse the rankings.

Returns:

    Each ranking contributes ``1 / (rrf_k + rank + 1)`` per chunk, where
    ``rank`` is 0-based. Scores are then normalised against the best score
    the fusion could produce for this configuration - a chunk ranked first
    by *both* signals scores ``1.0``, one ranked first by a single signal
    scores ``0.5``.

    Normalising against the theoretical maximum rather than the observed
    top hit is what keeps the scale stable: without it the achievable
    ceiling moves with ``rrf_k``, and a fixed ``min_score`` threshold
    silently becomes unreachable (or trivially easy) when that changes.

    Ties break by ascending chunk id for determinism. Returns up to
    ``top_k`` ``(Chunk, score)`` pairs, best first.
    """
    bm25 = index.bm25_search(query, top_k * 2)
    vector = index.vector_search(query, top_k * 2)

    scores: dict = {}
    for ranking in (bm25, vector):
        for rank, (chunk, _score) in enumerate(ranking):
            contribution = 1.0 / (rrf_k + rank + 1)
            entry = scores.setdefault(chunk.id, {"chunk": chunk, "score": 0.0})
            entry["score"] += contribution

    if not scores:
        return []

    ceiling = _N_SIGNALS / (rrf_k + 1.0)
    ranked = sorted(
        ((entry["chunk"], entry["score"] / ceiling)
         for entry in scores.values()),
        key=lambda item: (-item[1], item[0].id),
    )
    return ranked[:top_k]
