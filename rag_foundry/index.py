"""In-memory BM25 + hashed-vector index with JSON persistence."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .chunk import Chunk, chunk_text
from .config import RagConfig
from .ingest import load_documents

_TOKEN_RE = re.compile(r"\w+")
_INDEX_VERSION = 1


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _tokenize_ngrams(text: str) -> List[str]:
    """Unigrams plus character-joined bigrams for feature hashing."""
    tokens = _tokenize(text)
    bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
    return tokens + bigrams


def _hash_feature(feature: str, dims: int) -> Tuple[int, int]:
    """Deterministic feature-hashing bucket and sign from md5."""
    digest = __import__("hashlib").md5(feature.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % dims
    sign = 1 if digest[8] % 2 == 0 else -1
    return bucket, sign


def _vectorize(text: str, dims: int) -> Dict[int, float]:
    counts: Dict[int, float] = {}
    for feature in _tokenize_ngrams(text):
        bucket, sign = _hash_feature(feature, dims)
        counts[bucket] = counts.get(bucket, 0.0) + sign
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {bucket: value / norm for bucket, value in counts.items()}


def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b[bucket] for bucket, value in a.items() if bucket in b)


@dataclass
class InMemoryIndex:
    """BM25 inverted index plus sparse hashed vectors, all in memory."""

    chunks: List[Chunk] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75
    dims: int = 4096
    doc_lens: List[int] = field(default_factory=list)
    postings: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    idf: Dict[str, float] = field(default_factory=dict)
    vectors: List[Dict[int, float]] = field(default_factory=list)
    avgdl: float = 0.0

    def add(self, chunk: Chunk) -> None:
        """Append a chunk and update per-chunk token statistics."""
        tokens = _tokenize(chunk.text)
        idx = len(self.chunks)
        self.chunks.append(chunk)
        self.doc_lens.append(len(tokens))
        for term in set(tokens):
            self.postings.setdefault(term, []).append((idx, tokens.count(term)))
        self.vectors.append(_vectorize(chunk.text, self.dims))

    def build(self) -> None:
        """Recompute avgdl and BM25 idf from current postings."""
        total = sum(self.doc_lens)
        self.avgdl = (total / len(self.doc_lens)) if self.doc_lens else 0.0
        n = len(self.chunks)
        self.idf = {
            term: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Rank chunks by Okapi BM25 against ``query``."""
        if not self.chunks:
            return []
        tokens = _tokenize(query)
        scores: Dict[int, float] = {}
        for term in set(tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in self.postings[term]:
                if tf <= 0:
                    continue
                dl = self.doc_lens[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.chunks[item[0]].id))
        return [(self.chunks[idx], score) for idx, score in ranked[:top_k]]

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Rank chunks by cosine similarity of hashed vectors."""
        if not self.chunks:
            return []
        query_vec = _vectorize(query, self.dims)
        scored = [
            (chunk, _cosine(query_vec, vec))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored = [item for item in scored if item[1] > 0.0]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:top_k]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Return a previously indexed chunk by its identifier.

        Args:
            chunk_id: Identifier recorded when the chunk was indexed.

        Returns:
            The chunk, or ``None`` when the identifier is unknown.
        """
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def save(self, path: str) -> None:
        """Write index chunks and config to ``path`` atomically via a tmp file."""
        payload = {
            "version": _INDEX_VERSION,
            "config": {"k1": self.k1, "b": self.b, "dims": self.dims},
            "chunks": [
                {
                    "id": chunk.id,
                    "doc_source": chunk.doc_source,
                    "index": chunk.index,
                    "text": chunk.text,
                }
                for chunk in self.chunks
            ],
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> "InMemoryIndex":
        """Load a previously saved index, rebuilding statistics."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != _INDEX_VERSION:
            raise ValueError(f"unsupported index version: {payload.get('version')!r}")
        cfg = payload.get("config", {})
        index = cls(k1=cfg.get("k1", 1.5), b=cfg.get("b", 0.75), dims=cfg.get("dims", 4096))
        for record in payload.get("chunks", []):
            index.add(
                Chunk(
                    id=record["id"],
                    doc_source=record["doc_source"],
                    text=record["text"],
                    index=record["index"],
                )
            )
        index.build()
        return index


def build_index_from_paths(paths: Iterable[str], config: RagConfig) -> InMemoryIndex:
    """Ingest documents, chunk them, and build a fresh index."""
    index = InMemoryIndex(k1=config.bm25_k1, b=config.bm25_b, dims=config.vector_dims)
    for doc_source, text in load_documents(paths):
        for chunk in chunk_text(
            text,
            doc_source,
            strategy=config.chunk_strategy,
            size=config.chunk_size,
            overlap=config.chunk_overlap,
        ):
            index.add(chunk)
    index.build()
    return index
