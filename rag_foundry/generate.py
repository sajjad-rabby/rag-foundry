"""Answer generation with citations, plus an offline extractive fallback.

Without an API key

This module is the only part of the pipeline that talks to a remote
service. It is deliberately small: build a grounded prompt, call an
OpenAI-compatible chat completions endpoint with the standard library,
and keep a deterministic extractive fallback so the CLI stays useful
without an API key.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_foundry.config import RagConfig
from rag_foundry.index import Chunk

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer ONLY from the numbered "
    "context blocks provided by the user. Cite the block number for every "
    "claim using the [n] notation. If the context is insufficient, reply "
    "with the exact sentence: I don't have enough information in the "
    "provided sources. Never invent facts, names, numbers, or APIs."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    """A generated answer with its full prompt and the cited chunks."""

    text: str
    prompt: Tuple[str, str] = ("", "")
    sources: List[Chunk] = field(default_factory=list)


@dataclass
class Refusal:
    """A structured refusal produced before any LLM call."""

    reason: str
    message: str
    top_score: Optional[float] = None


class LLMError(RuntimeError):
    """Raised when the LLM HTTP call or response parsing fails."""


def build_prompt(question: str, chunks: List[Chunk]) -> Tuple[str, str]:
    """Build the prompt asking for a cited answer.

Each chunk is numbered so the model can

    Each chunk becomes a numbered block prefixed with its document source
    so the model can both cite it and trace it back to a file.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({chunk.doc_source})\n{chunk.text}")
    context = "\n\n".join(blocks)
    user = (
        "Context:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above and cite the "
        "block numbers you relied on with [n]."
    )
    return SYSTEM_PROMPT, user


def call_llm(system: str, user: str, config: RagConfig) -> str:
    """Call an OpenAI-compatible chat-completions endpoint.

Posts to

    Uses :mod:`urllib` only. Raises :class:`LLMError` for a missing API key,
    HTTP failures, or malformed responses. The endpoint is
    ``{llm_base_url}/v1/chat/completions``.
    """
    api_key = config.api_key()
    if not api_key:
        raise LLMError(
            f"no API key found: set the {config.api_key_env} environment variable"
        )
    url = f"{config.llm_base_url.rstrip('/')}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": config.llm_model,
        "temperature": config.llm_temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=config.llm_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise LLMError(f"LLM HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM connection error: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected LLM response shape: {data!r}") from exc


def _sanitize_citations(text: str, max_n: int) -> str:
    """Strip citation markers outside the valid chunk range.

Returns:

    Citations must reference a block number between 1 and ``max_n``
    inclusive. When ``max_n`` is zero or negative every citation is
    removed.
    """
    if max_n <= 0:
        return _CITATION_RE.sub("", text)

    def _keep(match: "re.Match[str]") -> str:
        n = int(match.group(1))
        if 1 <= n <= max_n:
            return match.group(0)
        return ""

    return _CITATION_RE.sub(_keep, text)


def _extractive_fallback(question: str, chunks: List[Chunk]) -> str:
    """Answer from the retrieved chunks alone, without calling a model.

Used when no API key is configured so the

    Scores sentences of the highest-ranked chunk by overlap with the
    question terms, then returns the top two sentences as a grounded
    extract with a single [1] citation.
    """
    if not chunks:
        return ""
    top = chunks[0]
    q_terms = set(re.findall(r"\w+", question.lower()))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", top.text) if s.strip()]
    if not sentences:
        sentences = [top.text.strip()]
    scored = []
    for s in sentences:
        terms = set(re.findall(r"\w+", s.lower()))
        scored.append((len(terms & q_terms), len(s), s))
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked = [s for _, _, s in scored[:2]]
    return " ".join(picked) + " [1]"


def answer_question(
    question: str,
    index: Any,
    config: RagConfig,
    use_llm: bool = True,
) -> Any:
    """Answer a question using an LLM, or the extractive fallback.

Behaviour:
    - empty index -> ``Refusal("no_chunks")``;
    - nothing matched at all -> ``Refusal("no_results")``;
    - best hybrid score below ``config.min_score`` -> ``Refusal("low_score")``;
    - ``use_llm=False`` -> extractive answer built locally (offline mode);
    - otherwise -> LLM answer with out-of-range citations stripped.

Returns:
    The answer text. With an LLM available the answer is model-generated and carries citations.
    """
    from rag_foundry.retrieve import hybrid_search

    results = hybrid_search(
        index, question, top_k=config.top_k, rrf_k=config.rrf_k
    )
    if not results:
        # Distinguish an empty corpus from a query that simply matched
        # nothing: the operator fix is different in each case.
        indexed = len(getattr(index, "chunks", []) or [])
        if indexed == 0:
            return Refusal(
                reason="no_chunks",
                message="The index is empty. Ingest documents first with: ragf ingest <paths>",
            )
        return Refusal(
            reason="no_results",
            message=(
                f"None of the {indexed} indexed chunks matched this question. "
                "Try different wording, or lower min_score if the corpus is "
                "known to cover the topic."
            ),
            top_score=0.0,
        )
    top_score = results[0][1]
    if top_score < config.min_score:
        return Refusal(
            reason="low_score",
            message=(
                "No indexed document is close enough to this question "
                f"(best score {top_score:.4f} < min_score {config.min_score:.4f})."
            ),
            top_score=top_score,
        )
    chunks = [chunk for chunk, _ in results]
    system, user = build_prompt(question, chunks)
    if not use_llm:
        return Answer(
            text=_extractive_fallback(question, chunks),
            prompt=(system, user),
            sources=chunks,
        )
    raw = call_llm(system, user, config)
    return Answer(
        text=_sanitize_citations(raw, len(chunks)),
        prompt=(system, user),
        sources=chunks,
    )
