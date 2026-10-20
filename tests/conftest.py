"""Shared fixtures and sample corpus for the rag-foundry test suite.

Every test runs offline: no network calls, no API key, no external services.
"""

from __future__ import annotations

import pytest

from rag_foundry.chunk import Chunk
from rag_foundry.config import RagConfig
from rag_foundry.index import InMemoryIndex

SAMPLE_DOCS = {
    "reliability.md": (
        "# Retry policy\n\n"
        "The service retries failed requests three times using exponential "
        "backoff with jitter. The base delay is 200 milliseconds and the "
        "maximum delay is 20 seconds. A circuit breaker opens after five "
        "consecutive failures and closes again after a 30 second cooldown.\n\n"
        "Idempotent requests are retried automatically. Non-idempotent "
        "requests require an explicit retry token supplied by the caller."
    ),
    "deployment.md": (
        "# Deployment\n\n"
        "The service is deployed to Kubernetes on three availability zones. "
        "Rolling updates use a canary release that shifts ten percent of "
        "traffic for ten minutes before completing. Rollback is automatic "
        "when the error rate exceeds one percent.\n\n"
        "Secrets are injected at runtime from a managed vault. No credentials "
        "are stored in the container image."
    ),
    "monitoring.md": (
        "# Monitoring\n\n"
        "Prometheus scrapes every service every fifteen seconds. Alerts fire "
        "on sustained error budget burn rather than on point-in-time error "
        "spikes. Distributed traces propagate W3C trace context across every "
        "hop, which is what makes latency attribution possible."
    ),
}


@pytest.fixture
def corpus(tmp_path):
    """Write the sample corpus to disk and return its directory path."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for name, text in SAMPLE_DOCS.items():
        (docs_dir / name).write_text(text, encoding="utf-8")
    return str(docs_dir)


@pytest.fixture
def config():
    """Return a test config tuned for deterministic offline behaviour."""
    return RagConfig(
        chunk_strategy="sentence",
        chunk_size=400,
        chunk_overlap=60,
        top_k=3,
        rrf_k=60,
        min_score=0.0,
        index_path="ragf_index.json",
    )


@pytest.fixture
def built_index(corpus, config):
    """Ingest the sample corpus and return a populated index."""
    from rag_foundry.index import build_index_from_paths

    return build_index_from_paths([corpus], config)


@pytest.fixture
def tiny_index():
    """Return a small hand-built index with three distinctive chunks."""
    index = InMemoryIndex()
    index.add(Chunk(id="c-0", doc_source="a.md",
                    text="Kubernetes deploys the service across three zones.",
                    index=0))
    index.add(Chunk(id="c-1", doc_source="b.md",
                    text="Exponential backoff retries a request three times.",
                    index=0))
    index.add(Chunk(id="c-2", doc_source="c.md",
                    text="Prometheus scrapes metrics every fifteen seconds.",
                    index=0))
    index.build()
    return index