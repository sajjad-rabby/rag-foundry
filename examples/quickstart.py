"""End-to-end quickstart: ingest, retrieve, answer, and evaluate.

Runs fully offline - no API key and no network calls. The sample corpus is
generated in a temporary directory so the script is self-contained.

Usage::

    python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_foundry import RagConfig, answer_question, build_index_from_paths  # noqa: E402
from rag_foundry.eval import run_eval, summary_to_markdown  # noqa: E402
from rag_foundry.retrieve import hybrid_search  # noqa: E402

DOCS = {
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

QUESTIONS = [
    "How many times does the service retry a failed request?",
    "Where are secrets stored?",
    "What does Prometheus do every fifteen seconds?",
]

QA_SET = [
    {"question": "How many times does the service retry a failed request?",
     "reference_answer": "Three times with exponential backoff and jitter.",
     "expected_keywords": ["retries", "backoff"]},
    {"question": "Where are secrets stored?",
     "reference_answer": "Injected at runtime from a managed vault.",
     "expected_keywords": ["vault", "injected"]},
    {"question": "How often does Prometheus scrape the services?",
     "reference_answer": "Every fifteen seconds.",
     "expected_keywords": ["fifteen", "scrape"]},
]


def write_corpus(directory: str) -> str:
    """Write the sample corpus and return the directory path."""
    os.makedirs(directory, exist_ok=True)
    for name, text in DOCS.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    return directory


def main() -> int:
    """Run the quickstart walkthrough."""
    config = RagConfig(chunk_strategy="sentence", chunk_size=400,
                       chunk_overlap=60, top_k=3, min_score=0.0)

    with tempfile.TemporaryDirectory() as workdir:
        docs_dir = write_corpus(os.path.join(workdir, "docs"))
        index_path = os.path.join(workdir, "ragf_index.json")

        print("=" * 68)
        print("STEP 1 - INGEST")
        print("=" * 68)
        index = build_index_from_paths([docs_dir], config)
        index.save(index_path)
        print(f"  documents indexed : {len(DOCS)}")
        print(f"  chunks produced   : {len(index.chunks)}")
        print(f"  average chunk len : {index.avgdl:.1f} tokens")
        print(f"  vocabulary size   : {len(index.postings)} terms")
        print(f"  index saved to    : {index_path}")

        print()
        print("=" * 68)
        print("STEP 2 - HYBRID RETRIEVAL (BM25 + vectors fused by RRF)")
        print("=" * 68)
        print("  Note: retrieval is lexical, not semantic. The tokenizer does")
        print("  not stem, so a query term must appear in the document in some")
        print("  shared form ('retry' will not match 'retries').")
        query = "retry backoff three times"
        print(f"  query: {query!r}")
        for chunk, score in hybrid_search(index, query, top_k=3, rrf_k=config.rrf_k):
            source = os.path.basename(chunk.doc_source)
            preview = chunk.text.replace("\n", " ")[:64]
            print(f"  [{score:.5f}] {source:<16} {preview}...")

        print()
        print("=" * 68)
        print("STEP 3 - GROUNDED ANSWERS (offline extractive mode)")
        print("=" * 68)
        for question in QUESTIONS:
            result = answer_question(question, index, config, use_llm=False)
            print(f"\n  Q: {question}")
            if hasattr(result, "reason"):
                print(f"  A: [refused: {result.reason}] {result.message}")
                continue
            print(f"  A: {result.text[:150]}")
            for i, chunk in enumerate(result.sources, start=1):
                print(f"     [{i}] {os.path.basename(chunk.doc_source)}")

        print()
        print("=" * 68)
        print("STEP 4 - REFUSAL WHEN RETRIEVAL FINDS NOTHING")
        print("=" * 68)
        strict = RagConfig(top_k=3, min_score=50.0)
        refusal = answer_question("What is the airspeed velocity of a swallow?",
                                  index, strict, use_llm=False)
        if hasattr(refusal, "reason"):
            print(f"  refused with reason : {refusal.reason}")
            print(f"  message             : {refusal.message}")
        else:
            print("  (threshold was low enough to answer)")

        print()
        print("=" * 68)
        print("STEP 5 - EVALUATION (keyword metrics, no API key needed)")
        print("=" * 68)
        import rag_foundry.eval as eval_module
        import rag_foundry.generate as generate_module

        # Force offline answering so the eval run needs no API key.
        original = eval_module.answer_question
        eval_module.answer_question = (
            lambda q, i, c: generate_module.answer_question(q, i, c, use_llm=False))
        try:
            report = run_eval(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa.json"),
                index,
                RagConfig(min_score=0.0, top_k=3),
                out_path=os.path.join(workdir, "eval_report.json"),
                use_judge=False,
            )
        finally:
            eval_module.answer_question = original

        print(summary_to_markdown(report))
        for item in report["items"]:
            status = "ok " if item["keyword_hits"] else "miss"
            print(f"  [{status}] {item['question']}")
            print(f"          hits  : {item['keyword_hits']}")
            print(f"          misses: {item['keyword_misses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())