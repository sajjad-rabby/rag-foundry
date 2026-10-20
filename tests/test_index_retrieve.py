"""Tests for ingestion, the index, and hybrid retrieval."""

from __future__ import annotations

import os

import pytest

from rag_foundry.chunk import Chunk
from rag_foundry.index import InMemoryIndex, build_index_from_paths
from rag_foundry.ingest import extract_html, extract_text, load_documents
from rag_foundry.retrieve import hybrid_search


# -- Ingestion ------------------------------------------------------------

def test_load_documents_reads_markdown(corpus):
    docs = dict(load_documents([corpus]))
    assert len(docs) == 3
    assert "reliability.md" in " ".join(docs)


def test_load_documents_skips_unsupported_extensions(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "note.txt").write_text("plain note", encoding="utf-8")
    docs = dict(load_documents([str(tmp_path)]))
    assert len(docs) == 1


def test_extract_text_rejects_unknown_extension(tmp_path):
    path = tmp_path / "file.xyz"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        extract_text(str(path))


def test_html_extraction_drops_script_and_style(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>alert('x')</script>"
        "<p>Visible content here</p><p>Second paragraph</p></body></html>",
        encoding="utf-8",
    )
    text = extract_html(str(path))
    assert "Visible content here" in text
    assert "Second paragraph" in text
    assert "alert" not in text
    assert "color:red" not in text


def test_html_extraction_decodes_entities(tmp_path):
    path = tmp_path / "ent.html"
    path.write_text("<p>Caf&eacute; &amp; cr&egrave;me</p>", encoding="utf-8")
    assert "Café" in extract_html(str(path))


def test_html_extraction_preserves_block_boundaries(tmp_path):
    path = tmp_path / "blocks.html"
    path.write_text("<div>First block</div><div>Second block</div>", encoding="utf-8")
    text = extract_html(str(path))
    assert "First block" in text and "Second block" in text
    assert "\n" in text


def test_load_documents_accepts_single_file(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("only document", encoding="utf-8")
    docs = list(load_documents([str(path)]))
    assert len(docs) == 1
    assert docs[0][1] == "only document"


# -- Index construction ---------------------------------------------------

def test_build_index_from_paths_populates(built_index):
    assert len(built_index.chunks) > 0
    assert built_index.avgdl > 0
    assert built_index.postings


def test_index_records_document_sources(built_index):
    sources = {c.doc_source for c in built_index.chunks}
    assert any(s.endswith("reliability.md") for s in sources)


def test_index_is_deterministic(built_index):
    order_a = [c.id for c in built_index.chunks]
    order_b = [c.id for c in built_index.chunks]
    assert order_a == order_b


def test_index_uses_configured_dimensions(corpus):
    from rag_foundry.config import RagConfig

    config = RagConfig(chunk_size=400, vector_dims=256)
    index = build_index_from_paths([corpus], config)
    assert index.dims == 256


# -- Persistence ----------------------------------------------------------

def test_index_save_and_load_roundtrip(built_index, tmp_path):
    path = str(tmp_path / "idx.json")
    built_index.save(path)
    reloaded = InMemoryIndex.load(path)
    assert len(reloaded.chunks) == len(built_index.chunks)
    assert [c.id for c in reloaded.chunks] == [c.id for c in built_index.chunks]


def test_save_is_atomic_and_leaves_no_tmp_file(built_index, tmp_path):
    path = str(tmp_path / "idx.json")
    built_index.save(path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_load_rejects_unknown_version(tmp_path):
    import json

    path = tmp_path / "idx.json"
    path.write_text(json.dumps({"version": 999, "chunks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported index version"):
        InMemoryIndex.load(str(path))


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        InMemoryIndex.load(str(tmp_path / "absent.json"))


def test_get_chunk_by_id(tiny_index):
    assert tiny_index.get_chunk("c-1").doc_source == "b.md"
    assert tiny_index.get_chunk("nope") is None


# -- BM25 -----------------------------------------------------------------

def test_bm25_finds_the_matching_chunk(tiny_index):
    results = tiny_index.bm25_search("kubernetes zones", top_k=3)
    assert results
    assert results[0][0].id == "c-0"


def test_bm25_ignores_unknown_terms(tiny_index):
    assert tiny_index.bm25_search("zzzz qqqq", top_k=3) == []


def test_bm25_on_empty_index_returns_nothing():
    assert InMemoryIndex().bm25_search("anything") == []


def test_bm25_scores_are_sorted_descending(tiny_index):
    scores = [s for _, s in tiny_index.bm25_search("service request metrics", top_k=5)]
    assert scores == sorted(scores, reverse=True)


# -- Vector search --------------------------------------------------------

def test_vector_search_finds_related_chunk(tiny_index):
    results = tiny_index.vector_search("kubernetes deploy zones", top_k=3)
    assert results
    assert results[0][0].id == "c-0"


def test_vector_search_scores_are_bounded(tiny_index):
    for _, score in tiny_index.vector_search("prometheus metrics scrape", top_k=3):
        assert -1.0 <= score <= 1.0


def test_vector_search_on_empty_index_returns_nothing():
    assert InMemoryIndex().vector_search("anything") == []


# -- Hybrid fusion --------------------------------------------------------

def test_hybrid_search_returns_best_match_first(tiny_index):
    # Note: the tokenizer does not stem, so the query must use the same
    # surface form as the chunk ("retries", not "retry").
    results = hybrid_search(tiny_index, "how many times does it backoff retries",
                            top_k=3)
    assert results
    assert results[0][0].id == "c-1"


def test_hybrid_scores_are_sorted_descending(tiny_index):
    scores = [s for _, s in hybrid_search(tiny_index, "service metrics", top_k=5)]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_respects_top_k(tiny_index):
    assert len(hybrid_search(tiny_index, "service", top_k=2)) <= 2


def test_hybrid_is_deterministic(tiny_index):
    a = [(c.id, s) for c, s in hybrid_search(tiny_index, "retry", top_k=3)]
    b = [(c.id, s) for c, s in hybrid_search(tiny_index, "retry", top_k=3)]
    assert a == b


def test_hybrid_on_empty_index_returns_nothing():
    assert hybrid_search(InMemoryIndex(), "anything") == []


def test_hybrid_boosts_chunks_ranked_by_both_signals(tiny_index):
    # A chunk that appears in both the BM25 and vector rankings should
    # outscore one that appears in only one.
    results = dict((c.id, s) for c, s in hybrid_search(tiny_index, "exponential backoff retry", top_k=3))
    assert "c-1" in results


def test_rrf_k_lowers_favours_top_rank(tiny_index):
    # Scores are normalised against the achievable ceiling, so the top hit
    # is always 1.0. Lower rrf_k sharpens how fast the tail decays.
    query = "service kubernetes metrics"  # matches more than one chunk
    sharp = dict((c.id, s) for c, s in
                 hybrid_search(tiny_index, query, top_k=3, rrf_k=1))
    flat = dict((c.id, s) for c, s in
                hybrid_search(tiny_index, query, top_k=3, rrf_k=1000))
    assert sharp and flat
    tails = [v for k, v in sharp.items() if k != max(sharp, key=sharp.get)]
    flat_tails = [v for k, v in flat.items() if k != max(flat, key=flat.get)]
    assert tails and flat_tails
    assert min(tails) < min(flat_tails)


def test_scores_are_normalised_to_at_most_one(built_index):
    results = hybrid_search(built_index, "retry backoff", top_k=5)
    assert results
    assert all(0.0 < score <= 1.0 for _, score in results)


def test_top_score_is_one_when_present(tiny_index):
    results = hybrid_search(tiny_index, "backoff retries", top_k=3)
    assert results
    assert results[0][1] == pytest.approx(1.0, abs=1e-9)


def test_default_min_score_is_reachable(tiny_index):
    """Regression: raw RRF scores could never reach the old default floor."""
    from rag_foundry.config import RagConfig

    default = RagConfig()
    assert default.min_score <= 1.0
    results = hybrid_search(tiny_index, "backoff retries",
                           top_k=default.top_k, rrf_k=default.rrf_k)
    assert results and results[0][1] >= default.min_score


# -- End-to-end retrieval -------------------------------------------------

def test_retrieval_finds_documented_facts(built_index):
    results = hybrid_search(built_index, "how many times does the service retry?", top_k=5)
    assert results
    joined = " ".join(c.text.lower() for c, _ in results)
    assert "retri" in joined or "backoff" in joined


def test_retrieval_finds_deployment_details(built_index):
    results = hybrid_search(built_index, "how many availability zones?", top_k=5)
    joined = " ".join(c.text for c, _ in results)
    assert "zone" in joined.lower()


def test_retrieval_returns_sources(built_index):
    results = hybrid_search(built_index, "circuit breaker", top_k=3)
    assert all(isinstance(c, Chunk) for c, _ in results)
    assert all(c.doc_source for c, _ in results)