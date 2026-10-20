"""Tests for configuration loading and the chunking strategies."""

from __future__ import annotations

import json

import pytest

from rag_foundry.chunk import chunk_text
from rag_foundry.config import RagConfig, load_config


# -- Configuration --------------------------------------------------------

def test_defaults_are_usable_without_a_file():
    config = RagConfig()
    assert config.chunk_strategy == "fixed"
    assert config.top_k == 5
    assert config.min_score == 0.1


def test_load_config_with_no_path_returns_defaults():
    assert load_config(None).chunk_size == 800


def test_from_dict_ignores_unknown_keys():
    config = RagConfig.from_dict({"top_k": 9, "not_a_real_key": 1})
    assert config.top_k == 9
    assert not hasattr(config, "not_a_real_key")


def test_from_dict_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        RagConfig.from_dict(["nope"])


def test_load_config_reads_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"top_k": 11, "vector_dims": 512}), encoding="utf-8")
    config = load_config(str(path))
    assert config.top_k == 11
    assert config.vector_dims == 512


def test_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv("RAGF_TEST_KEY", "sk-abc")
    config = RagConfig(api_key_env="RAGF_TEST_KEY")
    assert config.api_key() == "sk-abc"


def test_api_key_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("RAGF_ABSENT_KEY", raising=False)
    assert RagConfig(api_key_env="RAGF_ABSENT_KEY").api_key() is None


def test_to_dict_roundtrips():
    original = RagConfig(top_k=3)
    assert RagConfig.from_dict(original.to_dict()).top_k == 3


# -- Fixed chunking -------------------------------------------------------

def test_fixed_chunking_splits_long_text():
    text = "a" * 1000
    chunks = chunk_text(text, "doc.md", strategy="fixed", size=300, overlap=0)
    assert len(chunks) == 4
    assert all(len(c.text) <= 300 for c in chunks)


def test_fixed_chunking_assigns_sequential_indices():
    chunks = chunk_text("b" * 900, "doc.md", strategy="fixed", size=300, overlap=0)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_fixed_chunking_overlap_repeats_content():
    chunks = chunk_text("c" * 500, "doc.md", strategy="fixed", size=200, overlap=50)
    assert chunks[0].text[-50:] == chunks[1].text[:50]


def test_chunk_ids_are_stable_and_unique():
    first = chunk_text("hello world", "doc.md", size=100)
    second = chunk_text("hello world", "doc.md", size=100)
    assert first[0].id == second[0].id
    ids = [c.id for c in chunk_text("x" * 500, "doc.md", size=100, overlap=0)]
    assert len(set(ids)) == len(ids)


def test_chunk_ids_differ_per_document():
    a = chunk_text("same text", "a.md", size=100)[0]
    b = chunk_text("same text", "b.md", size=100)[0]
    assert a.id != b.id


def test_overlap_larger_than_size_is_clamped():
    chunks = chunk_text("d" * 300, "doc.md", strategy="fixed", size=100, overlap=500)
    assert len(chunks) > 1


def test_empty_text_produces_no_chunks():
    assert chunk_text("", "doc.md", strategy="fixed") == []


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        chunk_text("text", "doc.md", strategy="paragraph")


# -- Sentence chunking ----------------------------------------------------

def test_sentence_chunking_respects_size():
    text = "One sentence here. Another sentence follows. And a third one."
    chunks = chunk_text(text, "doc.md", strategy="sentence", size=40, overlap=0)
    assert all(len(c.text) <= 40 for c in chunks)


def test_sentence_chunking_keeps_sentences_whole():
    text = "Alpha bravo charlie. Delta echo foxtrot."
    chunks = chunk_text(text, "doc.md", strategy="sentence", size=60, overlap=0)
    joined = " ".join(c.text for c in chunks)
    assert "Alpha bravo charlie." in joined
    assert "Delta echo foxtrot." in joined


def test_sentence_chunking_hard_splits_overlong_sentence():
    sentence = "word " * 200
    chunks = chunk_text(sentence, "doc.md", strategy="sentence", size=100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_sentence_chunking_carries_overlap():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text(text, "doc.md", strategy="sentence", size=45, overlap=20)
    assert len(chunks) >= 2


def test_sentence_chunking_handles_newline_separators():
    text = "Line one content\nLine two content\nLine three content"
    chunks = chunk_text(text, "doc.md", strategy="sentence", size=25, overlap=0)
    assert len(chunks) >= 2


def test_both_strategies_cover_all_content():
    text = "Sentence number one. Sentence number two. Sentence number three."
    for strategy in ("fixed", "sentence"):
        chunks = chunk_text(text, "doc.md", strategy=strategy, size=40, overlap=0)
        assert chunks
        assert all(c.doc_source == "doc.md" for c in chunks)