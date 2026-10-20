"""Tests for grounded generation, citation sanitisation, and evaluation.

No test contacts an LLM. The network path is exercised through monkeypatching
so the suite is deterministic and runs without an API key.
"""

from __future__ import annotations

import json

import pytest

from rag_foundry.chunk import Chunk
from rag_foundry.config import RagConfig
from rag_foundry.eval import (
    EvalItem,
    _extract_json_object,
    _normalize_scores,
    judge_answer,
    keyword_hit_rate,
    run_eval,
    summary_to_markdown,
    write_report,
)
from rag_foundry.generate import (
    Answer,
    LLMError,
    Refusal,
    _extractive_fallback,
    _sanitize_citations,
    answer_question,
    build_prompt,
    call_llm,
)
from rag_foundry.index import InMemoryIndex


# -- Prompt construction --------------------------------------------------

def test_build_prompt_numbers_and_labels_each_block():
    chunks = [
        Chunk(id="c-0", doc_source="a.md", text="First fact.", index=0),
        Chunk(id="c-1", doc_source="b.md", text="Second fact.", index=0),
    ]
    system, user = build_prompt("What is the fact?", chunks)
    assert "[1] (a.md)" in user
    assert "[2] (b.md)" in user
    assert "First fact." in user
    assert "What is the fact?" in user
    assert "cite" in system.lower()


def test_build_prompt_with_no_chunks_is_still_valid():
    system, user = build_prompt("question", [])
    assert isinstance(system, str) and isinstance(user, str)


# -- Citation sanitisation ------------------------------------------------

def test_sanitize_keeps_in_range_citations():
    assert _sanitize_citations("Fact one [1] and fact two [2].", 2) == \
        "Fact one [1] and fact two [2]."


def test_sanitize_drops_out_of_range_citations():
    assert _sanitize_citations("Real [1] but invented [9].", 2) == \
        "Real [1] but invented ."


def test_sanitize_drops_citation_zero():
    assert "[0]" not in _sanitize_citations("Claim [0] here.", 3)


def test_sanitize_with_zero_context_removes_all():
    assert _sanitize_citations("Claim [1] here.", 0) == "Claim  here."


def test_generate_sanitize_treats_any_numeric_bracket_as_a_citation():
    # Known trade-off: the sanitizer matches any [digits] token, so a
    # non-citation like "array[0]" is also stripped. This is the safe
    # direction to err in - a stripped bracket is cosmetic, while a
    # retained fabricated citation is a correctness problem.
    assert _sanitize_citations("array[0] and list[1]", 5) == "array and list[1]"


def test_generate_sanitize_leaves_non_numeric_brackets_alone():
    assert _sanitize_citations("see [note] and [ref]", 5) == "see [note] and [ref]"


# -- Extractive fallback --------------------------------------------------

def test_extractive_fallback_cites_block_one():
    chunk = Chunk(id="c-0", doc_source="a.md",
                  text="Backoff retries three times. Jitter randomises the delay.",
                  index=0)
    result = _extractive_fallback("how many retries", [chunk])
    assert result.endswith("[1]")
    assert "Backoff" in result


def test_extractive_fallback_with_no_chunks_is_empty():
    assert _extractive_fallback("question", []) == ""


# -- LLM call guards ------------------------------------------------------

def test_call_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("RAGF_TEST_ABSENT", raising=False)
    config = RagConfig(api_key_env="RAGF_TEST_ABSENT")
    with pytest.raises(LLMError, match="no API key"):
        call_llm("system", "user", config)


def test_call_llm_wraps_connection_errors(monkeypatch):
    monkeypatch.setenv("RAGF_TEST_KEY", "sk-test")
    config = RagConfig(api_key_env="RAGF_TEST_KEY",
                       llm_base_url="http://127.0.0.1:1", llm_timeout=0.1)

    import urllib.request

    def boom(*args, **kwargs):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(LLMError, match="connection error"):
        call_llm("system", "user", config)


# -- answer_question behaviour --------------------------------------------

def test_answer_refuses_on_empty_index(config):
    result = answer_question("anything", InMemoryIndex(), config, use_llm=False)
    assert isinstance(result, Refusal)
    assert result.reason == "no_chunks"


def test_answer_refuses_with_no_results_when_index_has_content(tiny_index):
    config = RagConfig(min_score=0.0)
    # No token in this query appears in any chunk, so no ranking matches.
    result = answer_question("zzqqx wwvvu", tiny_index, config, use_llm=False)
    assert isinstance(result, Refusal)
    assert result.reason == "no_results"
    assert "3" in result.message


def test_answer_refuses_below_min_score(tiny_index):
    config = RagConfig(min_score=10.0)
    result = answer_question("kubernetes", tiny_index, config, use_llm=False)
    assert isinstance(result, Refusal)
    assert result.reason == "low_score"
    assert result.top_score is not None


def test_answer_offline_returns_extract(tiny_index):
    config = RagConfig(min_score=0.0, top_k=3)
    result = answer_question("backoff retries request", tiny_index, config,
                             use_llm=False)
    assert isinstance(result, Answer)
    assert result.text
    assert result.sources


def test_answer_includes_prompt_and_sources(built_index):
    config = RagConfig(min_score=0.0, top_k=3)
    result = answer_question("circuit breaker cooldown", built_index, config,
                             use_llm=False)
    assert isinstance(result, Answer)
    assert result.prompt[0] and result.prompt[1]
    assert all(c.doc_source for c in result.sources)


def test_answer_sanitizes_llm_citations(monkeypatch, tiny_index):
    config = RagConfig(min_score=0.0, top_k=2)
    monkeypatch.setattr("rag_foundry.generate.call_llm",
                        lambda s, u, c: "Grounded [1] but hallucinated [42].")
    result = answer_question("backoff retries", tiny_index, config, use_llm=True)
    assert isinstance(result, Answer)
    assert "[1]" in result.text
    assert "[42]" not in result.text


# -- Evaluation helpers ---------------------------------------------------

def test_keyword_hit_rate_splits_hits_and_misses():
    hits, misses = keyword_hit_rate("The Backoff policy applies.", ["backoff", "jitter"])
    assert hits == ["backoff"]
    assert misses == ["jitter"]


def test_keyword_hit_rate_is_case_insensitive():
    hits, _ = keyword_hit_rate("BACKOFF here", ["backoff"])
    assert hits == ["backoff"]


def test_keyword_hit_rate_with_no_keywords():
    assert keyword_hit_rate("text", []) == ([], [])


def test_extract_json_object_finds_embedded_object():
    text = 'Sure! {"faithfulness": 4, "relevance": 5, "citation_correctness": 3} done'
    parsed = _extract_json_object(text)
    assert parsed["faithfulness"] == 4


def test_extract_json_object_handles_braces_inside_strings():
    text = '{"note": "a } brace", "faithfulness": 5}'
    parsed = _extract_json_object(text)
    assert parsed["note"] == "a } brace"


def test_extract_json_object_returns_none_when_absent():
    assert _extract_json_object("no json here") is None


def test_normalize_scores_accepts_valid_triple():
    raw = {"faithfulness": 5, "relevance": 4, "citation_correctness": 3}
    assert _normalize_scores(raw) == raw


def test_normalize_scores_accepts_integer_floats():
    raw = {"faithfulness": 5.0, "relevance": 4.0, "citation_correctness": 3.0}
    assert _normalize_scores(raw)["faithfulness"] == 5


def test_normalize_scores_rejects_out_of_range():
    assert _normalize_scores({"faithfulness": 9, "relevance": 4,
                              "citation_correctness": 3}) is None


def test_normalize_scores_rejects_missing_key():
    assert _normalize_scores({"faithfulness": 4, "relevance": 4}) is None


def test_normalize_scores_rejects_boolean():
    assert _normalize_scores({"faithfulness": True, "relevance": 4,
                              "citation_correctness": 3}) is None


def test_normalize_scores_rejects_none():
    assert _normalize_scores(None) is None


# -- Judge ----------------------------------------------------------------

def test_judge_answer_degrades_on_llm_error(monkeypatch):
    config = RagConfig()
    monkeypatch.setattr("rag_foundry.eval.call_llm",
                        lambda s, u, c: (_ for _ in ()).throw(LLMError("down")))
    scores, error = judge_answer("q", "a", "ref", config)
    assert scores is None
    assert "down" in error


def test_judge_answer_parses_scores(monkeypatch):
    config = RagConfig()
    payload = '{"faithfulness": 4, "relevance": 5, "citation_correctness": 4}'
    monkeypatch.setattr("rag_foundry.eval.call_llm", lambda s, u, c: payload)
    scores, error = judge_answer("q", "a", "ref", config)
    assert error is None
    assert scores["faithfulness"] == 4


def test_judge_answer_flags_unparseable(monkeypatch):
    config = RagConfig()
    monkeypatch.setattr("rag_foundry.eval.call_llm", lambda s, u, c: "no json")
    scores, error = judge_answer("q", "a", "ref", config)
    assert scores is None
    assert "unparseable" in error


# -- Full evaluation run --------------------------------------------------

@pytest.fixture
def qa_file(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([
        {"question": "backoff retries three times",
         "reference_answer": "Retries three times with backoff.",
         "expected_keywords": ["backoff", "retries"]},
        {"question": "kubernetes availability zones",
         "reference_answer": "Three zones.",
         "expected_keywords": ["zone"]},
    ]), encoding="utf-8")
    return str(path)


def test_run_eval_offline_produces_report(qa_file, built_index, tmp_path, monkeypatch):
    monkeypatch.setattr("rag_foundry.eval.answer_question",
                        lambda q, i, c: answer_question(q, i, c, use_llm=False))
    out = tmp_path / "report.json"
    report = run_eval(qa_file, built_index, RagConfig(min_score=0.0),
                      out_path=str(out), use_judge=False)
    assert out.exists()
    assert report["summary"]["items"] == 2
    assert len(report["items"]) == 2
    assert "generated_at" in report
    assert report["model"]


def test_run_eval_records_keyword_hits(qa_file, built_index, tmp_path, monkeypatch):
    monkeypatch.setattr("rag_foundry.eval.answer_question",
                        lambda q, i, c: answer_question(q, i, c, use_llm=False))
    report = run_eval(qa_file, built_index, RagConfig(min_score=0.0),
                      out_path=str(tmp_path / "r.json"), use_judge=False)
    assert any(item["keyword_hits"] for item in report["items"])


def test_run_eval_without_judge_leaves_scores_null(qa_file, built_index, tmp_path,
                                                    monkeypatch):
    monkeypatch.setattr("rag_foundry.eval.answer_question",
                        lambda q, i, c: answer_question(q, i, c, use_llm=False))
    report = run_eval(qa_file, built_index, RagConfig(min_score=0.0),
                      out_path=str(tmp_path / "r.json"), use_judge=False)
    assert all(item["judge"] is None for item in report["items"])
    assert report["summary"]["mean_faithfulness"] is None


def test_run_eval_with_judge_records_scores(qa_file, built_index, tmp_path, monkeypatch):
    monkeypatch.setattr("rag_foundry.eval.answer_question",
                        lambda q, i, c: answer_question(q, i, c, use_llm=False))
    monkeypatch.setattr(
        "rag_foundry.eval.call_llm",
        lambda s, u, c: '{"faithfulness": 4, "relevance": 4, "citation_correctness": 5}')
    report = run_eval(qa_file, built_index, RagConfig(min_score=0.0),
                      out_path=str(tmp_path / "r.json"), use_judge=True)
    assert report["summary"]["judged"] == 2
    assert report["summary"]["mean_faithfulness"] == 4.0


def test_run_eval_rejects_bad_qa_file(built_index, tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"no_question_key": True}]), encoding="utf-8")
    monkeypatch.setattr("rag_foundry.eval.answer_question",
                        lambda q, i, c: answer_question(q, i, c, use_llm=False))
    with pytest.raises(ValueError, match="question"):
        run_eval(str(bad), built_index, RagConfig(), out_path=str(tmp_path / "r.json"),
                 use_judge=False)


def test_run_eval_accepts_items_wrapper(built_index, tmp_path, monkeypatch):
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"items": [
        {"question": "circuit breaker", "expected_keywords": ["breaker"]}]}),
        encoding="utf-8")
    monkeypatch.setattr("rag_foundry.eval.answer_question",
                        lambda q, i, c: answer_question(q, i, c, use_llm=False))
    report = run_eval(str(path), built_index, RagConfig(min_score=0.0),
                      out_path=str(tmp_path / "r.json"), use_judge=False)
    assert report["summary"]["items"] == 1


# -- Reporting ------------------------------------------------------------

def test_summary_to_markdown_renders_table():
    report = {"summary": {"items": 3, "refusals": 1, "judged": 2,
                          "mean_faithfulness": 4.0}}
    table = summary_to_markdown(report)
    assert "| Metric | Value |" in table
    assert "| Items | 3 |" in table
    assert "| Mean faithfulness | 4.0 |" in table


def test_summary_to_markdown_shows_na_for_missing():
    assert "n/a" in summary_to_markdown({"summary": {}})


def test_write_report_is_valid_json(tmp_path):
    report = {"summary": {"items": 1}}
    path = tmp_path / "out.json"
    write_report(report, str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_eval_item_defaults():
    item = EvalItem(question="q", answer="a", refusal="")
    assert item.keyword_hits == []
    assert item.judge is None