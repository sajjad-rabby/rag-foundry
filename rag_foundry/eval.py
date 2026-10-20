"""Answer evaluation.

Keyword overlap is always available; the LLM judge is optional and, when absent,

Every item records the question, the answer or refusal, keyword hits
against the reference answer, and (when an API key is available) three
1-5 judge scores. The judge is strictly optional: without a key the run
still produces a full report from keyword metrics alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rag_foundry.config import RagConfig
from rag_foundry.generate import LLMError, answer_question, call_llm

JUDGE_SYSTEM = (
    "You are a strict RAG evaluation judge. You will receive a question, "
    "an answer, and reference material. Reply with ONLY a JSON object of "
    "the form {\"faithfulness\": <1-5>, \"relevance\": <1-5>, "
    "\"citation_correctness\": <1-5>} where faithfulness means every "
    "claim is supported by the reference material, relevance means the "
    "answer addresses the question, and citation_correctness means "
    "citations are present, in range, and point at the right sources. "
    "No prose, no markdown fences."
)

_SCORE_KEYS = ("faithfulness", "relevance", "citation_correctness")


@dataclass
class EvalItem:
    """One evaluated question with all recorded metrics."""

    question: str
    answer: str
    refusal: str
    keyword_hits: List[str] = field(default_factory=list)
    keyword_misses: List[str] = field(default_factory=list)
    judge: Optional[Dict[str, int]] = None
    judge_error: Optional[str] = None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced JSON object out of a model response.

Returns:

    Scans brace depth while tracking string and escape state so braces
    inside JSON strings are ignored, then parses the candidate. Returns
    None when no balanced object parses.
    """
    start = None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None
    return None


def _normalize_scores(
    raw: Optional[Dict[str, Any]]
) -> Optional[Dict[str, int]]:
    """Validate judge scores, returning only clean 1-5 integer values."""
    if not raw:
        return None
    scores: Dict[str, int] = {}
    for key in _SCORE_KEYS:
        if key not in raw:
            return None
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float):
            if not value.is_integer():
                return None
            value = int(value)
        if not 1 <= value <= 5:
            return None
        scores[key] = value
    return scores


def judge_answer(
    question: str, answer: str, reference: str, config: RagConfig
) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """Score one answer against a reference using an LLM judge.

Returns:

    Any :class:`LLMError` or malformed judge reply degrades to
    ``(None, error)`` instead of failing the whole eval run.
    """
    user = (
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        f"Reference material:\n{reference}"
    )
    try:
        raw = call_llm(JUDGE_SYSTEM, user, config)
    except LLMError as exc:
        return None, str(exc)
    parsed = _normalize_scores(_extract_json_object(raw))
    if parsed is None:
        return None, f"judge returned unparseable scores: {raw[:200]!r}"
    return parsed, None


def keyword_hit_rate(
    text: str, keywords: List[str]
) -> Tuple[List[str], List[str]]:
    """Fraction of reference keywords present in the generated answer.

Returns:

    Matching is case-insensitive substring containment.
    """
    lowered = text.lower()
    hits = [kw for kw in keywords if kw.lower() in lowered]
    misses = [kw for kw in keywords if kw.lower() not in lowered]
    return hits, misses


def _load_qa_items(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load QA items from JSON, accepting a bare list or {"items": [...]}."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list):
        raise ValueError(
            "QA file must be a JSON list of items or an object with an "
            "'items' list"
        )
    for item in data:
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(
                "every QA item must be an object with a 'question' key"
            )
    return data


def _reference_text(item: Dict[str, Any]) -> str:
    """Build the reference text fed to the judge for one item."""
    parts = []
    reference = item.get("reference_answer")
    if reference:
        parts.append(f"Reference answer: {reference}")
    keywords = item.get("expected_keywords") or []
    if keywords:
        parts.append("Expected keywords: " + ", ".join(str(k) for k in keywords))
    return "\n".join(parts)


def run_eval(
    qa_path: Union[str, Path],
    index: Any,
    config: RagConfig,
    out_path: Union[str, Path] = "eval_report.json",
    use_judge: bool = True,
) -> Dict[str, Any]:
    """Evaluate every item in a dataset and summarise the results.

Returns:

    Reads QA items from ``qa_path``, answers each one against ``index``
    via :func:`rag_foundry.generate.answer_question`, records keyword
    hit rates, optionally asks the judge model for 1-5 scores, and
    writes a report with per-item detail and aggregate summary metrics.
    """
    items_raw = _load_qa_items(qa_path)
    eval_items: List[EvalItem] = []
    for item in items_raw:
        question = str(item["question"])
        keywords = [str(k) for k in (item.get("expected_keywords") or [])]
        result = answer_question(question, index, config)
        refusal = ""
        if isinstance(result, type(None)) or hasattr(result, "reason"):
            refusal = getattr(result, "reason", "")
        answer_text = getattr(result, "text", "") or ""
        if hasattr(result, "message") and hasattr(result, "reason"):
            answer_text = result.message
        hits, misses = keyword_hit_rate(answer_text, keywords)
        judge_scores: Optional[Dict[str, int]] = None
        judge_error: Optional[str] = None
        if use_judge and not refusal:
            reference = _reference_text(item)
            if reference:
                judge_scores, judge_error = judge_answer(
                    question, answer_text, reference, config
                )
        eval_items.append(
            EvalItem(
                question=question,
                answer=answer_text,
                refusal=refusal,
                keyword_hits=hits,
                keyword_misses=misses,
                judge=judge_scores,
                judge_error=judge_error,
            )
        )

    judged = [it for it in eval_items if it.judge is not None]
    judge_errors = [it for it in eval_items if it.judge_error is not None]
    refusals = [it for it in eval_items if it.refusal]
    hit_rates = []
    for it in eval_items:
        total = len(it.keyword_hits) + len(it.keyword_misses)
        if total:
            hit_rates.append(len(it.keyword_hits) / total)

    def _mean(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 3) if values else None

    def _mean_key(key: str) -> Optional[float]:
        return _mean([float(it.judge[key]) for it in judged if key in it.judge])

    summary = {
        "items": len(eval_items),
        "refusals": len(refusals),
        "judged": len(judged),
        "judge_errors": len(judge_errors),
        "mean_faithfulness": _mean_key("faithfulness"),
        "mean_relevance": _mean_key("relevance"),
        "mean_citation_correctness": _mean_key("citation_correctness"),
        "mean_keyword_hit_rate": _mean(hit_rates),
    }
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config.llm_model,
        "config_used": config.to_dict(),
        "items": [asdict(it) for it in eval_items],
        "summary": summary,
    }
    write_report(report, out_path)
    return report


def write_report(report: Dict[str, Any], out_path: Union[str, Path]) -> None:
    """Write the evaluation report as pretty-printed JSON."""
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def summary_to_markdown(report: Dict[str, Any]) -> str:
    """Render the report summary as a small Markdown table."""
    s = report.get("summary", {})

    def _fmt(value: Any) -> str:
        return "n/a" if value is None else str(value)

    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Items | {_fmt(s.get('items'))} |",
        f"| Refusals | {_fmt(s.get('refusals'))} |",
        f"| Judged | {_fmt(s.get('judged'))} |",
        f"| Judge errors | {_fmt(s.get('judge_errors'))} |",
        f"| Mean faithfulness | {_fmt(s.get('mean_faithfulness'))} |",
        f"| Mean relevance | {_fmt(s.get('mean_relevance'))} |",
        f"| Mean citation correctness | {_fmt(s.get('mean_citation_correctness'))} |",
        f"| Mean keyword hit rate | {_fmt(s.get('mean_keyword_hit_rate'))} |",
    ]
    return "\n".join(lines)
