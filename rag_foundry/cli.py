"""Command-line interface for the foundry.

Subcommands:
- ``ragf ingest``    ingest documents and build the index
- ``ragf query``     run a hybrid search
- ``ragf eval``      score answers against a JSONL set
- ``ragf report``             render an existing eval report
- ``ragf ingest <paths...>``  index documents (files or directories)
- ``ragf ask "question"``     answer a question from the index
- ``ragf eval qa.json``       run the evaluation suite
- ``ragf report``             render an existing eval report summary
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from rag_foundry import (
    RagConfig,
    answer_question,
    build_index_from_paths,
    load_config,
)
from rag_foundry.eval import run_eval, summary_to_markdown


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragf",
        description="A minimal, stdlib-first RAG pipeline.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to a JSON config file (default: built-in defaults)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index documents")
    ingest.add_argument("paths", nargs="+", help="files or directories to index")
    ingest.add_argument(
        "--index",
        default="ragf_index.json",
        help="output index path (default: ragf_index.json)",
    )

    ask = sub.add_parser("ask", help="answer a question from the index")
    ask.add_argument("question", help="the question to answer")
    ask.add_argument(
        "--index", default="ragf_index.json", help="index path to load"
    )
    ask.add_argument(
        "--no-llm",
        action="store_true",
        help="print the built prompt instead of calling the LLM",
    )
    ask.add_argument("--top-k", type=int, default=None, help="retrieve top N chunks")
    ask.add_argument(
        "--min-score", type=float, default=None, help="minimum score to answer"
    )

    ev = sub.add_parser("eval", help="run the evaluation suite")
    ev.add_argument("qa", help="path to the QA JSON file")
    ev.add_argument(
        "--index", default="ragf_index.json", help="index path to load"
    )
    ev.add_argument(
        "--out", default="eval_report.json", help="report output path"
    )
    ev.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the LLM judge (keyword metrics only)",
    )

    report = sub.add_parser("report", help="summarize an eval report")
    report.add_argument(
        "path", nargs="?", default="eval_report.json", help="report path"
    )
    return parser


def _apply_overrides(
    config: RagConfig, args: argparse.Namespace
) -> RagConfig:
    """Apply CLI-level overrides for ask-time knobs."""
    if getattr(args, "top_k", None) is not None:
        config.top_k = int(args.top_k)
    if getattr(args, "min_score", None) is not None:
        config.min_score = float(args.min_score)
    return config


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    index = build_index_from_paths(args.paths, config)
    index.save(args.index)
    print(f"indexed {len(index.chunks)} chunks -> {args.index}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from rag_foundry.index import InMemoryIndex

    config = _apply_overrides(load_config(args.config), args)
    index = InMemoryIndex.load(args.index)
    result = answer_question(args.question, index, config, use_llm=not args.no_llm)
    if hasattr(result, "reason") and hasattr(result, "message"):
        print(f"[refused: {result.reason}] {result.message}")
        return 1
    print(result.text)
    if result.sources:
        print("\nSources:")
        for i, chunk in enumerate(result.sources, start=1):
            print(f"  [{i}] {chunk.doc_source}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from rag_foundry.index import InMemoryIndex

    config = load_config(args.config)
    index = InMemoryIndex.load(args.index)
    report = run_eval(
        args.qa,
        index,
        config,
        out_path=args.out,
        use_judge=not args.no_judge,
    )
    print(f"wrote report -> {args.out}")
    print()
    print(summary_to_markdown(report))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    with open(args.path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    print(summary_to_markdown(report))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``ragf`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "ingest": _cmd_ingest,
        "ask": _cmd_ask,
        "eval": _cmd_eval,
        "report": _cmd_report,
    }
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
