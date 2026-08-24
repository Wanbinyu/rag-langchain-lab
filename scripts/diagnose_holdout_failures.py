"""Classify holdout retrieval failures into A–E buckets.

A: expected source never reaches a wide Top-N list (pool miss)
B: expected source appears in wide list but not Top-K (rank/rerank miss)
C: expected source in Top-K but evidence is insufficient/review-low (threshold)
D: expected source in Top-K but primary hit looks like a heading/fragment
E: source hit with usable evidence (healthy or only soft review)

Runs retrieval only — no Chat API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.cli import build_embeddings
from src.config import project_settings
from src.indexing import open_store
from src.retrieval import RetrievalReport, assess_evidence, retrieve_report


def _load_cases(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _source_rank(results: list, expected: set[str]) -> int | None:
    for rank, item in enumerate(results, start=1):
        source = str(item.document.metadata.get("source", ""))
        if source in expected:
            return rank
    return None


def _looks_fragment(text: str) -> bool:
    compact = " ".join(text.split())
    if len(compact) < 80:
        return True
    # Short title-like lines without sentence punctuation.
    if len(compact) < 120 and compact.count(" ") < 8 and "。" not in compact and ". " not in compact:
        return True
    return False


def classify_case(
    *,
    expect_abstention: bool,
    expected_sources: list[str],
    top_k_report,
    wide_report,
    top_k: int,
) -> dict:
    expected = set(expected_sources)
    top_sources = [
        str(item.document.metadata.get("source", ""))
        for item in top_k_report.results
    ]
    evidence = top_k_report.evidence
    top_score = evidence.top_score
    top_rank = _source_rank(top_k_report.results, expected)
    wide_rank = _source_rank(wide_report.results, expected)

    if expect_abstention:
        bucket = "OOS"
        note = "out_of_scope"
    elif not expected:
        bucket = "E"
        note = "no_expected_sources"
    elif top_rank is not None:
        primary = top_k_report.results[top_rank - 1]
        content = primary.document.page_content
        if evidence.status == "insufficient":
            bucket = "C"
            note = "source_in_topk_but_insufficient"
        elif _looks_fragment(content) and evidence.status != "sufficient":
            bucket = "D"
            note = "source_hit_but_fragment_context"
        else:
            bucket = "E"
            note = f"healthy_{evidence.status}"
    elif wide_rank is not None:
        bucket = "B"
        note = f"source_rank_{wide_rank}_outside_top_{top_k}"
    else:
        bucket = "A"
        note = "expected_source_not_in_wide_pool"

    return {
        "bucket": bucket,
        "note": note,
        "evidence_status": evidence.status,
        "evidence_reason": evidence.reason_code,
        "top_score": top_score,
        "score_gap": evidence.score_gap,
        "top_rank": top_rank,
        "wide_rank": wide_rank,
        "top_sources": top_sources,
        "retry_count": top_k_report.retry_count,
        "retry_query": top_k_report.retry_query,
        "total_ms": top_k_report.total_ms,
        "primary_excerpt": (
            " ".join(top_k_report.results[0].document.page_content.split())[:180]
            if top_k_report.results
            else ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose holdout retrieval failures (A–E)")
    parser.add_argument(
        "--dataset",
        default="eval/questions_graham_holdout.jsonl",
        help="JSONL evaluation set",
    )
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument(
        "--wide-k",
        type=int,
        default=10,
        help="Wide rank window used to separate pool miss (A) from rank miss (B)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path (default: eval/results/diagnose-holdout-<ts>.json)",
    )
    args = parser.parse_args()

    settings = project_settings()
    load_dotenv(settings.root / ".env")
    dataset = Path(args.dataset)
    if not dataset.is_file():
        dataset = settings.root / args.dataset
    cases = _load_cases(dataset)

    store = open_store(settings, build_embeddings())
    details: list[dict] = []
    buckets: Counter[str] = Counter()
    knowledge_hits = 0
    knowledge_total = 0

    print(f"Dataset: {dataset}")
    print(f"Top-K={args.top_k}  Wide-K={args.wide_k}")
    print("-" * 72)

    for row in cases:
        case_id = str(row.get("id", ""))
        question = str(row.get("question", "")).strip()
        expected_sources = [str(s) for s in row.get("expected_sources", [])]
        expect_abstention = bool(row.get("expect_abstention", False))

        # One wide retrieve, then slice Top-K for primary metrics. Avoids paying
        # a full second hybrid+rerank pass just for rank diagnostics.
        wide_k = max(args.wide_k, args.top_k)
        wide_report = retrieve_report(store, question, wide_k)
        top_results = wide_report.results[: args.top_k]
        top_evidence = assess_evidence(
            top_results,
            candidate_count=wide_report.evidence.candidate_count,
            reranker_provider=wide_report.evidence.reranker_provider,
            reranker_enabled=wide_report.evidence.reranker_enabled,
            rerank_ms=wide_report.evidence.rerank_ms,
            reranker_error=wide_report.evidence.reranker_error,
        )
        top_report = RetrievalReport(
            results=top_results,
            evidence=top_evidence,
            vector_ms=wide_report.vector_ms,
            rerank_ms=wide_report.rerank_ms,
            total_ms=wide_report.total_ms,
            retry_count=wide_report.retry_count,
            retry_query=wide_report.retry_query,
            context_results=wide_report.context_results,
        )
        classified = classify_case(
            expect_abstention=expect_abstention,
            expected_sources=expected_sources,
            top_k_report=top_report,
            wide_report=wide_report,
            top_k=args.top_k,
        )
        buckets[classified["bucket"]] += 1
        if not expect_abstention:
            knowledge_total += 1
            if classified["top_rank"] is not None:
                knowledge_hits += 1

        detail = {
            "id": case_id,
            "question": question,
            "expected_sources": expected_sources,
            "expect_abstention": expect_abstention,
            **classified,
        }
        details.append(detail)
        print(
            f"{case_id:12} [{classified['bucket']}] "
            f"ev={classified['evidence_status']:12} "
            f"score={classified['top_score']!s:>6} "
            f"rank={classified['top_rank']!s:>4} "
            f"wide={classified['wide_rank']!s:>4} "
            f"retry={classified['retry_count']} "
            f"| {classified['note']}"
        )

    recall = (knowledge_hits / knowledge_total) if knowledge_total else None
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "top_k": args.top_k,
        "wide_k": args.wide_k,
        "knowledge_source_hits": knowledge_hits,
        "knowledge_total": knowledge_total,
        "knowledge_recall_at_k": round(recall, 4) if recall is not None else None,
        "bucket_counts": dict(buckets),
        "cases": details,
    }

    out = Path(args.output) if args.output else None
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = settings.root / "eval" / "results" / f"diagnose-holdout-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    print(
        f"Knowledge Recall@{args.top_k}: "
        f"{knowledge_hits}/{knowledge_total}"
        + (f" = {recall:.2%}" if recall is not None else "")
    )
    print("Buckets:", dict(buckets))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
