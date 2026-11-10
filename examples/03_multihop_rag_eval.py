"""Run an isolated, reproducible MultiHop-RAG external benchmark.

The upstream corpus is downloaded into ``_imports/`` (gitignored), converted
to plain Markdown for the project's existing loader, and indexed in a
dedicated Chroma directory.  It never changes the primary ``data/`` corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.cli import build_embeddings, print_index_report
from src.config import Settings, project_settings
from src.evaluation import run_evaluation
from src.indexing import sync_index


UPSTREAM_REPOSITORY = "https://github.com/yixuantt/MultiHop-RAG"
UPSTREAM_DATASET = "yixuantt/MultiHopRAG"
UPSTREAM_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
UPSTREAM_LICENSE = "ODC-BY-1.0"
RAW_FILES = {
    "MultiHopRAG.json": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    "corpus.json": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
}
QUESTION_TYPES = (
    "comparison_query",
    "inference_query",
    "temporal_query",
    "null_query",
)
DEFAULT_SEED = "multihop-rag-external-v1"
DEFAULT_PER_TYPE = 25

IMPORT_ROOT = ROOT / "_imports" / "multihop-rag"
CORPUS_DIR = IMPORT_ROOT / "knowledge_base"
DATASET_PATH = IMPORT_ROOT / "questions_multihop_rag_100_v1.jsonl"
MANIFEST_PATH = IMPORT_ROOT / "manifest.json"
RETRIEVAL_SUMMARY_PATH = ROOT / "eval" / "reports" / "multihop_rag_100_v1.json"
ANSWER_SUMMARY_PATH = ROOT / "eval" / "reports" / "multihop_rag_100_answers_v1.json"

ISOLATED_ENV = {
    # A domain glossary must not leak from the Graham corpus into this test.
    "QUERY_GLOSSARY_PATH": "off",
    "QUERY_EXPANSION_ENABLED": "0",
    # The official benchmark attaches title, publisher and publication time to
    # every node.  Our generated heading carries those fields so the existing
    # generic indexer can repeat them on every chunk without adding a PDF/JSON
    # loader to the product path.
    "INDEX_INCLUDE_METADATA": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_url(endpoint: str, filename: str) -> str:
    base = endpoint.rstrip("/")
    return (
        f"{base}/datasets/{UPSTREAM_DATASET}/resolve/"
        f"{UPSTREAM_REVISION}/{filename}"
    )


def ensure_raw_files(import_root: Path = IMPORT_ROOT) -> None:
    """Download the pinned upstream files and verify their published bytes."""
    import_root.mkdir(parents=True, exist_ok=True)
    configured = os.getenv("HF_ENDPOINT", "https://huggingface.co").strip()
    endpoints = tuple(dict.fromkeys((configured, "https://hf-mirror.com")))

    for filename, expected_hash in RAW_FILES.items():
        target = import_root / filename
        if target.is_file():
            actual_hash = _sha256(target)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{target} 校验失败：期望 {expected_hash}，实际 {actual_hash}。"
                )
            continue

        errors: list[str] = []
        for endpoint in endpoints:
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                request = urllib.request.Request(
                    _download_url(endpoint, filename),
                    headers={"User-Agent": "rag-langchain-lab/1.0"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    temporary.write_bytes(response.read())
                actual_hash = _sha256(temporary)
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"SHA-256 不匹配：期望 {expected_hash}，实际 {actual_hash}"
                    )
                temporary.replace(target)
                break
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                temporary.unlink(missing_ok=True)
                errors.append(f"{endpoint}: {type(error).__name__}")
        else:
            raise RuntimeError(f"无法下载 {filename}（{'；'.join(errors)}）")


def source_name(url: str) -> str:
    """Return a stable local source path for one upstream article URL."""
    identity = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"documents/doc-{identity}.md"


def _markdown_document(item: dict[str, object]) -> str:
    title = str(item.get("title") or "Untitled").strip()
    publisher = str(item.get("source") or "Unknown").strip()
    published_at = str(item.get("published_at") or "Unknown").strip()
    category = str(item.get("category") or "Unknown").strip()
    indexed_title = (
        f"{title} | Publisher: {publisher} | Published at: {published_at} "
        f"| Category: {category}"
    )
    metadata = (
        ("Source", item.get("source")),
        ("Author", item.get("author")),
        ("Published at", item.get("published_at")),
        ("Category", item.get("category")),
        ("URL", item.get("url")),
    )
    lines = [f"# {indexed_title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in metadata if value)
    lines.extend(("", str(item.get("body") or "").strip(), ""))
    return "\n".join(lines)


def _rank_key(seed: str, row: dict[str, object]) -> str:
    question = str(row.get("query") or "")
    return hashlib.sha256(f"{seed}\0{question}".encode("utf-8")).hexdigest()


def prepare_benchmark(
    query_path: Path,
    corpus_path: Path,
    corpus_dir: Path,
    dataset_path: Path,
    *,
    per_type: int = DEFAULT_PER_TYPE,
    seed: str = DEFAULT_SEED,
) -> dict[str, object]:
    """Convert the full corpus and create a deterministic balanced sample."""
    if per_type < 1:
        raise ValueError("per_type 必须大于 0")
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not isinstance(corpus, list):
        raise ValueError("上游 MultiHop-RAG 文件必须是 JSON 数组")

    corpus_dir.mkdir(parents=True, exist_ok=True)
    source_by_url: dict[str, str] = {}
    for document in corpus:
        url = str(document.get("url") or "").strip()
        if not url or url in source_by_url:
            raise ValueError("corpus.json 包含空 URL 或重复 URL")
        source = source_name(url)
        source_by_url[url] = source
        target = corpus_dir / source
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _markdown_document(document)
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")

    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in queries:
        by_type[str(row.get("question_type"))].append(row)
    missing_types = [name for name in QUESTION_TYPES if len(by_type[name]) < per_type]
    if missing_types:
        raise ValueError(f"题量不足：{', '.join(missing_types)}")

    selected: list[dict[str, object]] = []
    for question_type in QUESTION_TYPES:
        ranked = sorted(by_type[question_type], key=lambda row: _rank_key(seed, row))
        selected.extend(ranked[:per_type])

    cases: list[dict[str, object]] = []
    for row in selected:
        question = str(row["query"]).strip()
        question_type = str(row["question_type"])
        answer = str(row["answer"]).strip()
        expected_sources: list[str] = []
        for evidence in row.get("evidence_list", []):
            url = str(evidence.get("url") or "").strip()
            if url not in source_by_url:
                raise ValueError(f"证据 URL 未出现在 corpus.json：{url}")
            source = source_by_url[url]
            if source not in expected_sources:
                expected_sources.append(source)
        expect_abstention = question_type == "null_query"
        if bool(expected_sources) == expect_abstention:
            raise ValueError(f"题目证据与类型不一致：{question}")
        case_id = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        cases.append(
            {
                "id": f"multihop-{case_id}",
                "category": question_type,
                "question": question,
                "expected_sources": expected_sources,
                # Official answers are usually short entities, dates or names.
                # Exact containment is intentionally reported as a strict proxy.
                "expected_keywords": [] if expect_abstention else [answer],
                "expect_abstention": expect_abstention,
                "reference_answer": answer,
            }
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_text = "".join(
        json.dumps(case, ensure_ascii=False) + "\n" for case in cases
    )
    dataset_path.write_text(dataset_text, encoding="utf-8")
    return {
        "corpus_documents": len(corpus),
        "upstream_queries": len(queries),
        "sample_cases": len(cases),
        "sample_per_type": per_type,
        "sample_type_counts": dict(Counter(case["category"] for case in cases)),
        "sample_sha256": hashlib.sha256(dataset_text.encode("utf-8")).hexdigest(),
        "seed": seed,
    }


def profile_settings(root: Path = ROOT) -> Settings:
    base = project_settings(root)
    return Settings(
        root=base.root,
        data_dir=CORPUS_DIR,
        chroma_dir=root / "work" / "multihop-rag" / "chroma",
        collection_name="multihop_rag_external_v1",
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        top_k=5,
    )


@contextmanager
def isolated_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in ISOLATED_ENV}
    os.environ.update(ISOLATED_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def multihop_retrieval_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    """Add complete-evidence metrics missing from ordinary single-source QA."""
    answerable = [case for case in cases if not case["expect_abstention"]]
    source_recalls: list[float] = []
    complete_hits: list[bool] = []
    any_hits: list[bool] = []
    by_type: dict[str, list[float]] = defaultdict(list)
    for case in answerable:
        expected = set(case["expected_sources"])
        retrieved = set(case["retrieved_sources"])
        recall = len(expected & retrieved) / len(expected) if expected else 0.0
        source_recalls.append(recall)
        complete_hits.append(expected <= retrieved)
        any_hits.append(bool(expected & retrieved))
        by_type[str(case["category"])].append(recall)
    return {
        "answerable_cases": len(answerable),
        "any_evidence_hit_rate": round(sum(any_hits) / len(any_hits), 4),
        "complete_evidence_hit_rate": round(sum(complete_hits) / len(complete_hits), 4),
        "mean_evidence_source_recall": round(sum(source_recalls) / len(source_recalls), 4),
        "mean_evidence_source_recall_by_type": {
            name: round(sum(values) / len(values), 4)
            for name, values in sorted(by_type.items())
        },
    }


def _binary_stance(answer: str | None) -> str:
    """Classify the explicit conclusion of a bilingual yes/no answer."""
    if not answer:
        return "unknown"
    first_line = next((line.strip() for line in answer.splitlines() if line.strip()), "")
    conclusion = re.sub(r"^(结论|answer|conclusion)\s*[:：]\s*", "", first_line.casefold())
    unknown_markers = (
        "根据现有资料无法回答",
        "无法回答",
        "无法判断",
        "insufficient information",
        "cannot answer",
        "cannot determine",
    )
    if any(marker in conclusion for marker in unknown_markers):
        return "unknown"
    negative_markers = (
        "不被资料支持",
        "不支持",
        "不能证实",
        "无法证实",
        "并非",
        "不是",
        "没有",
        "不一致",
        "否",
    )
    if conclusion == "no" or conclusion.startswith("no,") or any(
        marker in conclusion for marker in negative_markers
    ):
        return "no"
    positive_markers = (
        "资料支持",
        "可以确认",
        "能够确认",
        "一致",
        "相同",
        "确实",
        "是的",
    )
    if conclusion == "yes" or conclusion.startswith("yes,") or any(
        marker in conclusion for marker in positive_markers
    ):
        return "yes"
    return "unknown"


def _reference_match(reference: str, answer: str | None) -> bool:
    if not answer:
        return False
    expected = reference.casefold().strip()
    if expected in {"yes", "no"}:
        return _binary_stance(answer) == expected
    actual = answer.casefold()
    if expected in actual:
        return True
    # Entity answers sometimes replace English conjunctions with Chinese ones.
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", expected)
        if token not in {"a", "an", "and", "the"}
    ]
    return bool(tokens) and all(token in actual for token in tokens)


def multihop_answer_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    """Score short official answers with a bilingual deterministic proxy."""
    scored: list[tuple[str, bool]] = []
    binary_cases = 0
    for case in cases:
        category = str(case["category"])
        if case["expect_abstention"]:
            correct = bool(case.get("abstention_hit"))
        else:
            reference = str(case.get("reference_answer") or "").strip()
            binary_cases += reference.casefold() in {"yes", "no"}
            correct = _reference_match(reference, case.get("answer"))
        scored.append((category, correct))
    by_type: dict[str, list[bool]] = defaultdict(list)
    for category, correct in scored:
        by_type[category].append(correct)
    return {
        "scored_cases": len(scored),
        "correct_count": sum(correct for _, correct in scored),
        "correct_rate": round(sum(correct for _, correct in scored) / len(scored), 4),
        "binary_reference_cases": binary_cases,
        "method": "bilingual_conclusion_stance_or_reference_entity_containment",
        "correct_rate_by_type": {
            name: round(sum(values) / len(values), 4)
            for name, values in sorted(by_type.items())
        },
    }


def publish_summary(
    report: dict[str, object],
    preparation: dict[str, object],
    *,
    top_k: int,
    with_answers: bool,
) -> Path:
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": "multihop-rag-external-100-v1",
        "claim_boundary": (
            "Deterministic public external sample. It is suitable for baseline comparison "
            "but ceases to be blind if used for tuning."
        ),
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "dataset": UPSTREAM_DATASET,
            "revision": UPSTREAM_REVISION,
            "license": UPSTREAM_LICENSE,
            "raw_sha256": RAW_FILES,
        },
        "preparation": preparation,
        "configuration": {
            "collection": "multihop_rag_external_v1",
            "top_k": top_k,
            "answers_generated": with_answers,
            "isolated_from_primary_corpus": True,
        },
        "summary": report["summary"],
        "multihop_summary": multihop_retrieval_summary(report["cases"]),
        "multihop_answer_summary": (
            multihop_answer_summary(report["cases"]) if with_answers else None
        ),
    }
    summary_path = ANSWER_SUMMARY_PATH if with_answers else RETRIEVAL_SUMMARY_PATH
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行隔离的 MultiHop-RAG 100 题公开挑战集")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--per-type", type=int, default=DEFAULT_PER_TYPE)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--with-answers", action="store_true", help="调用 Chat API 生成答案")
    parser.add_argument("--publish-summary", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = create_parser().parse_args()
    top_k = min(max(args.top_k, 1), 10)
    load_dotenv(ROOT / ".env")

    ensure_raw_files()
    preparation = prepare_benchmark(
        IMPORT_ROOT / "MultiHopRAG.json",
        IMPORT_ROOT / "corpus.json",
        CORPUS_DIR,
        DATASET_PATH,
        per_type=args.per_type,
        seed=args.seed,
    )
    manifest = {
        "upstream_revision": UPSTREAM_REVISION,
        "raw_sha256": RAW_FILES,
        **preparation,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preparation, ensure_ascii=False, indent=2))

    settings = profile_settings()
    print("同步隔离的 MultiHop-RAG 索引...")
    index_report = sync_index(settings, build_embeddings, force=args.force_index)
    print_index_report(index_report)

    with isolated_environment():
        report, detailed_path = run_evaluation(
            settings,
            top_k=top_k,
            include_answers=args.with_answers,
            dataset_path=DATASET_PATH,
        )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            multihop_retrieval_summary(report["cases"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.with_answers:
        print(json.dumps(multihop_answer_summary(report["cases"]), ensure_ascii=False, indent=2))
    print(f"详细报告: {detailed_path}")
    if args.publish_summary:
        if int(report["summary"].get("answer_error_count", 0)):
            raise SystemExit("答案调用存在错误，不发布不完整摘要。")
        summary_path = publish_summary(
            report,
            preparation,
            top_k=top_k,
            with_answers=args.with_answers,
        )
        print(f"可提交摘要: {summary_path}")


if __name__ == "__main__":
    main()
