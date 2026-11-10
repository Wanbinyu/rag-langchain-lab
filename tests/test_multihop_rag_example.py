from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "03_multihop_rag_eval.py"
SPEC = importlib.util.spec_from_file_location("multihop_rag_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _document(index: int) -> dict[str, object]:
    return {
        "title": f"Document {index}",
        "author": "Example Author",
        "source": "Example",
        "published_at": "2024-01-01T00:00:00+00:00",
        "category": "technology",
        "url": f"https://example.test/{index}",
        "body": f"Body {index}",
    }


def test_preparation_is_balanced_deterministic_and_source_consistent(tmp_path: Path) -> None:
    corpus = [_document(index) for index in range(8)]
    queries = []
    for question_type in MODULE.QUESTION_TYPES:
        for index in range(3):
            is_null = question_type == "null_query"
            queries.append(
                {
                    "query": f"{question_type} question {index}",
                    "answer": "Insufficient information." if is_null else f"answer {index}",
                    "question_type": question_type,
                    "evidence_list": [] if is_null else [
                        {"url": corpus[index]["url"]},
                        {"url": corpus[index + 1]["url"]},
                    ],
                }
            )

    query_path = tmp_path / "queries.json"
    corpus_path = tmp_path / "corpus.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    corpus_dir = tmp_path / "knowledge_base"
    dataset_path = tmp_path / "questions.jsonl"

    first = MODULE.prepare_benchmark(
        query_path,
        corpus_path,
        corpus_dir,
        dataset_path,
        per_type=2,
        seed="fixed",
    )
    first_text = dataset_path.read_text(encoding="utf-8")
    second = MODULE.prepare_benchmark(
        query_path,
        corpus_path,
        corpus_dir,
        dataset_path,
        per_type=2,
        seed="fixed",
    )
    cases = [json.loads(line) for line in first_text.splitlines()]
    sources = {
        path.relative_to(corpus_dir).as_posix()
        for path in corpus_dir.rglob("*.md")
    }

    assert first == second
    assert dataset_path.read_text(encoding="utf-8") == first_text
    assert first["sample_cases"] == 8
    assert set(first["sample_type_counts"].values()) == {2}
    assert len(sources) == len(corpus)
    first_document = next(corpus_dir.rglob("*.md")).read_text(encoding="utf-8")
    assert "Publisher: Example" in first_document.splitlines()[0]
    assert "Published at: 2024-01-01" in first_document.splitlines()[0]
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        expected = set(case["expected_sources"])
        assert expected <= sources
        assert bool(expected) != bool(case["expect_abstention"])


def test_multihop_summary_requires_all_evidence_sources() -> None:
    cases = [
        {
            "category": "comparison_query",
            "expect_abstention": False,
            "expected_sources": ["a.md", "b.md"],
            "retrieved_sources": ["a.md", "other.md"],
        },
        {
            "category": "inference_query",
            "expect_abstention": False,
            "expected_sources": ["c.md", "d.md"],
            "retrieved_sources": ["d.md", "c.md"],
        },
        {
            "category": "null_query",
            "expect_abstention": True,
            "expected_sources": [],
            "retrieved_sources": ["other.md"],
        },
    ]

    summary = MODULE.multihop_retrieval_summary(cases)

    assert summary["answerable_cases"] == 2
    assert summary["any_evidence_hit_rate"] == 1.0
    assert summary["complete_evidence_hit_rate"] == 0.5
    assert summary["mean_evidence_source_recall"] == 0.75


def test_answer_summary_handles_bilingual_binary_and_entity_answers() -> None:
    cases = [
        {
            "category": "comparison_query",
            "expect_abstention": False,
            "reference_answer": "no",
            "answer": "结论：没有。两篇文章的观点并不相同。",
        },
        {
            "category": "temporal_query",
            "expect_abstention": False,
            "reference_answer": "Yes",
            "answer": "结论：资料支持该时间顺序。",
        },
        {
            "category": "inference_query",
            "expect_abstention": False,
            "reference_answer": "Taylor Swift and Travis Kelce",
            "answer": "结论：相关人物是 Taylor Swift 与 Travis Kelce。",
        },
        {
            "category": "null_query",
            "expect_abstention": True,
            "reference_answer": "Insufficient information.",
            "answer": "根据现有资料无法回答",
            "abstention_hit": True,
        },
    ]

    summary = MODULE.multihop_answer_summary(cases)

    assert summary["correct_count"] == 4
    assert summary["correct_rate"] == 1.0
    assert summary["binary_reference_cases"] == 2
