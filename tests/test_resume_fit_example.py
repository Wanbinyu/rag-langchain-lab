from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.loaders import load_documents_report


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "resume_fit_ai" / "knowledge_base"
DATASET = ROOT / "eval" / "questions_resume_fit_ai_transfer_v1.jsonl"
FROZEN_DATASET = ROOT / "eval" / "questions_resume_fit_ai_frozen_v1.jsonl"
FROZEN_MANIFEST = ROOT / "eval" / "frozen" / "resume_fit_ai_frozen_v1_manifest.json"
ANSWER_DATASET = ROOT / "eval" / "questions_resume_fit_ai_answers_v1.jsonl"


def _cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_resume_fit_example_is_self_contained_and_source_consistent() -> None:
    documents, report = load_documents_report(CORPUS)
    sources = {str(document.metadata["source"]) for document in documents}
    cases = _cases()

    assert report.status == "ok"
    assert len(documents) == 5
    assert len(cases) >= 20
    assert len({str(case["id"]) for case in cases}) == len(cases)
    assert any(bool(case.get("expect_abstention")) for case in cases)
    for case in cases:
        expected_sources = {str(value) for value in case.get("expected_sources", [])}
        assert expected_sources <= sources
        assert bool(expected_sources) != bool(case.get("expect_abstention"))


def test_resume_fit_example_does_not_publish_common_secret_shapes() -> None:
    corpus_text = "\n".join(path.read_text(encoding="utf-8") for path in CORPUS.glob("*.md"))
    forbidden = (
        r"\bsk-[A-Za-z0-9_-]{12,}",
        r"BEGIN PRIVATE KEY",
        r"ADMIN_STATS_TOKEN\s*=",
        r"OPENAI_API_KEY\s*=",
        r"DEEPSEEK_API_KEY\s*=",
        r"QWEN_API_KEY\s*=",
    )
    assert not any(re.search(pattern, corpus_text, re.IGNORECASE) for pattern in forbidden)


def test_resume_fit_frozen_set_is_locked_and_disjoint_from_transfer() -> None:
    frozen_text = FROZEN_DATASET.read_text(encoding="utf-8")
    frozen_bytes = frozen_text.encode("utf-8")
    frozen = [json.loads(line) for line in frozen_text.splitlines() if line]
    transfer_questions = {str(case["question"]) for case in _cases()}
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    sources = {path.name for path in CORPUS.glob("*.md")}

    assert len(frozen) == manifest["case_count"] == 30
    assert len({str(case["id"]) for case in frozen}) == len(frozen)
    assert sum(not bool(case["expect_abstention"]) for case in frozen) == 20
    assert sum(bool(case["expect_abstention"]) for case in frozen) == 10
    assert not ({str(case["question"]) for case in frozen} & transfer_questions)
    assert hashlib.sha256(frozen_bytes).hexdigest() == manifest["sha256"]
    for case in frozen:
        assert set(case.get("expected_sources", [])) <= sources


def test_resume_fit_answer_audit_is_small_and_has_rubrics() -> None:
    rows = [
        json.loads(line)
        for line in ANSWER_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rubric = json.loads(
        (ROOT / "eval" / "answer_keywords_resume_fit_ai_answers_v1.json").read_text(
            encoding="utf-8"
        )
    )["cases"]
    assert 10 <= len(rows) <= 15
    assert len({str(case["id"]) for case in rows}) == len(rows)
    assert set(rubric) == {str(case["id"]) for case in rows}
    assert all(
        rubric[str(case["id"])]["required_groups"]
        for case in rows
        if not case["expect_abstention"]
    )
    assert all(
        not rubric[str(case["id"])]["required_groups"]
        for case in rows
        if case["expect_abstention"]
    )
