from __future__ import annotations

import json
import re
from pathlib import Path

from src.loaders import load_documents_report


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "resume_fit_ai" / "knowledge_base"
DATASET = ROOT / "eval" / "questions_resume_fit_ai_transfer_v1.jsonl"


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
