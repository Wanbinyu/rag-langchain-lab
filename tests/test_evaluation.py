from __future__ import annotations

import unittest
import json
from pathlib import Path

from src.chains import build_answer_prompt
from src.evaluation import (
    CaseResult,
    _answer_keyword_group_scores,
    _answer_point_scores,
    assess_answer_structure,
    keyword_groups_for_question,
    load_cases,
    summarize,
)


class EvaluationSummaryTests(unittest.TestCase):
    def test_summary_calculates_retrieval_and_answer_metrics(self) -> None:
        results = [
            CaseResult(
                id="first", category="test", question="q1", expected_sources=("a.md",),
                retrieved_sources=("a.md",), first_expected_rank=1, source_hit=True,
                reciprocal_rank=1.0, top_score=0.9, retrieval_ms=10.0, answer="answer",
                answer_ms=100.0, answer_keyword_hit=True, abstention_hit=None, answer_error=None,
            ),
            CaseResult(
                id="second", category="test", question="q2", expected_sources=("b.md",),
                retrieved_sources=("other.md", "b.md"), first_expected_rank=2, source_hit=True,
                reciprocal_rank=0.5, top_score=0.7, retrieval_ms=20.0, answer="answer",
                answer_ms=200.0, answer_keyword_hit=False, abstention_hit=None, answer_error=None,
            ),
            CaseResult(
                id="negative", category="test", question="q3", expected_sources=(),
                retrieved_sources=("a.md",), first_expected_rank=None, source_hit=None,
                reciprocal_rank=None, top_score=0.4, retrieval_ms=30.0,
                answer="根据现有资料无法回答", answer_ms=150.0, answer_keyword_hit=None,
                abstention_hit=True, answer_error=None,
            ),
        ]
        summary = summarize(results)
        self.assertEqual(summary["total_cases"], 3)
        self.assertEqual(summary["hit_rate_at_k"], 1.0)
        self.assertEqual(summary["mrr"], 0.75)
        self.assertEqual(summary["keyword_hit_rate"], 0.5)
        self.assertEqual(summary["abstention_hit_rate"], 1.0)
        self.assertEqual(summary["retrieval_p95_ms"], 30.0)

    def test_graham_answer_points_are_loaded_from_sidecar(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = load_cases(root / "eval" / "questions_graham_85.jsonl")
        positive = [case for case in cases if not case.expect_abstention]
        self.assertEqual(len(positive), 65)
        self.assertEqual(len(positive[0].answer_points), 3)
        self.assertTrue(all(case.answer_points for case in positive))

    def test_answer_point_proxy_detects_covered_and_missing_points(self) -> None:
        scores, covered, coverage = _answer_point_scores(
            ("用户真正想要的东西", "尽快上线并获得真实反馈"),
            "先做出用户真正想要的东西，再尽快上线获得真实反馈并迭代。",
        )
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(covered))
        self.assertGreater(coverage or 0.0, 0.35)

    def test_answer_point_proxy_scores_refusal_as_zero(self) -> None:
        scores, covered, coverage = _answer_point_scores(
            ("用户真正想要的东西",),
            "根据现有资料无法回答",
        )
        self.assertEqual(scores, (0.0,))
        self.assertEqual(covered, (False,))
        self.assertEqual(coverage, 0.0)

    def test_normalized_answer_point_proxy_handles_audited_synonyms(self) -> None:
        raw_scores, raw_covered, _ = _answer_point_scores(
            ("Default Alive 表示按当前烧钱速度，公司能在资金耗尽前达到盈亏平衡。",),
            "公司可以依靠现有资金实现盈利。",
        )
        normalized_scores, normalized_covered, normalized_coverage = _answer_point_scores(
            ("Default Alive 表示按当前烧钱速度，公司能在资金耗尽前达到盈亏平衡。",),
            "公司可以依靠现有资金实现盈利。",
            normalized=True,
        )
        self.assertFalse(raw_covered[0])
        self.assertGreater(normalized_scores[0], raw_scores[0])
        self.assertTrue(normalized_covered[0])
        self.assertGreater(normalized_coverage or 0.0, 0.35)

    def test_graham_keyword_groups_are_loaded_from_sidecar(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = load_cases(root / "eval" / "questions_graham_150.jsonl")
        positive = [case for case in cases if not case.expect_abstention]
        self.assertEqual(len(positive), 125)
        self.assertTrue(all(case.answer_keyword_groups for case in positive))
        self.assertEqual(len(cases[0].answer_keyword_groups), 3)

    def test_keyword_group_proxy_accepts_an_audited_alias(self) -> None:
        scores, covered, coverage = _answer_keyword_group_scores(
            (("无法规模化", "不能规模化"), ("少数用户", "一小批用户")),
            "早期先做不能规模化的事情，让一小批用户真正喜欢产品。",
        )
        self.assertEqual(scores, (1.0, 1.0))
        self.assertEqual(covered, (True, True))
        self.assertEqual(coverage, 1.0)

    def test_answer_structure_review_combines_coverage_and_evidence(self) -> None:
        groups = (("用户真正想要",), ("快速迭代",))
        complete = assess_answer_structure(
            groups,
            "结论：做出用户真正想要的东西，并快速迭代。",
            "sufficient",
        )
        self.assertEqual(complete.status, "sufficient")
        self.assertEqual(complete.covered_count, 2)

        partial = assess_answer_structure(
            groups,
            "结论：做出用户真正想要的东西。",
            "sufficient",
        )
        self.assertEqual(partial.status, "review")
        self.assertEqual(partial.reason_code, "missing_required_groups")
        self.assertEqual(partial.missing_groups, (("快速迭代",),))

        insufficient = assess_answer_structure(groups, "根据现有资料无法回答", "insufficient")
        self.assertEqual(insufficient.status, "insufficient")
        self.assertEqual(insufficient.reason_code, "evidence_insufficient")

    def test_dashboard_can_lookup_only_known_rubrics(self) -> None:
        root = Path(__file__).resolve().parents[1]
        known = keyword_groups_for_question(root, "如何找到好的创业想法？")
        unknown = keyword_groups_for_question(root, "这是一个没有评测标注的问题？")
        self.assertTrue(known)
        self.assertEqual(unknown, ())

    def test_holdout_dataset_is_separate_and_rubric_is_loaded(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = load_cases(root / "eval" / "questions_graham_holdout.jsonl")
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(not case.expect_abstention for case in cases), 15)
        self.assertEqual(sum(bool(case.answer_keyword_groups) for case in cases), 15)

    def test_answer_prompt_contains_structure_contract_without_gold_terms(self) -> None:
        prompt = build_answer_prompt().format_messages(context="资料", question="问题")[0].content
        self.assertIn("结论", prompt)
        self.assertIn("核心要点", prompt)
        self.assertIn("证据边界", prompt)
        self.assertNotIn("Default Alive", prompt)

    def test_graham_keyword_sidecar_is_complete_and_utf8_clean(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rubric_path = root / "eval" / "answer_keywords_graham_150.json"
        dataset_path = root / "eval" / "questions_graham_150.jsonl"
        payload = json.loads(rubric_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["cases"]), 150)
        self.assertNotIn("?", rubric_path.read_text(encoding="utf-8"))
        dataset_rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertFalse(
            any(
                "?" in json.dumps(row.get("expected_keywords", []), ensure_ascii=False)
                or "?" in str(row.get("annotation_note", ""))
                for row in dataset_rows
            )
        )
        self.assertTrue(
            all(
                not entry["required_groups"]
                for entry in payload["cases"].values()
                if entry["expect_abstention"]
            )
        )

    def test_partial_answer_with_abstention_caveat_keeps_covered_points(self) -> None:
        scores, covered, coverage = _answer_point_scores(
            ("感到害怕",),
            "资料说明创业想法可能是可怕的，但无法回答更具体的方法。",
            normalized=True,
        )
        self.assertGreater(scores[0], 0.0)
        self.assertTrue(covered[0])
        self.assertGreater(coverage or 0.0, 0.35)

    def test_partial_answer_caveat_is_not_counted_as_full_refusal(self) -> None:
        result = CaseResult(
            id="partial",
            category="test",
            question="q",
            expected_sources=("a.md",),
            retrieved_sources=("a.md",),
            first_expected_rank=1,
            source_hit=True,
            reciprocal_rank=1.0,
            top_score=0.8,
            retrieval_ms=10.0,
            answer="已给出部分证据，但无法回答更具体的方法。",
            answer_ms=1.0,
            answer_keyword_hit=None,
            abstention_hit=None,
            answer_error=None,
        )
        self.assertEqual(summarize([result])["answer_refusal_count"], 0)


if __name__ == "__main__":
    unittest.main()
