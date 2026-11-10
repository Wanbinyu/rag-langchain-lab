from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.glossary import configure_glossary, reset_glossary_cache
from src.retrieval import (
    EvidenceAssessment,
    RetrievalReport,
    RetrievalResult,
    _diversify_results,
    _expand_parent_neighbor_context,
    _expand_query,
    _is_fragment_text,
    _repair_fragment_results,
    _rerank_query,
    _retry_query,
    _rule_rewrite_query,
    _rule_topic_terms,
    _rrf_fuse,
    _should_retry,
    _source_intro_candidates,
    assess_evidence,
)

GRAHAM_GLOSSARY = (
    Path(__file__).resolve().parents[1] / "eval" / "glossaries" / "graham.json"
)


def _result(
    score: float,
    rerank_score: float | None = None,
    source: str = "guide.md",
    content: str | None = None,
) -> RetrievalResult:
    # Keep body text long enough that fragment detection does not fire.
    body = content or (
        "This is a substantive evidence paragraph used by unit tests so fragment "
        "repair and heading detection do not treat the fixture as a short title."
    )
    return RetrievalResult(
        document=Document(page_content=body, metadata={"source": source}),
        score=score,
        elapsed_ms=1.0,
        rerank_score=rerank_score,
        reranker_provider="cross_encoder" if rerank_score is not None else "disabled",
    )


class EvidenceAssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_glossary(None)
        reset_glossary_cache()

    def tearDown(self) -> None:
        reset_glossary_cache()
        configure_glossary(None)

    def _graham_glossary_env(self) -> dict[str, str]:
        return {"QUERY_GLOSSARY_PATH": str(GRAHAM_GLOSSARY), "QUERY_EXPANSION_ENABLED": "1"}

    def test_query_expansion_is_inert_without_corpus_glossary(self) -> None:
        with patch.dict("os.environ", {"QUERY_GLOSSARY_PATH": "off"}, clear=False):
            reset_glossary_cache()
            self.assertEqual(_expand_query("如何找到好的创业想法？"), "如何找到好的创业想法？")
            self.assertEqual(_rule_topic_terms("创业者怎样从日常经历里发现值得做的产品？"), [])

    def test_query_expansion_adds_english_domain_terms(self) -> None:
        with patch.dict("os.environ", self._graham_glossary_env(), clear=False):
            expanded = _expand_query("如何找到好的创业想法？")
        self.assertIn("startup ideas", expanded)

    def test_query_expansion_keeps_unmatched_questions_unchanged(self) -> None:
        question = "这个天气问题不在知识库里"
        with patch.dict("os.environ", self._graham_glossary_env(), clear=False):
            self.assertEqual(_expand_query(question), question)

    def test_reranker_query_keeps_original_and_domain_terms(self) -> None:
        question = "如何找到好的创业想法？"
        with patch.dict(
            "os.environ",
            {**self._graham_glossary_env(), "RERANKER_USE_EXPANDED_QUERY": "1"},
            clear=False,
        ):
            expanded = _rerank_query(question)
        self.assertIn(question, expanded)
        self.assertIn("startup ideas", expanded)

    def test_cross_language_miss_topics_add_audited_english_review_terms(self) -> None:
        cases = (
            ("Paul Graham 如何看待“专家”和“新手”在创业中的差异？", "how to be an expert in a changing world"),
            ("为什么 Paul Graham 说“Ideas are worthless”？", "startups in 13 sentences"),
            ("Paul Graham 如何看待“声望”和“兴趣”之间的关系？", "how to do what you love"),
            ("Paul Graham 如何看待“产品迭代”？", "launch and iterate"),
            ("创业过程中如何避免被情绪影响决策？", "user feedback"),
            ("Paul Graham 如何看待“社会压力”对个人选择的影响？", "keep your identity small"),
        )
        with patch.dict("os.environ", self._graham_glossary_env(), clear=False):
            for question, expected_term in cases:
                with self.subTest(question=question):
                    self.assertIn(expected_term, _expand_query(question))

    def test_calibration_questions_get_rule_english_terms(self) -> None:
        cases = (
            ("创业者怎样从日常经历里发现值得做的产品？", "how to get startup ideas"),
            ("为什么初创团队有时要亲手给第一批客户搭建产品？", "do things that don't scale"),
            ("Default Alive 的现金跑道判断具体看什么？", "default alive"),
            ("什么样的创业构想会让创始人本能地退缩？", "frighteningly ambitious startup ideas"),
            ("创业时低成本运营给团队带来什么战略优势？", "ramen profitable"),
        )
        with patch.dict("os.environ", self._graham_glossary_env(), clear=False):
            for question, expected_term in cases:
                with self.subTest(question=question):
                    self.assertIn(expected_term, _rule_topic_terms(question))
                    rewritten = _rule_rewrite_query(question)
                    self.assertIsNotNone(rewritten)
                    assert rewritten is not None
                    self.assertIn(expected_term, rewritten)
                    self.assertTrue(rewritten.startswith(expected_term) or expected_term in rewritten)

    def test_rule_retry_query_prefers_english_lead_form(self) -> None:
        question = "创业者怎样从日常经历里发现值得做的产品？"
        with patch.dict(
            "os.environ",
            {
                **self._graham_glossary_env(),
                "RETRIEVAL_RETRY_ENABLED": "1",
                "RETRIEVAL_REWRITE_MODE": "rule",
            },
            clear=False,
        ):
            retry = _retry_query(question)
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertIn("how to get startup ideas", retry)
        # English terms lead; original question is retained after them.
        self.assertLess(retry.find("how to get startup ideas"), retry.find(question))

    def test_retry_can_be_disabled(self) -> None:
        with patch.dict(
            "os.environ",
            {"RETRIEVAL_RETRY_ENABLED": "0"},
            clear=False,
        ):
            self.assertIsNone(_retry_query("Default Alive 的现金跑道判断具体看什么？"))

    def _report(self, status: str, top_score: float | None) -> RetrievalReport:
        results = [_result(0.1, top_score if top_score is not None else 0.0)] if top_score is not None else []
        evidence = EvidenceAssessment(
            status=status,  # type: ignore[arg-type]
            top_score=top_score,
            score_gap=None,
            reason_code="test",
            reranker_enabled=True,
            reranker_provider="cross_encoder",
            candidate_count=3,
            rerank_ms=1.0,
        )
        return RetrievalReport(
            results=results,
            evidence=evidence,
            vector_ms=1.0,
            rerank_ms=1.0,
            total_ms=2.0,
        )

    def test_should_retry_skips_review_by_default(self) -> None:
        with patch.dict("os.environ", {"RETRIEVAL_RETRY_ON_REVIEW": "0"}, clear=False):
            self.assertFalse(
                _should_retry(self._report("review", 0.40), "english terms\nquestion")
            )

    def test_should_retry_insufficient_in_band(self) -> None:
        with patch.dict(
            "os.environ",
            {"RETRIEVAL_RETRY_HOPELESS": "0.12"},
            clear=False,
        ):
            self.assertTrue(
                _should_retry(self._report("insufficient", 0.20), "english terms\nquestion")
            )

    def test_should_retry_skips_hopeless_oos_scores(self) -> None:
        with patch.dict("os.environ", {"RETRIEVAL_RETRY_HOPELESS": "0.12"}, clear=False):
            self.assertFalse(
                _should_retry(self._report("insufficient", 0.05), "english terms\nquestion")
            )

    def test_should_retry_never_when_sufficient(self) -> None:
        self.assertFalse(
            _should_retry(self._report("sufficient", 0.70), "english terms\nquestion")
        )

    def test_empty_results_are_insufficient(self) -> None:
        assessment = assess_evidence(
            [],
            candidate_count=0,
            reranker_provider="cross_encoder",
            reranker_enabled=True,
            rerank_ms=0.0,
        )
        self.assertEqual(assessment.status, "insufficient")
        self.assertEqual(assessment.reason_code, "no_results")

    def test_strong_top_result_is_sufficient(self) -> None:
        assessment = assess_evidence(
            [_result(0.1, 0.86), _result(0.1, 0.42)],
            candidate_count=8,
            reranker_provider="cross_encoder",
            reranker_enabled=True,
            rerank_ms=12.0,
        )
        self.assertEqual(assessment.status, "sufficient")
        self.assertEqual(assessment.reason_code, "strong_relevance")
        self.assertEqual(assessment.score_gap, 0.44)

    def test_close_scores_require_review(self) -> None:
        assessment = assess_evidence(
            [_result(0.1, 0.7), _result(0.1, 0.68, "other.md")],
            candidate_count=8,
            reranker_provider="cross_encoder",
            reranker_enabled=True,
            rerank_ms=12.0,
        )
        self.assertEqual(assessment.status, "review")
        self.assertEqual(assessment.reason_code, "ambiguous_top_results")

    def test_low_score_is_insufficient(self) -> None:
        assessment = assess_evidence(
            [_result(0.1, 0.12)],
            candidate_count=8,
            reranker_provider="cross_encoder",
            reranker_enabled=True,
            rerank_ms=12.0,
        )
        self.assertEqual(assessment.status, "insufficient")
        self.assertEqual(assessment.reason_code, "low_relevance")

    def test_same_source_cluster_promotes_soft_floor_to_review(self) -> None:
        """Correct essay dominates top ranks between soft floor and review cut."""
        with patch.dict(
            "os.environ",
            {
                "RERANKER_REVIEW_THRESHOLD": "0.30",
                "RERANKER_SOFT_REVIEW_FLOOR": "0.28",
                "EVIDENCE_SAME_SOURCE_SUPPORT": "1",
            },
            clear=False,
        ):
            assessment = assess_evidence(
                [
                    _result(0.1, 0.29, "default-alive.md"),
                    _result(0.1, 0.288, "default-alive.md"),
                    _result(0.1, 0.285, "default-alive.md"),
                ],
                candidate_count=12,
                reranker_provider="cross_encoder",
                reranker_enabled=True,
                rerank_ms=8.0,
            )
        self.assertEqual(assessment.status, "review")
        self.assertEqual(assessment.reason_code, "same_source_support")

    def test_oos_low_mixed_sources_stay_insufficient(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "RERANKER_REVIEW_THRESHOLD": "0.30",
                "RERANKER_SOFT_REVIEW_FLOOR": "0.28",
                "EVIDENCE_SAME_SOURCE_SUPPORT": "1",
            },
            clear=False,
        ):
            assessment = assess_evidence(
                [
                    _result(0.1, 0.209, "a.md"),
                    _result(0.1, 0.149, "b.md"),
                    _result(0.1, 0.120, "c.md"),
                ],
                candidate_count=12,
                reranker_provider="cross_encoder",
                reranker_enabled=True,
                rerank_ms=8.0,
            )
        self.assertEqual(assessment.status, "insufficient")
        self.assertEqual(assessment.reason_code, "low_relevance")

    def test_calibrated_sufficient_threshold_default(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "RERANKER_SUFFICIENT_THRESHOLD": "0.52",
                "RERANKER_REVIEW_THRESHOLD": "0.30",
                "RERANKER_MIN_MARGIN": "0.04",
            },
            clear=False,
        ):
            assessment = assess_evidence(
                [_result(0.1, 0.55), _result(0.1, 0.30, "other.md")],
                candidate_count=8,
                reranker_provider="cross_encoder",
                reranker_enabled=True,
                rerank_ms=5.0,
            )
        self.assertEqual(assessment.status, "sufficient")

    def test_fragment_primary_blocks_full_sufficiency(self) -> None:
        heading = RetrievalResult(
            document=Document(
                page_content="9. Get ramen profitable.",
                metadata={"source": "essay.md", "chunk_index": 9},
            ),
            score=0.1,
            elapsed_ms=1.0,
            rerank_score=0.70,
            reranker_provider="cross_encoder",
        )
        with patch.dict(
            "os.environ",
            {"RERANKER_SUFFICIENT_THRESHOLD": "0.52", "RERANKER_REVIEW_THRESHOLD": "0.30"},
            clear=False,
        ):
            assessment = assess_evidence(
                [heading, _result(0.1, 0.20, "other.md")],
                candidate_count=8,
                reranker_provider="cross_encoder",
                reranker_enabled=True,
                rerank_ms=5.0,
            )
        self.assertEqual(assessment.status, "review")
        self.assertEqual(assessment.reason_code, "fragment_primary")

    def test_is_fragment_detects_numbered_heading(self) -> None:
        self.assertTrue(_is_fragment_text("9. Get ramen profitable."))
        self.assertTrue(_is_fragment_text("short"))
        self.assertFalse(
            _is_fragment_text(
                "Get ramen profitable means making just enough to pay the founders' living expenses "
                "so the company can survive without raising money immediately."
            )
        )

    def test_fragment_repair_replaces_heading_with_body(self) -> None:
        class FakeCollection:
            name = "test-fragment-repair"

            @staticmethod
            def count() -> int:
                return 2

        class FakeStore:
            _collection = FakeCollection()

            @staticmethod
            def get(include: list[str]) -> dict[str, object]:
                return {
                    "documents": [
                        "9. Get ramen profitable.",
                        (
                            "Get ramen profitable means earning just enough to cover the founders' "
                            "living expenses so you are not forced to raise money this month. "
                            "That runway freedom is a strategic advantage."
                        ),
                    ],
                    "metadatas": [
                        {"source": "essay.md", "chunk_index": 9},
                        {"source": "essay.md", "chunk_index": 10},
                    ],
                }

        heading = RetrievalResult(
            document=Document(
                page_content="9. Get ramen profitable.",
                metadata={"source": "essay.md", "chunk_index": 9},
            ),
            score=0.5,
            elapsed_ms=1.0,
            rerank_score=0.55,
        )
        with patch.dict("os.environ", {"FRAGMENT_PRIMARY_REPAIR_ENABLED": "1"}, clear=False):
            repaired = _repair_fragment_results(FakeStore(), [heading])
        self.assertEqual(len(repaired), 1)
        self.assertIn("living expenses", repaired[0].document.page_content)
        self.assertTrue(repaired[0].document.metadata.get("repaired_from_fragment"))
        self.assertEqual(repaired[0].rerank_score, 0.55)

    def test_heuristic_fallback_never_claims_full_sufficiency(self) -> None:
        assessment = assess_evidence(
            [_result(0.1, 0.86), _result(0.1, 0.42)],
            candidate_count=8,
            reranker_provider="heuristic_fallback",
            reranker_enabled=False,
            rerank_ms=1.0,
        )
        self.assertEqual(assessment.status, "review")
        self.assertEqual(assessment.reason_code, "heuristic_only")

    def test_rrf_fuses_dense_and_bm25_without_raw_score_comparison(self) -> None:
        dense = _result(0.9, source="dense.md")
        lexical = RetrievalResult(
            document=Document(page_content="lexical", metadata={"source": "lexical.md"}),
            score=0.0,
            elapsed_ms=1.0,
            retrieval_method="bm25",
            bm25_score=8.0,
        )
        fused = _rrf_fuse([dense], [lexical], limit=2)
        self.assertEqual({item.document.metadata["source"] for item in fused}, {"dense.md", "lexical.md"})
        self.assertTrue(all(item.fusion_score is not None for item in fused))

    def test_source_diversity_gives_a_second_source_a_slot(self) -> None:
        results = [
            _result(0.9, source="same.md"),
            RetrievalResult(
                document=Document(page_content="second", metadata={"source": "same.md", "chunk_index": 1}),
                score=0.8,
                elapsed_ms=1.0,
            ),
            RetrievalResult(
                document=Document(page_content="other", metadata={"source": "other.md", "chunk_index": 0}),
                score=0.7,
                elapsed_ms=1.0,
            ),
        ]
        selected = _diversify_results(results, top_k=2)
        self.assertEqual([item.document.metadata["source"] for item in selected], ["same.md", "other.md"])

    def test_source_intro_replaces_heading_only_candidates(self) -> None:
        class FakeCollection:
            name = "test-source-intro"

            @staticmethod
            def count() -> int:
                return 2

        class FakeStore:
            _collection = FakeCollection()

            @staticmethod
            def get(include: list[str]) -> dict[str, object]:
                return {
                    "documents": [
                        "Heading",
                        "A substantive opening paragraph. " * 20,
                    ],
                    "metadatas": [
                        {"source": "essay.md", "chunk_index": 0},
                        {"source": "essay.md", "chunk_index": 1},
                    ],
                }

        heading = RetrievalResult(
            document=Document(page_content="Heading", metadata={"source": "essay.md", "chunk_index": 0}),
            score=0.5,
            elapsed_ms=1.0,
        )
        with patch.dict("os.environ", {"SOURCE_INTRO_ENABLED": "1"}, clear=False):
            expanded = _source_intro_candidates(FakeStore(), [heading])
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[-1].retrieval_method, "source_intro")
        self.assertEqual(expanded[-1].document.metadata["chunk_index"], 1)

    def test_parent_and_neighbor_context_is_separate_from_primary_scores(self) -> None:
        class FakeCollection:
            name = "test-context-expansion"

            @staticmethod
            def count() -> int:
                return 4

        class FakeStore:
            _collection = FakeCollection()

            @staticmethod
            def get(include: list[str]) -> dict[str, object]:
                rows = []
                for index, content in enumerate(("one", "two", "three", "four")):
                    rows.append(
                        (
                            content,
                            {
                                "source": "essay.md",
                                "document_id": "doc-1",
                                "parent_id": "parent-0" if index < 3 else "parent-1",
                                "chunk_index": index,
                                "chunk_count": 4,
                                "chunk_id": f"chunk-{index}",
                            },
                        )
                    )
                return {
                    "documents": [content for content, _ in rows],
                    "metadatas": [metadata for _, metadata in rows],
                }

        anchor = RetrievalResult(
            document=Document(
                page_content="two",
                metadata={
                    "source": "essay.md",
                    "document_id": "doc-1",
                    "parent_id": "parent-0",
                    "chunk_index": 1,
                    "chunk_count": 4,
                    "chunk_id": "chunk-1",
                },
            ),
            score=0.9,
            elapsed_ms=3.0,
            rerank_score=0.8,
        )
        with patch.dict(
            "os.environ",
            {
                "CONTEXT_EXPANSION_ENABLED": "1",
                "CONTEXT_NEIGHBOR_RADIUS": "1",
                "CONTEXT_MAX_CHUNKS": "6",
            },
            clear=False,
        ):
            expanded = _expand_parent_neighbor_context(FakeStore(), [anchor])
        self.assertEqual(expanded[0].score, 0.9)
        roles = [item.document.metadata.get("context_role") for item in expanded[1:]]
        self.assertIn("parent", roles)
        self.assertIn("neighbor", roles)


if __name__ == "__main__":
    unittest.main()
