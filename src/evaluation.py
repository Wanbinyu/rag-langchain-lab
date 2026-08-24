from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from .chains import answer_question, build_chat_model
from .cli import build_embeddings
from .config import Settings, project_settings
from .guardrails import assess_query_boundary
from .indexing import open_store
from .retrieval import retrieve_report


ABSTENTION_TEXT = "根据现有资料无法回答"
ABSTENTION_MARKERS = (ABSTENTION_TEXT, "资料中没有答案", "无法回答")


_COVERAGE_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)

# These are deliberately small, auditable aliases for the Graham rubric. The
# raw lexical score remains unchanged; normalized scoring is an additional
# signal used to reduce false negatives from common Chinese/English wording.
_COVERAGE_SYNONYM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("发现真实问题", ("发现真实存在的问题", "发现身边的问题", "发现真实存在的痛点", "发现问题")),
    ("自己想要", ("创始人自己想要", "自己真正想要", "自己希望使用")),
    ("自己能做出来", ("自己能够做出来", "自己可以做出来", "能做出来")),
    ("别人未意识到价值", ("其他人还没意识到价值", "少数人意识到有价值", "尚未意识到价值", "没有充分意识到它的价值")),
    ("无法规模化", ("不能规模化", "不可规模化")),
    ("大块连续时间", ("大块时间", "较长的连续时间", "至少半天", "深度工作时间")),
    ("切成小时间块", ("以小时为单位的间隔", "传统的预约簿", "被会议切割", "会议切割", "许多小时间块")),
    ("真实用户反馈", ("真实反馈", "用户反馈", "告诉你真正应该做的是什么", "教会你该构建什么", "上线能教会你该构建什么")),
    ("资金耗尽前盈亏平衡", ("依靠现有资金实现盈利", "达到盈亏平衡", "实现盈利")),
    ("工作生活不严格分开", ("工作和生活彼此分离", "工作和生活是彼此分离的", "工作/生活是彼此分离", "工作/生活平衡", "工作与生活平衡", "严格分开的工作与生活平衡", "工作生活融合")),
    ("覆盖创始人基本生活开销", ("够支付创始人生活费用", "刚好够支付创始人生活费用", "收入刚好够支付创始人生活费用", "覆盖创始人的基本生活费用")),
    ("像说话一样写", ("用口语化的语言写作", "口语化的语言写作", "像说话一样自然表达", "口语化表达")),
    ("少量核心用户", ("一小批客户", "少量用户", "一批核心热爱者")),
    ("强烈反馈", ("正向反馈", "告诉你该聚焦什么", "真实需求反馈")),
    ("不需要立即融资", ("不再需要这个月就拿到投资", "不急于立刻融资")),
    ("害怕", ("感到害怕", "可怕", "可怕的", "有些害怕")),
    ("身份认知改变", ("威胁到你的身份认同", "改变对自身身份的认识")),
    ("看起来非常困难", ("工作量大", "过于困难", "看起来很难", "困难")),
)

_COVERAGE_ALIAS_MAP = {
    alias.casefold(): canonical
    for canonical, aliases in _COVERAGE_SYNONYM_GROUPS
    for alias in (canonical, *aliases)
}
_COVERAGE_ALIAS_PATTERN = re.compile(
    "|".join(re.escape(alias) for alias in sorted(_COVERAGE_ALIAS_MAP, key=len, reverse=True)),
    re.IGNORECASE,
)


def _normalize_coverage_text(text: str) -> str:
    """Normalize only audited rubric aliases before the optional score."""
    normalized = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return _COVERAGE_ALIAS_PATTERN.sub(
        lambda match: _COVERAGE_ALIAS_MAP[match.group(0).casefold()],
        normalized,
    )


def _is_full_abstention(answer: str) -> bool:
    """Return true only for a standalone refusal, not a partial answer with a caveat."""
    compact = re.sub(r"[\s`*_#：:。.!！?？]+", "", answer)
    return compact in {
        re.sub(r"\s+", "", marker)
        for marker in ABSTENTION_MARKERS
        if marker
    }


def _coverage_features(text: str) -> tuple[set[str], set[str]]:
    units: set[str] = set()
    phrases: set[str] = set()
    for token in _COVERAGE_TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            chars = list(token)
            units.update(chars)
            for size in range(2, min(4, len(chars)) + 1):
                phrases.update(
                    "".join(chars[index:index + size])
                    for index in range(len(chars) - size + 1)
                )
        else:
            units.add(token)
            phrases.add(token)
    return units, phrases


def _answer_point_scores(
    answer_points: tuple[str, ...],
    answer: str | None,
    *,
    normalized: bool = False,
) -> tuple[tuple[float, ...], tuple[bool, ...], float | None]:
    if not answer_points or answer is None:
        return (), (), None
    if _is_full_abstention(answer):
        scores = tuple(0.0 for _ in answer_points)
        return scores, tuple(False for _ in answer_points), 0.0
    if normalized:
        answer = _normalize_coverage_text(answer)
    answer_units, answer_phrases = _coverage_features(answer)
    try:
        threshold = float(os.getenv("ANSWER_POINT_COVERAGE_THRESHOLD", "0.35"))
    except (TypeError, ValueError):
        threshold = 0.35
    scores: list[float] = []
    covered: list[bool] = []
    for point in answer_points:
        if normalized:
            point = _normalize_coverage_text(point)
        point_units, point_phrases = _coverage_features(point)
        unit_coverage = len(point_units & answer_units) / max(len(point_units), 1)
        phrase_coverage = (
            len(point_phrases & answer_phrases) / max(len(point_phrases), 1)
            if point_phrases
            else unit_coverage
        )
        score = min(1.0, 0.25 * unit_coverage + 0.75 * phrase_coverage)
        scores.append(round(score, 4))
        covered.append(score >= threshold)
    return tuple(scores), tuple(covered), round(statistics.mean(scores), 4) if scores else 0.0


def _answer_keyword_group_scores(
    keyword_groups: tuple[tuple[str, ...], ...],
    answer: str | None,
) -> tuple[tuple[float, ...], tuple[bool, ...], float | None]:
    """Score auditable keyword groups without requiring an exact answer string."""
    if not keyword_groups or answer is None:
        return (), (), None
    if _is_full_abstention(answer):
        scores = tuple(0.0 for _ in keyword_groups)
        return scores, tuple(False for _ in keyword_groups), 0.0
    normalized_answer = _normalize_coverage_text(answer).casefold()
    covered = tuple(
        any(term.strip().casefold() in normalized_answer for term in group if term.strip())
        for group in keyword_groups
    )
    scores = tuple(1.0 if value else 0.0 for value in covered)
    return scores, covered, round(statistics.mean(scores), 4) if scores else 0.0


def _runtime_retrieval_parameters() -> dict[str, str]:
    names = (
        "QUERY_EXPANSION_ENABLED",
        "RERANKER_USE_EXPANDED_QUERY",
        "QUERY_BOUNDARY_GUARD_ENABLED",
        "RETRIEVAL_CANDIDATE_MULTIPLIER",
        "RETRIEVAL_CANDIDATE_MIN",
        "RETRIEVAL_CANDIDATE_MAX",
        "LEXICAL_CANDIDATE_LIMIT",
        "LEXICAL_PROVIDER",
        "HYBRID_RETRIEVAL_ENABLED",
        "BM25_CANDIDATE_LIMIT",
        "BM25_TITLE_BOOST",
        "HYBRID_CANDIDATE_POOL",
        "RRF_K",
        "SOURCE_DIVERSITY_ENABLED",
        "MAX_CHUNKS_PER_SOURCE",
        "PRIMARY_SOURCE_SCORE_MARGIN",
        "PRIMARY_SOURCE_ADVANTAGE_MARGIN",
        "SOURCE_INTRO_ENABLED",
        "SOURCE_INTRO_SOURCE_LIMIT",
        "SOURCE_INTRO_MIN_CHARS",
        "CONTEXT_EXPANSION_ENABLED",
        "CONTEXT_NEIGHBOR_RADIUS",
        "CONTEXT_MAX_SOURCES",
        "CONTEXT_MAX_CHUNKS",
        "PARENT_CHUNK_SIZE",
        "RETRIEVAL_RETRY_ENABLED",
        "RETRIEVAL_RETRY_SCORE",
        "RETRIEVAL_RETRY_TOP_K",
        "RETRIEVAL_RETRY_KEEP_ORIGINAL",
        "ANSWER_POINT_COVERAGE_THRESHOLD",
        "RERANKER_SUFFICIENT_THRESHOLD",
        "RERANKER_REVIEW_THRESHOLD",
        "RERANKER_MIN_MARGIN",
    )
    defaults = {
        "QUERY_EXPANSION_ENABLED": "1",
        "RERANKER_USE_EXPANDED_QUERY": "1",
        "QUERY_BOUNDARY_GUARD_ENABLED": "1",
        "RETRIEVAL_CANDIDATE_MULTIPLIER": "3",
        "RETRIEVAL_CANDIDATE_MIN": "9",
        "RETRIEVAL_CANDIDATE_MAX": "18",
        "LEXICAL_CANDIDATE_LIMIT": "3",
        "LEXICAL_PROVIDER": "bm25",
        "HYBRID_RETRIEVAL_ENABLED": "1",
        "BM25_CANDIDATE_LIMIT": "6",
        "BM25_TITLE_BOOST": "0.2",
        "HYBRID_CANDIDATE_POOL": "18",
        "RRF_K": "60",
        "SOURCE_DIVERSITY_ENABLED": "1",
        "MAX_CHUNKS_PER_SOURCE": "2",
        "PRIMARY_SOURCE_SCORE_MARGIN": "0.08",
        "PRIMARY_SOURCE_ADVANTAGE_MARGIN": "0.15",
        "SOURCE_INTRO_ENABLED": "1",
        "SOURCE_INTRO_SOURCE_LIMIT": "3",
        "SOURCE_INTRO_MIN_CHARS": "180",
        "CONTEXT_EXPANSION_ENABLED": "1",
        "CONTEXT_NEIGHBOR_RADIUS": "1",
        "CONTEXT_MAX_SOURCES": "3",
        "CONTEXT_MAX_CHUNKS": "10",
        "PARENT_CHUNK_SIZE": "3",
        "RETRIEVAL_RETRY_ENABLED": "1",
        "RETRIEVAL_RETRY_SCORE": "0.45",
        "RETRIEVAL_RETRY_TOP_K": "6",
        "RETRIEVAL_RETRY_KEEP_ORIGINAL": "1",
        "ANSWER_POINT_COVERAGE_THRESHOLD": "0.35",
        "RERANKER_SUFFICIENT_THRESHOLD": "0.58",
        "RERANKER_REVIEW_THRESHOLD": "0.38",
        "RERANKER_MIN_MARGIN": "0.04",
    }
    return {name: os.getenv(name, defaults[name]) for name in names}


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    category: str
    question: str
    expected_sources: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    expect_abstention: bool = False
    reference_answer: str | None = None
    answer_points: tuple[str, ...] = ()
    answer_keyword_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class AnswerReview:
    """Post-generation structure review; rubric terms never enter the prompt."""

    status: str
    reason_code: str
    reason: str
    coverage: float | None
    covered_count: int
    group_count: int
    covered: tuple[bool, ...] = ()
    missing_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class CaseResult:
    id: str
    category: str
    question: str
    expected_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    first_expected_rank: int | None
    source_hit: bool | None
    reciprocal_rank: float | None
    top_score: float | None
    retrieval_ms: float
    answer: str | None
    answer_ms: float | None
    answer_keyword_hit: bool | None
    abstention_hit: bool | None
    answer_error: str | None
    expect_abstention: bool = False
    reference_answer: str | None = None
    rerank_score: float | None = None
    evidence_status: str | None = None
    evidence_reason: str | None = None
    reranker_provider: str | None = None
    answer_points: tuple[str, ...] = ()
    retry_count: int = 0
    retry_query: str | None = None
    answer_point_scores: tuple[float, ...] = ()
    answer_point_covered: tuple[bool, ...] = ()
    answer_point_coverage: float | None = None
    answer_point_normalized_scores: tuple[float, ...] = ()
    answer_point_normalized_covered: tuple[bool, ...] = ()
    answer_point_normalized_coverage: float | None = None
    context_result_count: int = 0
    parent_context_count: int = 0
    neighbor_context_count: int = 0
    answer_keyword_groups: tuple[tuple[str, ...], ...] = ()
    answer_keyword_group_scores: tuple[float, ...] = ()
    answer_keyword_group_covered: tuple[bool, ...] = ()
    answer_keyword_group_coverage: float | None = None
    answer_review_status: str | None = None
    answer_review_reason_code: str | None = None
    answer_review_reason: str | None = None
    answer_review_missing_groups: tuple[tuple[str, ...], ...] = ()


def _load_answer_points(path: Path) -> dict[str, tuple[str, ...]]:
    """Load optional human rubric points kept beside the Graham test set."""
    # Keep rubrics isolated per dataset so adding a larger or revised test set
    # cannot silently reuse the 65-question annotations.  The legacy filename
    # remains a fallback for the original Graham set.
    dataset_stem = path.stem
    rubric_candidates = [
        path.with_name(f"answer_points_{dataset_stem.removeprefix('questions_')}.json"),
        path.with_name("answer_points_graham_65.json"),
    ]
    rubric_path = next((candidate for candidate in rubric_candidates if candidate.is_file()), None)
    if rubric_path is None:
        return {}
    try:
        payload = json.loads(rubric_path.read_text(encoding="utf-8"))
        cases = payload.get("cases", {})
        if not isinstance(cases, dict):
            raise ValueError("cases 必须是对象")
        return {
            str(case_id): tuple(str(point) for point in points)
            for case_id, points in cases.items()
            if isinstance(points, list) and points
        }
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"答案要点文件格式无效: {rubric_path}") from error


def _load_answer_keyword_groups(path: Path) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Load optional synonym-aware keyword groups kept beside a test set."""
    dataset_stem = path.stem
    rubric_path = path.with_name(
        f"answer_keywords_{dataset_stem.removeprefix('questions_')}.json"
    )
    if not rubric_path.is_file():
        return {}
    try:
        payload = json.loads(rubric_path.read_text(encoding="utf-8"))
        cases = payload.get("cases", {})
        if not isinstance(cases, dict):
            raise ValueError("cases 必须是对象")
        groups_by_case: dict[str, tuple[tuple[str, ...], ...]] = {}
        for case_id, entry in cases.items():
            if not isinstance(entry, dict):
                continue
            raw_groups = entry.get("required_groups", [])
            if not isinstance(raw_groups, list):
                raise ValueError(f"{case_id}.required_groups 必须是数组")
            groups: list[tuple[str, ...]] = []
            for raw_group in raw_groups:
                if not isinstance(raw_group, list):
                    raise ValueError(f"{case_id}.required_groups 中的组必须是数组")
                terms = tuple(str(term).strip() for term in raw_group if str(term).strip())
                if terms:
                    groups.append(terms)
            if groups:
                groups_by_case[str(case_id)] = tuple(groups)
        return groups_by_case
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"答案关键字文件格式无效: {rubric_path}") from error


def assess_answer_structure(
    keyword_groups: tuple[tuple[str, ...], ...],
    answer: str | None,
    evidence_status: str | None,
) -> AnswerReview:
    """Turn post-generation keyword coverage into a conservative review state.

    The required groups are read only after generation.  This prevents a gold
    answer rubric from steering the model while still making omissions visible
    to the caller and evaluation report.
    """
    if not keyword_groups:
        return AnswerReview(
            status="unscored",
            reason_code="no_keyword_rubric",
            reason="当前问题没有配置可审阅的答案要点。",
            coverage=None,
            covered_count=0,
            group_count=0,
        )

    _scores, covered, coverage = _answer_keyword_group_scores(keyword_groups, answer)
    covered_count = sum(covered)
    missing_groups = tuple(
        group for group, is_covered in zip(keyword_groups, covered) if not is_covered
    )
    if evidence_status == "insufficient":
        return AnswerReview(
            status="insufficient",
            reason_code="evidence_insufficient",
            reason="检索证据不足，不能仅凭答案关键词确认结论。",
            coverage=coverage,
            covered_count=covered_count,
            group_count=len(keyword_groups),
            covered=covered,
            missing_groups=missing_groups,
        )
    if answer is None or _is_full_abstention(answer):
        return AnswerReview(
            status="review",
            reason_code="answer_refusal_with_evidence",
            reason="回答未覆盖可审阅要点，但检索层没有判定为证据不足。",
            coverage=coverage,
            covered_count=covered_count,
            group_count=len(keyword_groups),
            covered=covered,
            missing_groups=missing_groups,
        )
    if not missing_groups and evidence_status == "sufficient":
        status = "sufficient"
        reason_code = "all_required_groups_covered"
        reason = "答案覆盖全部必需概念组，且检索证据充分。"
    elif not missing_groups:
        status = "review"
        reason_code = "keyword_complete_evidence_review"
        reason = "答案覆盖全部必需概念组，但检索证据仍需复核。"
    else:
        status = "review"
        reason_code = "missing_required_groups"
        reason = f"答案覆盖 {covered_count}/{len(keyword_groups)} 个必需概念组，需检查遗漏或同义表达。"
    return AnswerReview(
        status=status,
        reason_code=reason_code,
        reason=reason,
        coverage=coverage,
        covered_count=covered_count,
        group_count=len(keyword_groups),
        covered=covered,
        missing_groups=missing_groups,
    )


@lru_cache(maxsize=8)
def _question_keyword_index(root_string: str) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Index exact evaluation questions for optional post-answer dashboard review."""
    root = Path(root_string)
    index: dict[str, tuple[tuple[str, ...], ...]] = {}
    for dataset_path in sorted((root / "eval").glob("questions_*.jsonl")):
        groups_by_case = _load_answer_keyword_groups(dataset_path)
        if not groups_by_case:
            continue
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            case_id = str(data.get("id", ""))
            question = str(data.get("question", "")).strip()
            if question and case_id in groups_by_case:
                index.setdefault(question, groups_by_case[case_id])
    return index


def keyword_groups_for_question(root: Path, question: str) -> tuple[tuple[str, ...], ...]:
    """Return a rubric for an exact known evaluation question, if one exists."""
    return _question_keyword_index(str(root.resolve())).get(question.strip(), ())


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    answer_points = _load_answer_points(path)
    answer_keyword_groups = _load_answer_keyword_groups(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        try:
            cases.append(
                EvaluationCase(
                    id=str(data["id"]),
                    category=str(data["category"]),
                    question=str(data["question"]),
                    expected_sources=tuple(str(value) for value in data.get("expected_sources", [])),
                    expected_keywords=tuple(str(value) for value in data.get("expected_keywords", [])),
                    expect_abstention=bool(data.get("expect_abstention", False)),
                    reference_answer=(str(data["reference_answer"]) if data.get("reference_answer") else None),
                    answer_points=tuple(
                        str(value)
                        for value in data.get("answer_points", answer_points.get(str(data["id"]), ()))
                    ),
                    answer_keyword_groups=tuple(
                        tuple(str(term) for term in group)
                        for group in answer_keyword_groups.get(str(data["id"]), ())
                    ),
                )
            )
        except KeyError as error:
            raise ValueError(f"评测集第 {line_number} 行缺少字段: {error}") from error
    if not cases:
        raise ValueError(f"评测集为空: {path}")
    return cases


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, math.ceil(len(sorted_values) * percentile / 100) - 1)
    return sorted_values[index]


def summarize(results: list[CaseResult]) -> dict[str, object]:
    source_cases = [result for result in results if result.source_hit is not None]
    retrieval_times = [result.retrieval_ms for result in results]
    top_scores = [result.top_score for result in results if result.top_score is not None]
    answer_times = [result.answer_ms for result in results if result.answer_ms is not None]
    keyword_cases = [result for result in results if result.answer_keyword_hit is not None]
    abstention_cases = [result for result in results if result.abstention_hit is not None]
    evidence_cases = [result for result in results if result.evidence_status is not None]
    retry_cases = [result for result in results if result.retry_count]
    point_cases = [result for result in results if result.answer_point_coverage is not None]
    point_scores = [
        score
        for result in point_cases
        for score in result.answer_point_scores
    ]
    point_flags = [
        covered
        for result in point_cases
        for covered in result.answer_point_covered
    ]
    normalized_point_cases = [
        result for result in results if result.answer_point_normalized_coverage is not None
    ]
    normalized_point_scores = [
        score
        for result in normalized_point_cases
        for score in result.answer_point_normalized_scores
    ]
    normalized_point_flags = [
        covered
        for result in normalized_point_cases
        for covered in result.answer_point_normalized_covered
    ]
    keyword_group_cases = [
        result for result in results if result.answer_keyword_group_coverage is not None
    ]
    keyword_group_scores = [
        score
        for result in keyword_group_cases
        for score in result.answer_keyword_group_scores
    ]
    keyword_group_flags = [
        covered
        for result in keyword_group_cases
        for covered in result.answer_keyword_group_covered
    ]
    answer_review_cases = [
        result
        for result in results
        if result.answer_review_status in {"sufficient", "review", "insufficient"}
    ]
    answer_review_counts = Counter(result.answer_review_status for result in answer_review_cases)
    context_counts = [result.context_result_count for result in results]
    parent_context_counts = [result.parent_context_count for result in results]
    neighbor_context_counts = [result.neighbor_context_count for result in results]

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    positive_cases = [result for result in results if result.source_hit is not None]
    abstention_expected_cases = [
        result for result in results if result.expect_abstention or result.source_hit is None
    ]
    evidence_counts = Counter(result.evidence_status for result in evidence_cases)
    reranker_counts = Counter(result.reranker_provider for result in results if result.reranker_provider)
    return {
        "total_cases": len(results),
        "positive_cases": len(positive_cases),
        "abstention_expected_cases": len(abstention_expected_cases),
        "abstention_scored_cases": len(abstention_cases),
        "source_scored_cases": len(source_cases),
        "source_hit_count": sum(bool(result.source_hit) for result in source_cases),
        "source_miss_count": sum(not bool(result.source_hit) for result in source_cases),
        "hit_rate_at_k": rate([bool(result.source_hit) for result in source_cases]),
        "mrr": round(
            sum(result.reciprocal_rank or 0 for result in source_cases) / len(source_cases), 4
        ) if source_cases else None,
        "avg_top_score": round(statistics.mean(top_scores), 4) if top_scores else None,
        "retrieval_avg_ms": round(statistics.mean(retrieval_times), 1) if retrieval_times else 0.0,
        "retrieval_p50_ms": round(statistics.median(retrieval_times), 1) if retrieval_times else 0.0,
        "retrieval_p95_ms": round(_percentile(retrieval_times, 95), 1),
        "answers_generated": len(answer_times),
        "model_answer_calls": sum(time_ms > 0 for time_ms in answer_times),
        "answer_refusal_count": sum(
            bool(result.answer) and _is_full_abstention(result.answer)
            for result in results
        ),
        "answer_avg_ms": round(statistics.mean([time_ms for time_ms in answer_times if time_ms > 0]), 1)
        if any(time_ms > 0 for time_ms in answer_times)
        else None,
        "keyword_hit_rate": rate([bool(result.answer_keyword_hit) for result in keyword_cases]),
        "abstention_hit_rate": rate([bool(result.abstention_hit) for result in abstention_cases]),
        "abstention_correct_count": sum(bool(result.abstention_hit) for result in abstention_cases),
        "evidence_sufficient_rate": rate(
            [result.evidence_status == "sufficient" for result in evidence_cases]
        ),
        "evidence_review_rate": rate(
            [result.evidence_status == "review" for result in evidence_cases]
        ),
        "evidence_insufficient_rate": rate(
            [result.evidence_status == "insufficient" for result in evidence_cases]
        ),
        "evidence_counts": dict(evidence_counts),
        "reranker_provider_counts": dict(reranker_counts),
        "answer_error_count": sum(bool(result.answer_error) for result in results),
        "answer_points_annotated_cases": sum(bool(result.answer_points) for result in results),
        "answer_point_scored_cases": len(point_cases),
        "answer_point_coverage": (
            round(statistics.mean(point_scores), 4) if point_scores else None
        ),
        "answer_point_hit_rate": rate(point_flags),
        "answer_points_covered": sum(point_flags),
        "answer_points_scored": len(point_flags),
        "answer_point_normalized_coverage": (
            round(statistics.mean(normalized_point_scores), 4)
            if normalized_point_scores
            else None
        ),
        "answer_point_normalized_hit_rate": rate(normalized_point_flags),
        "answer_points_normalized_covered": sum(normalized_point_flags),
        "answer_points_normalized_scored": len(normalized_point_flags),
        "answer_keyword_groups_annotated_cases": sum(
            bool(result.answer_keyword_groups) for result in results
        ),
        "answer_keyword_group_scored_cases": len(keyword_group_cases),
        "answer_keyword_group_coverage": (
            round(statistics.mean(keyword_group_scores), 4)
            if keyword_group_scores
            else None
        ),
        "answer_keyword_group_hit_rate": rate(keyword_group_flags),
        "answer_keyword_groups_covered": sum(keyword_group_flags),
        "answer_keyword_groups_scored": len(keyword_group_flags),
        "answer_review_scored_cases": len(answer_review_cases),
        "answer_review_status_counts": dict(answer_review_counts),
        "answer_review_sufficient_rate": rate(
            [result.answer_review_status == "sufficient" for result in answer_review_cases]
        ),
        "answer_review_review_rate": rate(
            [result.answer_review_status == "review" for result in answer_review_cases]
        ),
        "answer_review_insufficient_rate": rate(
            [result.answer_review_status == "insufficient" for result in answer_review_cases]
        ),
        "answer_review_missing_group_cases": sum(
            bool(result.answer_review_missing_groups) for result in answer_review_cases
        ),
        "retrieval_retry_count": len(retry_cases),
        "retrieval_retry_rate": round(len(retry_cases) / len(results), 4) if results else 0.0,
        "avg_context_result_count": round(statistics.mean(context_counts), 2) if context_counts else 0.0,
        "avg_parent_context_count": round(statistics.mean(parent_context_counts), 2) if parent_context_counts else 0.0,
        "avg_neighbor_context_count": round(statistics.mean(neighbor_context_counts), 2) if neighbor_context_counts else 0.0,
    }


def evaluate(
    settings: Settings,
    cases: list[EvaluationCase],
    *,
    top_k: int = 3,
    include_answers: bool = False,
    embeddings: Embeddings | None = None,
) -> list[CaseResult]:
    vector_store = open_store(settings, embeddings or build_embeddings())
    llm = build_chat_model() if include_answers else None
    results: list[CaseResult] = []

    for case in cases:
        retrieval = retrieve_report(vector_store, case.question, top_k)
        retrieved = retrieval.results
        context_results = retrieval.context_results or retrieved
        parent_context_count = sum(
            result.document.metadata.get("context_role") == "parent"
            for result in context_results
        )
        neighbor_context_count = sum(
            result.document.metadata.get("context_role") == "neighbor"
            for result in context_results
        )
        sources = tuple(str(item.document.metadata.get("source", "unknown")) for item in retrieved)
        first_expected_rank = next(
            (rank for rank, source in enumerate(sources, start=1) if source in case.expected_sources),
            None,
        )
        source_hit = None if case.expect_abstention else first_expected_rank is not None
        reciprocal_rank = None if case.expect_abstention else (1 / first_expected_rank if first_expected_rank else 0.0)
        answer = None
        answer_ms = None
        answer_error = None
        keyword_hit = None
        abstention_hit = None
        answer_point_scores: tuple[float, ...] = ()
        answer_point_covered: tuple[bool, ...] = ()
        answer_point_coverage: float | None = None
        answer_point_normalized_scores: tuple[float, ...] = ()
        answer_point_normalized_covered: tuple[bool, ...] = ()
        answer_point_normalized_coverage: float | None = None
        answer_keyword_group_scores: tuple[float, ...] = ()
        answer_keyword_group_covered: tuple[bool, ...] = ()
        answer_keyword_group_coverage: float | None = None
        answer_review: AnswerReview | None = None
        boundary = assess_query_boundary(case.question)

        if llm:
            try:
                if retrieval.evidence.status == "insufficient" or boundary.blocked:
                    # Keep evaluation behavior identical to /api/ask: do not
                    # make the chat model answer without sufficient evidence or
                    # for an explicit template-generation request.
                    answer, answer_ms = ABSTENTION_TEXT, 0.0
                else:
                    answer, answer_ms = answer_question(
                        llm,
                        case.question,
                        context_results,
                    )
                if case.expect_abstention:
                    abstention_hit = any(marker in answer for marker in ABSTENTION_MARKERS)
                elif case.expected_keywords:
                    normalized_answer = answer.lower()
                    keyword_hit = any(keyword.lower() in normalized_answer for keyword in case.expected_keywords)
                (
                    answer_point_scores,
                    answer_point_covered,
                    answer_point_coverage,
                ) = _answer_point_scores(case.answer_points, answer)
                (
                    answer_point_normalized_scores,
                    answer_point_normalized_covered,
                    answer_point_normalized_coverage,
                ) = _answer_point_scores(case.answer_points, answer, normalized=True)
                (
                    answer_keyword_group_scores,
                    answer_keyword_group_covered,
                    answer_keyword_group_coverage,
                ) = _answer_keyword_group_scores(case.answer_keyword_groups, answer)
                if case.answer_keyword_groups:
                    answer_review = assess_answer_structure(
                        case.answer_keyword_groups,
                        answer,
                        retrieval.evidence.status,
                    )
            except Exception as error:
                answer_error = f"{type(error).__name__}: {error}"

        results.append(
            CaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                expected_sources=case.expected_sources,
                expect_abstention=case.expect_abstention,
                retrieved_sources=sources,
                first_expected_rank=first_expected_rank,
                source_hit=source_hit,
                reciprocal_rank=reciprocal_rank,
                top_score=round(retrieved[0].score, 4) if retrieved else None,
                retrieval_ms=round(retrieved[0].elapsed_ms, 1) if retrieved else 0.0,
                answer=answer,
                answer_ms=round(answer_ms, 1) if answer_ms is not None else None,
                answer_keyword_hit=keyword_hit,
                abstention_hit=abstention_hit,
                answer_error=answer_error,
                reference_answer=case.reference_answer,
                rerank_score=(
                    round(retrieved[0].rerank_score, 4)
                    if retrieved and retrieved[0].rerank_score is not None
                    else None
                ),
                evidence_status=retrieval.evidence.status,
                evidence_reason=retrieval.evidence.reason_code,
                reranker_provider=retrieval.evidence.reranker_provider,
                answer_points=case.answer_points,
                retry_count=retrieval.retry_count,
                retry_query=retrieval.retry_query,
                answer_point_scores=answer_point_scores,
                answer_point_covered=answer_point_covered,
                answer_point_coverage=answer_point_coverage,
                answer_point_normalized_scores=answer_point_normalized_scores,
                answer_point_normalized_covered=answer_point_normalized_covered,
                answer_point_normalized_coverage=answer_point_normalized_coverage,
                context_result_count=len(context_results),
                parent_context_count=parent_context_count,
                neighbor_context_count=neighbor_context_count,
                answer_keyword_groups=case.answer_keyword_groups,
                answer_keyword_group_scores=answer_keyword_group_scores,
                answer_keyword_group_covered=answer_keyword_group_covered,
                answer_keyword_group_coverage=answer_keyword_group_coverage,
                answer_review_status=answer_review.status if answer_review else None,
                answer_review_reason_code=answer_review.reason_code if answer_review else None,
                answer_review_reason=answer_review.reason if answer_review else None,
                answer_review_missing_groups=answer_review.missing_groups if answer_review else (),
            )
        )
    return results


def write_report(
    settings: Settings,
    results: list[CaseResult],
    top_k: int,
    include_answers: bool,
    dataset_path: Path,
) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = settings.root / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"baseline-{timestamp}.json"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parameters": {
            "collection": settings.collection_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": top_k,
            "answers_generated": include_answers,
            "answer_structure_contract": "v1" if include_answers else None,
            "answer_review_mode": "post_generation_keyword_groups" if include_answers else None,
            "dataset": str(dataset_path),
            "retrieval": _runtime_retrieval_parameters(),
        },
        "summary": summarize(results),
        "cases": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def run_evaluation(
    settings: Settings,
    *,
    top_k: int = 3,
    include_answers: bool = False,
    dataset_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    dataset_path = dataset_path or (settings.root / "eval" / "questions.jsonl")
    if not dataset_path.is_absolute():
        dataset_path = settings.root / dataset_path
    cases = load_cases(dataset_path)
    results = evaluate(settings, cases, top_k=top_k, include_answers=include_answers)
    report_path = write_report(settings, results, top_k, include_answers, dataset_path)
    return {
        "parameters": {
            "collection": settings.collection_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": top_k,
            "answers_generated": include_answers,
            "answer_structure_contract": "v1" if include_answers else None,
            "answer_review_mode": "post_generation_keyword_groups" if include_answers else None,
            "dataset": str(dataset_path),
            "retrieval": _runtime_retrieval_parameters(),
        },
        "summary": summarize(results),
        "cases": [asdict(result) for result in results],
        "report_path": str(report_path),
    }, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 检索与回答评测")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument("--with-answers", action="store_true", help="同时调用 Chat 模型生成答案")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/questions.jsonl"),
        help="评测集 JSONL 路径，默认 eval/questions.jsonl",
    )
    args = parser.parse_args()
    settings = project_settings()
    load_dotenv(settings.root / ".env")
    report, report_path = run_evaluation(
        settings,
        top_k=min(max(args.top_k, 1), 10),
        include_answers=args.with_answers,
        dataset_path=args.dataset,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
