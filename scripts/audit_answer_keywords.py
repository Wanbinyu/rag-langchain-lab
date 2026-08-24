"""Audit generated answers against the curated keyword-group rubric.

This reads an existing evaluation report and never calls an embedding or chat
provider, so rubric changes can be compared cheaply and reproducibly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

# Running a script from ``scripts/`` puts that directory first on sys.path;
# add the project root so the existing ``src`` package is importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import _answer_keyword_group_scores, _is_full_abstention


def _coverage_rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def audit(report_path: Path, rubric_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    rubric_cases = rubric.get("cases", {})
    report_cases = report.get("cases", [])
    dataset_path = Path(str(report.get("parameters", {}).get("dataset", "")))
    legacy_keywords: dict[str, tuple[str, ...]] = {}
    if dataset_path.is_file():
        legacy_rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        legacy_keywords = {
            str(row["id"]): tuple(str(term) for term in row.get("expected_keywords", []))
            for row in legacy_rows
        }

    details: list[dict[str, object]] = []
    positive_details: list[dict[str, object]] = []
    answerable_details: list[dict[str, object]] = []
    all_group_flags: list[bool] = []
    case_coverages: list[float] = []
    case_any_flags: list[bool] = []
    case_all_flags: list[bool] = []
    evidence_counts: Counter[str] = Counter()
    legacy_flags: list[bool] = []

    for case in report_cases:
        case_id = str(case["id"])
        entry = rubric_cases.get(case_id, {})
        groups = tuple(
            tuple(str(term) for term in group)
            for group in entry.get("required_groups", [])
        )
        answer = case.get("answer")
        scores, covered, coverage = _answer_keyword_group_scores(groups, answer)
        covered_count = sum(covered)
        group_count = len(groups)
        uncovered = [
            {
                "group_index": index + 1,
                "terms": list(group),
            }
            for index, (group, is_covered) in enumerate(zip(groups, covered))
            if not is_covered
        ]
        detail = {
            "id": case_id,
            "question": case.get("question"),
            "expect_abstention": bool(case.get("expect_abstention", False)),
            "answer": answer,
            "evidence_status": case.get("evidence_status"),
            "source_hit": case.get("source_hit"),
            "answer_review_status": case.get("answer_review_status"),
            "answer_review_reason": case.get("answer_review_reason"),
            "answer_review_missing_groups": case.get("answer_review_missing_groups", []),
            "required_group_count": group_count,
            "covered_group_count": covered_count,
            "group_coverage": coverage,
            "group_scores": list(scores),
            "group_covered": list(covered),
            "uncovered_groups": uncovered,
        }
        if not bool(case.get("expect_abstention", False)) and legacy_keywords.get(case_id):
            normalized_answer = str(answer or "").casefold()
            legacy_flags.append(
                bool(answer)
                and not _is_full_abstention(str(answer))
                and any(term.casefold() in normalized_answer for term in legacy_keywords[case_id])
            )
        details.append(detail)
        if case.get("evidence_status"):
            evidence_counts[str(case["evidence_status"])] += 1
        if not bool(case.get("expect_abstention", False)):
            positive_details.append(detail)
            if group_count:
                all_group_flags.extend(covered)
                case_coverages.append(float(coverage or 0.0))
                case_any_flags.append(any(covered))
                case_all_flags.append(all(covered))
                if answer and not _is_full_abstention(answer):
                    answerable_details.append(detail)

    source_summary = report.get("summary", {})
    positive_count = len(positive_details)
    group_count = len(all_group_flags)
    covered_count = sum(all_group_flags)
    answer_available_count = sum(
        bool(detail["answer"]) and not _is_full_abstention(str(detail["answer"]))
        for detail in positive_details
    )
    answerable_group_flags = [
        covered
        for detail in answerable_details
        for covered in detail["group_covered"]
    ]
    answerable_case_coverages = [
        float(detail["group_coverage"] or 0.0) for detail in answerable_details
    ]
    answerable_case_any_flags = [
        bool(detail["covered_group_count"]) for detail in answerable_details
    ]
    answerable_case_all_flags = [
        detail["covered_group_count"] == detail["required_group_count"]
        for detail in answerable_details
    ]
    answerable_group_count = len(answerable_group_flags)
    answerable_covered_count = sum(answerable_group_flags)
    output = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_report": str(report_path),
        "rubric": {
            "path": str(rubric_path),
            "version": rubric.get("version"),
        },
        "summary": {
            "total_cases": len(report_cases),
            "positive_cases": positive_count,
            "abstention_cases": len(report_cases) - positive_count,
            "abstention_hit_rate": source_summary.get("abstention_hit_rate"),
            "positive_answers_available": answer_available_count,
            "positive_answers_refused_by_evidence": positive_count - answer_available_count,
            "required_groups_scored": group_count,
            "required_groups_covered": covered_count,
            "required_group_coverage": round(covered_count / group_count, 4) if group_count else None,
            "answerable_cases": len(answerable_details),
            "answerable_required_groups_scored": answerable_group_count,
            "answerable_required_groups_covered": answerable_covered_count,
            "answerable_required_group_coverage": (
                round(answerable_covered_count / answerable_group_count, 4)
                if answerable_group_count
                else None
            ),
            "case_mean_group_coverage": round(mean(case_coverages), 4) if case_coverages else None,
            "case_any_group_hit_rate": _coverage_rate(case_any_flags),
            "case_all_required_groups_hit_rate": _coverage_rate(case_all_flags),
            "cases_with_all_required_groups": sum(case_all_flags),
            "answerable_case_mean_group_coverage": (
                round(mean(answerable_case_coverages), 4)
                if answerable_case_coverages
                else None
            ),
            "answerable_case_any_group_hit_rate": _coverage_rate(answerable_case_any_flags),
            "answerable_case_all_required_groups_hit_rate": _coverage_rate(answerable_case_all_flags),
            "answerable_cases_with_all_required_groups": sum(answerable_case_all_flags),
            "evidence_status_counts": dict(evidence_counts),
            "source_recall_at_k": source_summary.get("hit_rate_at_k"),
            "source_mrr": source_summary.get("mrr"),
            "legacy_keyword_hit_rate": source_summary.get("keyword_hit_rate"),
            "legacy_keyword_hit_rate_recomputed": _coverage_rate(legacy_flags),
            "legacy_keyword_scored_cases": len(legacy_flags),
            "normalized_answer_point_coverage": source_summary.get("answer_point_normalized_coverage"),
        },
        "cases": details,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="离线审计答案关键字组覆盖率")
    parser.add_argument("--report", type=Path, required=True, help="已有的带答案评测报告")
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("eval/answer_keywords_graham_150.json"),
        help="关键词组侧车文件",
    )
    parser.add_argument("--output", type=Path, help="输出审计报告路径")
    args = parser.parse_args()
    root = ROOT
    report_path = args.report if args.report.is_absolute() else root / args.report
    rubric_path = args.rubric if args.rubric.is_absolute() else root / args.rubric
    output_path = args.output
    if output_path is None:
        output_path = root / "eval" / "results" / (
            f"keyword-audit-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}.json"
        )
    elif not output_path.is_absolute():
        output_path = root / output_path

    result = audit(report_path, rubric_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"审计报告已写入: {output_path}")


if __name__ == "__main__":
    main()
