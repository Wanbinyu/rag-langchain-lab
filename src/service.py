from __future__ import annotations

from .chains import answer_question, build_chat_model
from .guardrails import QueryBoundaryAssessment, assess_query_boundary
from .retrieval import RetrievalReport


def _insufficient_answer() -> str:
    return (
        "根据当前召回到的资料，证据不足，暂不生成可靠答案。"
        "请换一种问法、补充关键词，或同步更多知识库文档。"
    )


def _boundary_answer(reason_code: str) -> str:
    if reason_code == "template_generation":
        return "当前知识库不生成模板类内容；可以回答资料中明确讨论的概念和方法。"
    if reason_code == "capability_execution":
        return (
            "资料可以说明产品能力或操作边界，但不能代替系统执行改写、部署、"
            "恢复任务或查看用户数据。"
        )
    if reason_code == "live_status":
        return "静态知识库不能回答当前运行状态、实时数量或“今年/今天”这类时效性问题。"
    return "当前问题超出知识库的回答范围。"


def guarded_answer(
    question: str, retrieval: RetrievalReport,
) -> tuple[str, float, QueryBoundaryAssessment]:
    """Shared CLI/Web generation gate; rejected questions never construct an API client."""
    boundary = assess_query_boundary(question)
    if boundary.blocked:
        return _boundary_answer(boundary.reason_code), 0.0, boundary
    if retrieval.evidence.status == "insufficient":
        return _insufficient_answer(), 0.0, boundary
    answer, elapsed = answer_question(
        build_chat_model(), question, retrieval.context_results or retrieval.results,
    )
    return answer, elapsed, boundary
