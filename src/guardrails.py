from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal


BoundaryStatus = Literal["allowed", "blocked"]


@dataclass(frozen=True)
class QueryBoundaryAssessment:
    status: BoundaryStatus
    reason_code: str

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"


_TEMPLATE_RE = re.compile(r"(?:模板|模版|template)", re.IGNORECASE)
_GENERATION_RE = re.compile(
    r"(?:帮我|替我|写|生成|创建|制作|draft|write|generate|create)",
    re.IGNORECASE,
)

# The user is asking this assistant to perform an action, not to explain a
# documented capability. Documentation about rewrite/deploy/restore is not
# authorization to execute those actions here.
_EXECUTION_RE = re.compile(
    r"(?:"
    r"请直接.{0,24}改写|"
    r"帮我写一份|"
    r"请找回|"
    r"请恢复|"
    r"给我查看|"
    r"rewrite my|"
    r"write me a|"
    r"generate (?:a |an |the )?(?:config|resume|nginx)"
    r")",
    re.IGNORECASE,
)

# Static corpora cannot answer live counts, current role status, or "this
# year/today" publication questions. Keep this narrower than any use of
# 当前/现在 so documented defaults remain answerable.
_LIVE_STATUS_RE = re.compile(
    r"(?:"
    r"此刻|"
    r"目前有多少|"
    r"现在有多少|"
    r"现在还在|"
    r"今年发表|"
    r"今年.{0,12}新文章|"
    r"今天.{0,20}(?:收盘|天气|单价|价格)|"
    r"现在的价格|"
    r"付费用户|"
    r"实时|"
    r"当前实际运行|"
    r"线上实例当前|"
    r"最新的(?:token|财报|单价)|"
    r"published this year|"
    r"how many (?:paying )?users|"
    r"currently running|"
    r"still (?:managing|running|in charge)"
    r")",
    re.IGNORECASE,
)


def _enabled() -> bool:
    value = (os.getenv("QUERY_BOUNDARY_GUARD_ENABLED") or "1").strip().lower()
    return value not in {"0", "false", "off", "no", "disabled"}


def assess_query_boundary(question: str) -> QueryBoundaryAssessment:
    """Reject requests the knowledge base cannot faithfully fulfill.

    Template generation, capability-execution, and live-status checks stay
    narrow so ordinary conceptual questions still reach retrieval.
    """
    if not _enabled():
        return QueryBoundaryAssessment("allowed", "within_configured_scope")
    if _TEMPLATE_RE.search(question) and _GENERATION_RE.search(question):
        return QueryBoundaryAssessment("blocked", "template_generation")
    if _EXECUTION_RE.search(question):
        return QueryBoundaryAssessment("blocked", "capability_execution")
    if _LIVE_STATUS_RE.search(question):
        return QueryBoundaryAssessment("blocked", "live_status")
    return QueryBoundaryAssessment("allowed", "within_configured_scope")
