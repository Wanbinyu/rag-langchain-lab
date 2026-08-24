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


def _enabled() -> bool:
    value = (os.getenv("QUERY_BOUNDARY_GUARD_ENABLED") or "1").strip().lower()
    return value not in {"0", "false", "off", "no", "disabled"}


def assess_query_boundary(question: str) -> QueryBoundaryAssessment:
    """Reject explicit template-generation requests at the intent layer.

    A relevant source can still be insufficient for a requested deliverable. Keeping
    this rule narrow avoids rejecting ordinary conceptual questions about writing.
    """
    if _enabled() and _TEMPLATE_RE.search(question) and _GENERATION_RE.search(question):
        return QueryBoundaryAssessment("blocked", "template_generation")
    return QueryBoundaryAssessment("allowed", "within_configured_scope")
