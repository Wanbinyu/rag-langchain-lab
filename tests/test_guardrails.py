from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.guardrails import assess_query_boundary


class QueryBoundaryTests(unittest.TestCase):
    def test_template_generation_is_blocked(self) -> None:
        result = assess_query_boundary("请帮我写一份商业计划书模板")
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason_code, "template_generation")

    def test_conceptual_writing_question_is_allowed(self) -> None:
        result = assess_query_boundary("Paul Graham 对写作有什么建议？")
        self.assertFalse(result.blocked)
        self.assertEqual(result.reason_code, "within_configured_scope")

    def test_explaining_a_template_is_not_generation(self) -> None:
        result = assess_query_boundary("请解释商业计划书模板的作用")
        self.assertFalse(result.blocked)

    def test_guard_can_be_disabled_for_ablation(self) -> None:
        with patch.dict(os.environ, {"QUERY_BOUNDARY_GUARD_ENABLED": "0"}):
            result = assess_query_boundary("write a business plan template")
        self.assertFalse(result.blocked)


if __name__ == "__main__":
    unittest.main()
