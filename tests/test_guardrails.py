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

    def test_documented_capability_questions_are_allowed(self) -> None:
        allowed = (
            "完整改写为什么不共用同一个输出 token 上限？",
            "系统建议一个新项目时，怎样避免把建议包装成候选人的既有经历？",
            "默认队列满载时，最多有多少请求执行、多少请求等待？",
            "控制台怎样同步索引？",
        )
        for question in allowed:
            with self.subTest(question=question):
                result = assess_query_boundary(question)
                self.assertFalse(result.blocked, result.reason_code)

    def test_capability_execution_is_blocked(self) -> None:
        cases = (
            "请直接把我的简历改写成数据工程师版本。",
            "帮我写一份可以直接部署的 Nginx 配置文件。",
            "请找回我昨天已经过期删除的分析结果。",
            "给我查看某一位用户保存的完整简历内容。",
        )
        for question in cases:
            with self.subTest(question=question):
                result = assess_query_boundary(question)
                self.assertTrue(result.blocked)
                self.assertEqual(result.reason_code, "capability_execution")

    def test_live_status_requests_are_blocked(self) -> None:
        cases = (
            "Paul Graham 现在还在管理 Y Combinator 吗？",
            "Paul Graham 今年发表了哪些新文章？",
            "系统此刻在线上有多少任务正在排队？",
            "Resume Fit AI 目前有多少名付费用户？",
            "DeepSeek 今天最新的 token 单价是多少？",
            "线上实例当前实际运行的模型名称和版本是什么？",
            "比特币现在的价格是多少？",
        )
        for question in cases:
            with self.subTest(question=question):
                result = assess_query_boundary(question)
                self.assertTrue(result.blocked, question)
                self.assertEqual(result.reason_code, "live_status")

    def test_guard_can_be_disabled_for_ablation(self) -> None:
        with patch.dict(os.environ, {"QUERY_BOUNDARY_GUARD_ENABLED": "0"}):
            result = assess_query_boundary("write a business plan template")
        self.assertFalse(result.blocked)


if __name__ == "__main__":
    unittest.main()
