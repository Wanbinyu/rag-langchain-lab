from __future__ import annotations

import os
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .retrieval import RetrievalResult, format_context


def build_chat_model() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("未找到 OPENAI_API_KEY。请编辑项目根目录的 .env 文件。")
    kwargs: dict[str, object] = {
        "model": os.getenv("OPENAI_CHAT_MODEL", "deepseek-v4-flash"),
        "temperature": 0,
        "timeout": max(float(os.getenv("CHAT_TIMEOUT_SECONDS", "60")), 1.0),
        "max_retries": max(int(os.getenv("CHAT_MAX_RETRIES", "1")), 0),
    }
    if base_url := os.getenv("OPENAI_API_BASE"):
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_answer_prompt() -> ChatPromptTemplate:
    """Create the shared answer contract used by dashboard and evaluation."""
    return ChatPromptTemplate.from_template(
        """你是一个严谨的知识库问答助手。只根据下面的「参考资料」回答问题。

回答规则：
1. 先综合所有直接相关片段，再给出简洁、明确的结论和关键要点。
2. 将答案拆成 2-4 个互不重复的核心要点；先回答定义或结论，再回答原因、条件、影响或例子（仅在资料支持时回答）。
3. 资料明确支持一部分但不完整时，回答已被支持的部分，并明确说明资料未覆盖的部分；不要因为缺少背景就把整题拒答。
4. 只有在所有片段都没有直接支持问题时，才说「根据现有资料无法回答」。不要用常识、猜测或资料之外的知识补全。
5. 使用下面的固定结构，便于后续审阅：
   结论：一句话回答问题。
   核心要点：用项目符号列出关键概念。
   证据边界：说明资料明确支持和未覆盖的部分；没有缺口时写“资料已覆盖上述结论”。
   来源：列出实际使用的来源文件名。
6. 结尾列出实际使用的来源文件名；不同来源冲突时，指出冲突并建议复核原文。

参考资料：
{context}

问题：{question}
"""
    )


def answer_question(llm: ChatOpenAI, question: str, results: list[RetrievalResult]) -> tuple[str, float]:
    prompt = build_answer_prompt()
    chain = prompt | llm | StrOutputParser()
    started = time.perf_counter()
    answer = chain.invoke({"context": format_context(results), "question": question})
    return answer, (time.perf_counter() - started) * 1000
