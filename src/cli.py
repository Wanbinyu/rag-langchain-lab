from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .chains import answer_question, build_chat_model
from .config import DEFAULT_LOCAL_EMBED_MODEL, Settings, project_settings
from .indexing import IndexReport, index_status, open_store, sync_index
from .retrieval import print_retrieval_diagnostics, retrieve, retrieve_report

COMMANDS = {"index", "ask", "chat", "status"}


def build_embeddings() -> Embeddings:
    provider = (os.getenv("EMBEDDING_PROVIDER") or "local").strip().lower()
    if provider in {"local", "fastembed"}:
        model = os.getenv(
            "LOCAL_EMBED_MODEL",
            DEFAULT_LOCAL_EMBED_MODEL,
        )
        print(f"Embedding: local FastEmbed ({model})")
        return FastEmbedEmbeddings(model_name=model)
    if provider in {"openai", "openai-compatible", "api"}:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("API Embedding 需要 OPENAI_API_KEY。")
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        kwargs: dict[str, object] = {"model": model}
        if base_url := os.getenv("OPENAI_API_BASE"):
            kwargs["base_url"] = base_url
        print(f"Embedding: API ({model})")
        return OpenAIEmbeddings(**kwargs)
    raise RuntimeError(f"未知 EMBEDDING_PROVIDER={provider!r}，请用 local 或 openai-compatible")


def print_index_report(report: IndexReport) -> None:
    print("\n索引状态:")
    print(f"  扫描文件: {report.scanned_files} | 索引片段: {report.total_chunks}")
    print(
        "  本次变化: "
        f"新增 {report.added_files}，更新 {report.updated_files}，"
        f"删除 {report.removed_files}，跳过 {report.skipped_files}"
    )


    print(
        f"  数据完整性: {report.data_integrity_status} "
        f"({report.data_issue_count} issues) | "
        f"索引完整性: {report.index_integrity_status} "
        f"({report.index_integrity_issue_count} issues)"
    )


def sync_and_report(settings: Settings, force: bool = False) -> None:
    report = sync_index(settings, build_embeddings, force=force)
    print_index_report(report)


def answer_once(settings: Settings, question: str, top_k: int) -> None:
    question = question.strip()
    if not question:
        print("（空问题，已跳过）")
        return
    store = open_store(settings, build_embeddings())
    retrieval = retrieve_report(store, question, top_k)
    results = retrieval.results
    print(f"\n问题: {question}")
    print_retrieval_diagnostics(results)
    print("\n生成回答中...")
    answer, answer_ms = answer_question(
        build_chat_model(), question, retrieval.context_results or results
    )
    print("\n回答:")
    print(answer)
    print(f"\n生成耗时: {answer_ms:.0f} ms")
    print("-" * 40)


def interactive_loop(settings: Settings, top_k: int) -> None:
    print("\n交互问答已启动。输入 exit / quit / q 退出。")
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            return
        try:
            answer_once(settings, question, top_k)
        except Exception as error:
            print(f"回答失败: {type(error).__name__}: {error}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模块化、可观察的本地 RAG 学习工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="按文件哈希增量同步 data/ 到 Chroma")
    index.add_argument("--force", action="store_true", help="忽略哈希状态并重建所有文件")

    ask = subparsers.add_parser("ask", help="提问一次，并展示检索片段和相关度")
    ask.add_argument("question", nargs="+", help="问题文本")
    ask.add_argument("--sync", action="store_true", help="提问前增量同步知识库")
    ask.add_argument("-k", "--top-k", type=int, default=3, help="召回片段数，默认 3")

    chat = subparsers.add_parser("chat", help="进入连续问答模式")
    chat.add_argument("--sync", action="store_true", help="启动前增量同步知识库")
    chat.add_argument("-k", "--top-k", type=int, default=3, help="召回片段数，默认 3")

    subparsers.add_parser("status", help="查看当前 RAG 索引状态")
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    normalized = list(argv)
    if not normalized:
        normalized = ["chat", "--sync"]
    elif normalized[0] not in COMMANDS and not normalized[0].startswith("-"):
        # 保留旧用法：python examples/01_basic_rag.py 什么是 RAG
        normalized = ["ask", "--sync", *normalized]
    return create_parser().parse_args(normalized)


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    settings = settings or project_settings()
    load_dotenv(settings.root / ".env")
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        if args.command == "index":
            sync_and_report(settings, force=args.force)
        elif args.command == "status":
            chunks, sources = index_status(settings)
            print(f"索引 collection: {settings.collection_name}")
            print(f"索引片段: {chunks} | 来源文件: {sources}")
        elif args.command == "ask":
            if args.sync:
                sync_and_report(settings)
            answer_once(settings, " ".join(args.question), args.top_k)
        elif args.command == "chat":
            if args.sync:
                sync_and_report(settings)
            interactive_loop(settings, args.top_k)
    except Exception as error:
        print(f"错误: {type(error).__name__}: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
