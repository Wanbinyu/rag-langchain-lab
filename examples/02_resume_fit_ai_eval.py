"""Run an isolated transfer evaluation on the Resume Fit AI knowledge pack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.cli import build_embeddings, print_index_report
from src.config import Settings, project_settings
from src.evaluation import run_evaluation
from src.indexing import sync_index


PROFILE_ROOT = ROOT / "examples" / "resume_fit_ai"
DATASET_PATH = ROOT / "eval" / "questions_resume_fit_ai_transfer_v1.jsonl"
SUMMARY_PATH = ROOT / "eval" / "reports" / "resume_fit_ai_transfer_v1.json"

MODE_ENV: dict[str, dict[str, str]] = {
    "dense": {
        "QUERY_EXPANSION_ENABLED": "0",
        "HYBRID_RETRIEVAL_ENABLED": "0",
        "BM25_CANDIDATE_LIMIT": "0",
        "RERANKER_PROVIDER": "off",
        "RETRIEVAL_RETRY_ENABLED": "0",
    },
    "hybrid": {
        "QUERY_EXPANSION_ENABLED": "0",
        "HYBRID_RETRIEVAL_ENABLED": "1",
        "BM25_CANDIDATE_LIMIT": "6",
        "RERANKER_PROVIDER": "off",
        "RETRIEVAL_RETRY_ENABLED": "0",
    },
    "rerank": {
        "QUERY_EXPANSION_ENABLED": "0",
        "HYBRID_RETRIEVAL_ENABLED": "1",
        "BM25_CANDIDATE_LIMIT": "6",
        "RERANKER_PROVIDER": "local",
        "RETRIEVAL_RETRY_ENABLED": "0",
    },
}


def profile_settings(root: Path = ROOT) -> Settings:
    base = project_settings(root)
    return Settings(
        root=base.root,
        data_dir=root / "examples" / "resume_fit_ai" / "knowledge_base",
        chroma_dir=root / "work" / "resume-fit-ai-transfer" / "chroma",
        collection_name="resume_fit_ai_transfer_v1",
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        top_k=base.top_k,
    )


@contextmanager
def retrieval_mode(mode: str) -> Iterator[None]:
    values = MODE_ENV[mode]
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def publish_summary(mode_reports: dict[str, dict[str, object]], settings: Settings) -> Path:
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": "resume-fit-ai-transfer-v1",
        "claim_boundary": (
            "Public transfer/acceptance dataset; not a blind test and not evidence of unseen generalization."
        ),
        "corpus": {
            "path": str(settings.data_dir.relative_to(settings.root)).replace("\\", "/"),
            "documents": len(list(settings.data_dir.glob("*.md"))),
            "contains_private_source": False,
            "contains_user_data": False,
        },
        "dataset": str(DATASET_PATH.relative_to(settings.root)).replace("\\", "/"),
        "modes": mode_reports,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return SUMMARY_PATH


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Resume Fit AI 第二领域迁移评测")
    parser.add_argument("--mode", choices=["dense", "hybrid", "rerank", "all"], default="rerank")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument("--force-index", action="store_true", help="强制重建隔离索引")
    parser.add_argument("--with-answers", action="store_true", help="调用 Chat API 评测回答")
    parser.add_argument(
        "--publish-summary",
        action="store_true",
        help="写入可提交的紧凑实验摘要（建议与 --mode all 一起使用）",
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = create_parser().parse_args()
    settings = profile_settings()
    load_dotenv(settings.root / ".env")
    if args.publish_summary and args.mode != "all":
        raise SystemExit("--publish-summary 必须与 --mode all 一起使用，避免发布不完整消融结果。")

    print("同步隔离的 Resume Fit AI 示例索引...")
    index_report = sync_index(settings, build_embeddings, force=args.force_index)
    print_index_report(index_report)

    modes = list(MODE_ENV) if args.mode == "all" else [args.mode]
    summaries: dict[str, dict[str, object]] = {}
    top_k = min(max(args.top_k, 1), 10)
    for mode in modes:
        print(f"\n运行模式: {mode}")
        with retrieval_mode(mode):
            report, detailed_path = run_evaluation(
                settings,
                top_k=top_k,
                include_answers=args.with_answers,
                dataset_path=DATASET_PATH,
            )
        summary = dict(report["summary"])
        summaries[mode] = {
            "configuration": MODE_ENV[mode],
            "top_k": top_k,
            "answers_generated": args.with_answers,
            "summary": summary,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"详细报告: {detailed_path}")

    if args.publish_summary:
        path = publish_summary(summaries, settings)
        print(f"可提交摘要: {path}")


if __name__ == "__main__":
    main()
