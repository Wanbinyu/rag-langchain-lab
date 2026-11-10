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
DATASETS = {
    "transfer": ROOT / "eval" / "questions_resume_fit_ai_transfer_v1.jsonl",
    "frozen": ROOT / "eval" / "questions_resume_fit_ai_frozen_v1.jsonl",
    "answers": ROOT / "eval" / "questions_resume_fit_ai_answers_v1.jsonl",
}

SMALL_CORPUS_BUDGET = {
    # Five short source documents do not need the general-purpose 18-candidate
    # budget. Six candidates preserve the measured Top-3 quality while cutting
    # Cross-Encoder work substantially.
    "RETRIEVAL_CANDIDATE_MIN": "5",
    "RETRIEVAL_CANDIDATE_MAX": "6",
    "HYBRID_CANDIDATE_POOL": "6",
    # Transfer checks the generic path: no Graham (or any) domain glossary.
    "QUERY_GLOSSARY_PATH": "off",
}

MODE_ENV: dict[str, dict[str, str]] = {
    "dense": {
        **SMALL_CORPUS_BUDGET,
        "QUERY_EXPANSION_ENABLED": "0",
        "HYBRID_RETRIEVAL_ENABLED": "0",
        "BM25_CANDIDATE_LIMIT": "0",
        "RERANKER_PROVIDER": "off",
        "RETRIEVAL_RETRY_ENABLED": "0",
    },
    "hybrid": {
        **SMALL_CORPUS_BUDGET,
        "QUERY_EXPANSION_ENABLED": "0",
        "HYBRID_RETRIEVAL_ENABLED": "1",
        "BM25_CANDIDATE_LIMIT": "6",
        "RERANKER_PROVIDER": "off",
        "RETRIEVAL_RETRY_ENABLED": "0",
    },
    "rerank": {
        **SMALL_CORPUS_BUDGET,
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


def publish_summary(
    mode_reports: dict[str, dict[str, object]],
    settings: Settings,
    dataset_name: str,
) -> Path:
    dataset_path = DATASETS[dataset_name]
    summary_path = settings.root / "eval" / "reports" / f"resume_fit_ai_{dataset_name}_v1.json"
    boundaries = {
        "transfer": "Public transfer/acceptance dataset; not a blind test and not evidence of unseen generalization.",
        "frozen": "Prospectively frozen before first execution and excluded from tuning; small sample, not a generalization claim.",
        "answers": "Small deterministic answer/citation/refusal audit; citation validity checks provenance, not semantic faithfulness.",
    }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": f"resume-fit-ai-{dataset_name}-v1",
        "claim_boundary": boundaries[dataset_name],
        "corpus": {
            "path": str(settings.data_dir.relative_to(settings.root)).replace("\\", "/"),
            "documents": len(list(settings.data_dir.glob("*.md"))),
            "contains_private_source": False,
            "contains_user_data": False,
        },
        "dataset": str(dataset_path.relative_to(settings.root)).replace("\\", "/"),
        "modes": mode_reports,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Resume Fit AI 第二领域迁移评测")
    parser.add_argument("--mode", choices=["dense", "hybrid", "rerank", "all"], default="rerank")
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="transfer",
        help="评测集：迁移验收、冻结测试或小型答案审计",
    )
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument("--force-index", action="store_true", help="强制重建隔离索引")
    parser.add_argument("--with-answers", action="store_true", help="调用 Chat API 评测回答")
    parser.add_argument("--publish-summary", action="store_true", help="写入可提交的紧凑实验摘要")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = create_parser().parse_args()
    settings = profile_settings()
    load_dotenv(settings.root / ".env")
    if args.dataset == "frozen" and args.mode != "rerank":
        raise SystemExit("冻结集只允许最终 rerank 配置，不用于消融比较。")
    if args.publish_summary and args.dataset == "transfer" and args.mode != "all":
        raise SystemExit("transfer 的 --publish-summary 必须与 --mode all 一起使用。")
    if args.dataset == "answers" and (not args.with_answers or args.mode != "rerank"):
        raise SystemExit("answers 数据集必须使用 --mode rerank --with-answers。")

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
                dataset_path=DATASETS[args.dataset],
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
        answer_errors = sum(
            int(report["summary"].get("answer_error_count", 0))
            for report in summaries.values()
        )
        if answer_errors:
            raise SystemExit(
                f"检测到 {answer_errors} 个答案调用错误；保留本地详细报告，但不发布不完整摘要。"
            )
        path = publish_summary(summaries, settings, args.dataset)
        print(f"可提交摘要: {path}")


if __name__ == "__main__":
    main()
