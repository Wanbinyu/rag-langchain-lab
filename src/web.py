from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from dotenv import load_dotenv

from .cli import build_embeddings
from .config import DEFAULT_LOCAL_EMBED_MODEL, Settings, project_settings
from .evaluation import AnswerReview, assess_answer_structure, keyword_groups_for_question, run_evaluation
from .glossary import configure_glossary
from .service import guarded_answer
from .indexing import IndexReport, check_index_integrity, index_status, open_store, sync_index
from .loaders import list_directories, load_documents_report
from .retrieval import RetrievalReport, RetrievalResult, retrieve_report

STATIC_DIR = Path(__file__).resolve().parents[1] / "web"
RAG_LOCK = threading.RLock()


def _report_as_dict(report: IndexReport) -> dict[str, object]:
    return {
        "scanned_files": report.scanned_files,
        "total_chunks": report.total_chunks,
        "added_files": report.added_files,
        "updated_files": report.updated_files,
        "removed_files": report.removed_files,
        "skipped_files": report.skipped_files,
        "data_integrity_status": report.data_integrity_status,
        "data_issue_count": report.data_issue_count,
        "empty_files": list(report.empty_files),
        "unreadable_files": list(report.unreadable_files),
        "decode_warning_files": list(report.decode_warning_files),
        "index_integrity_status": report.index_integrity_status,
        "index_integrity_issue_count": report.index_integrity_issue_count,
    }


def _result_as_dict(result: RetrievalResult, rank: int) -> dict[str, object]:
    content = result.document.page_content.strip()
    return {
        "rank": rank,
        # Keep score for clients using the original API; expose the meaning explicitly too.
        "score": round(result.score, 4),
        "vector_score": round(result.score, 4),
        "dense_distance": round(result.dense_distance, 4) if result.dense_distance is not None else None,
        "rerank_score": round(result.rerank_score, 4) if result.rerank_score is not None else None,
        "bm25_score": round(result.bm25_score, 4) if result.bm25_score is not None else None,
        "fusion_score": round(result.fusion_score, 6) if result.fusion_score is not None else None,
        "reranker_provider": result.reranker_provider,
        "retrieval_method": result.retrieval_method,
        "source": result.document.metadata.get("source", "unknown"),
        "title": result.document.metadata.get("title", ""),
        "chunk_index": result.document.metadata.get("chunk_index", "?"),
        "chunk_count": result.document.metadata.get("chunk_count"),
        "document_id": result.document.metadata.get("document_id"),
        "parent_id": result.document.metadata.get("parent_id"),
        "prev_chunk_id": result.document.metadata.get("prev_chunk_id"),
        "next_chunk_id": result.document.metadata.get("next_chunk_id"),
        "char_start": result.document.metadata.get("char_start"),
        "char_end": result.document.metadata.get("char_end"),
        "context_role": result.document.metadata.get("context_role", "primary"),
        "context_distance": result.document.metadata.get("context_distance"),
        "content": content,
    }


def _evidence_as_dict(report: RetrievalReport) -> dict[str, object]:
    evidence = report.evidence
    return {
        "status": evidence.status,
        "top_score": evidence.top_score,
        "score_gap": evidence.score_gap,
        "reason_code": evidence.reason_code,
        "reranker_enabled": evidence.reranker_enabled,
        "reranker_provider": evidence.reranker_provider,
        "candidate_count": evidence.candidate_count,
        "vector_ms": report.vector_ms,
        "rerank_ms": evidence.rerank_ms,
        "total_ms": report.total_ms,
        "reranker_error": evidence.reranker_error,
    }


def _answer_review_as_dict(review: AnswerReview) -> dict[str, object]:
    return {
        "status": review.status,
        "reason_code": review.reason_code,
        "reason": review.reason,
        "coverage": review.coverage,
        "covered_count": review.covered_count,
        "group_count": review.group_count,
        "covered": list(review.covered),
        "missing_groups": [list(group) for group in review.missing_groups],
    }


def _retrieval_as_dict(report: RetrievalReport) -> dict[str, object]:
    return {
        "retrieval_ms": report.total_ms,
        "vector_ms": report.vector_ms,
        "rerank_ms": report.rerank_ms,
        "retry_count": report.retry_count,
        "retry_query": report.retry_query,
        "evidence": _evidence_as_dict(report),
        "results": [
            _result_as_dict(result, rank)
            for rank, result in enumerate(report.results, start=1)
        ],
        "context_results": [
            _result_as_dict(result, rank)
            for rank, result in enumerate(report.context_results, start=1)
        ],
    }


class RAGDashboardHandler(BaseHTTPRequestHandler):
    server_version = "RAGLabDashboard/1.0"

    @property
    def settings(self) -> Settings:
        return self.server.settings  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")

    def _send_json(self, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > 1_000_000:
            raise ValueError("请求正文大小无效")
        payload = json.loads(self.rfile.read(size).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求正文必须是 JSON 对象")
        return payload

    @staticmethod
    def _top_k(value: object) -> int:
        try:
            top_k = int(value)
        except (TypeError, ValueError):
            return 3
        return min(max(top_k, 1), 10)

    def _evaluation_dataset(self, value: object) -> Path:
        requested = str(value or "eval/questions.jsonl").strip()
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = self.settings.root / candidate
        candidate = candidate.resolve()
        eval_root = (self.settings.root / "eval").resolve()
        try:
            candidate.relative_to(eval_root)
        except ValueError as error:
            raise ValueError("评测集必须位于项目 eval/ 目录内") from error
        if candidate.suffix.lower() != ".jsonl" or not candidate.is_file():
            raise ValueError(f"评测集不存在或格式不是 JSONL: {candidate.name}")
        return candidate

    def _status_payload(self) -> dict[str, object]:
        with RAG_LOCK:
            chunks, indexed_sources = index_status(self.settings)
            documents, data_integrity = load_documents_report(self.settings.data_dir)
            directories = list_directories(self.settings.data_dir)
            index_integrity = check_index_integrity(self.settings, documents=documents)
        files = []
        for document in documents:
            source = str(document.metadata["source"])
            file_path = self.settings.data_dir / source
            files.append(
                {
                    "source": source,
                    "characters": len(document.page_content),
                    "hash": str(document.metadata["content_hash"])[:10],
                    "modified_at": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(file_path.stat().st_mtime)
                    ),
                }
            )
        return {
            "chunks": chunks,
            "indexed_sources": indexed_sources,
            "documents": len(documents),
            "collection": self.settings.collection_name,
            "embedding_provider": "local" if not self._is_api_embedding() else "api",
            "embedding_model": (
                os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                if self._is_api_embedding()
                else os.getenv("LOCAL_EMBED_MODEL", DEFAULT_LOCAL_EMBED_MODEL)
            ),
            "embedding_languages": "中文 / English" if not self._is_api_embedding() else "取决于 API 模型",
            "directories": directories,
            "files": files,
            "data_integrity": {
                "status": data_integrity.status,
                "scanned_files": data_integrity.scanned_files,
                "loaded_files": data_integrity.loaded_files,
                "issue_count": data_integrity.issue_count,
                "empty_files": list(data_integrity.empty_files),
                "unreadable_files": list(data_integrity.unreadable_files),
                "decode_warning_files": list(data_integrity.decode_warning_files),
            },
            "index_integrity": {
                "status": index_integrity.status,
                "indexed_chunks": index_integrity.indexed_chunks,
                "indexed_sources": index_integrity.indexed_sources,
                "expected_sources": index_integrity.expected_sources,
                "missing_sources": list(index_integrity.missing_sources),
                "stale_sources": list(index_integrity.stale_sources),
                "invalid_sources": list(index_integrity.invalid_sources),
                "invalid_chunks": index_integrity.invalid_chunks,
                "duplicate_chunk_ids": index_integrity.duplicate_chunk_ids,
                "issue_count": index_integrity.issue_count,
                "issues": list(index_integrity.issues),
            },
        }

    @staticmethod
    def _is_api_embedding() -> bool:
        import os

        return (os.getenv("EMBEDDING_PROVIDER") or "local").lower() in {
            "openai",
            "openai-compatible",
            "api",
        }

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_types = {
            ".css": "text/css; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_index(self, force: bool) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: str, payload: dict[str, object]) -> None:
            message = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            self.wfile.write(message.encode("utf-8"))
            self.wfile.flush()

        try:
            with RAG_LOCK:
                report = sync_index(
                    self.settings,
                    build_embeddings,
                    force=force,
                    progress_callback=lambda message, percent: emit(
                        "progress", {"message": message, "percent": percent}
                    ),
                )
            emit("complete", {"report": _report_as_dict(report)})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            try:
                emit("error", {"message": f"{type(error).__name__}: {error}"})
            except (BrokenPipeError, ConnectionResetError):
                return
        finally:
            # 每次同步只有有限事件；关闭连接能让非浏览器客户端及时结束读取。
            self.close_connection = True

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/status":
            try:
                self._send_json(self._status_payload())
            except Exception as error:
                self._send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/index-stream":
            force = parse_qs(parsed.query).get("force", ["0"])[0] == "1"
            self._stream_index(force)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/evaluate":
                top_k = self._top_k(payload.get("top_k", 3))
                include_answers = bool(payload.get("include_answers"))
                dataset_path = self._evaluation_dataset(payload.get("dataset"))
                with RAG_LOCK:
                    report, _ = run_evaluation(
                        self.settings,
                        top_k=top_k,
                        include_answers=include_answers,
                        dataset_path=dataset_path,
                    )
                self._send_json(report)
                return
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("请输入问题")
            top_k = self._top_k(payload.get("top_k", 3))
            if parsed.path == "/api/retrieve":
                with RAG_LOCK:
                    retrieval = retrieve_report(
                        open_store(self.settings, build_embeddings()), question, top_k
                    )
                response = {"question": question}
                response.update(_retrieval_as_dict(retrieval))
                self._send_json(response)
                return
            if parsed.path == "/api/ask":
                with RAG_LOCK:
                    report = (
                        sync_index(self.settings, build_embeddings)
                        if bool(payload.get("sync"))
                        else None
                    )
                    retrieval = retrieve_report(
                        open_store(self.settings, build_embeddings()), question, top_k
                    )
                # The retrieved documents form a snapshot; network generation does not
                # need the lock that protects indexing and retrieval.
                answer, answer_ms, boundary = guarded_answer(question, retrieval)
                keyword_groups = keyword_groups_for_question(self.settings.root, question)
                answer_review = assess_answer_structure(
                    keyword_groups, answer, retrieval.evidence.status,
                )
                response = {
                    "question": question,
                    "answer": answer,
                    "answer_ms": round(answer_ms, 1),
                    "query_boundary": {
                        "status": boundary.status,
                        "reason_code": boundary.reason_code,
                    },
                    "answer_review": _answer_review_as_dict(answer_review),
                    "sync_report": _report_as_dict(report) if report else None,
                }
                response.update(_retrieval_as_dict(retrieval))
                self._send_json(response)
                return
            self._send_json({"error": "未知 API 路径"}, HTTPStatus.NOT_FOUND)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json(
                {"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )


def run_server(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> None:
    load_dotenv(settings.root / ".env")
    configure_glossary(settings.data_dir)
    server = ThreadingHTTPServer((host, port), RAGDashboardHandler)
    server.settings = settings  # type: ignore[attr-defined]
    print(f"RAG 控制台已启动: http://{host}:{port}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n控制台已停止。")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地 RAG 可视化控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(project_settings(), args.host, args.port)


if __name__ == "__main__":
    main()
