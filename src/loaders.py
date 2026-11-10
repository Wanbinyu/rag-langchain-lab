from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}
SKIP_FILENAMES = {"retrieval_glossary.json"}


@dataclass(frozen=True)
class DataIntegrityReport:
    """Issues found while scanning source files."""

    scanned_files: int
    loaded_files: int
    empty_files: tuple[str, ...] = ()
    unreadable_files: tuple[str, ...] = ()
    decode_warning_files: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.unreadable_files:
            return "error"
        if self.empty_files or self.decode_warning_files:
            return "warning"
        return "ok"

    @property
    def issue_count(self) -> int:
        return len(self.empty_files) + len(self.unreadable_files) + len(self.decode_warning_files)


def _document_title(path: Path, text: str) -> str:
    """Use the first Markdown heading, falling back to the file stem."""
    match = re.search(r"(?m)^\s*#\s+(.+?)\s*$", text)
    if match:
        return match.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").strip()


def read_text_file(path: Path) -> str:
    """读取常见中英文文本编码，无法识别时保留可读内容。"""
    return _decode_text(path.read_bytes())


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def list_directories(data_dir: Path) -> list[str]:
    """返回 data/ 下的所有目录，包含没有可索引文件的空目录。"""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    return [
        path.relative_to(data_dir).as_posix()
        for path in sorted(data_dir.rglob("*"))
        if path.is_dir() and not any(part.startswith(".") for part in path.relative_to(data_dir).parts)
    ]


def load_documents_report(data_dir: Path) -> tuple[list[Document], DataIntegrityReport]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.name not in SKIP_FILENAMES
        and not any(part.startswith(".") for part in path.relative_to(data_dir).parts)
    )
    documents: list[Document] = []
    empty_files: list[str] = []
    unreadable_files: list[str] = []
    decode_warning_files: list[str] = []
    for path in files:
        source = path.relative_to(data_dir).as_posix()
        try:
            raw = path.read_bytes()
            text = _decode_text(raw)
        except OSError:
            unreadable_files.append(source)
            continue
        if "\ufffd" in text:
            decode_warning_files.append(source)
        text = text.strip()
        if not text:
            empty_files.append(source)
            continue
        relative_parts = path.relative_to(data_dir).parts
        directory = "/".join(relative_parts[:-1])
        document_id = hashlib.sha256(f"source:{source}".encode("utf-8")).hexdigest()[:24]
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "title": _document_title(path, text),
                    "directory": directory,
                    "filename": path.name,
                    "content_hash": hashlib.sha256(raw).hexdigest(),
                    "document_id": document_id,
                },
            )
        )
    return documents, DataIntegrityReport(
        scanned_files=len(files),
        loaded_files=len(documents),
        empty_files=tuple(empty_files),
        unreadable_files=tuple(unreadable_files),
        decode_warning_files=tuple(decode_warning_files),
    )


def load_documents(data_dir: Path) -> list[Document]:
    """Load supported files while preserving the original list-only API."""
    documents, _ = load_documents_report(data_dir)
    return documents
