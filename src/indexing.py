from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .loaders import load_documents_report

INDEX_VERSION = "v4-context-integrity"
EMBEDDING_BATCH_SIZE = 2048


@dataclass(frozen=True)
class IndexReport:
    scanned_files: int
    total_chunks: int
    added_files: int
    updated_files: int
    removed_files: int
    skipped_files: int
    data_integrity_status: str = "ok"
    data_issue_count: int = 0
    empty_files: tuple[str, ...] = ()
    unreadable_files: tuple[str, ...] = ()
    decode_warning_files: tuple[str, ...] = ()
    index_integrity_status: str = "ok"
    index_integrity_issue_count: int = 0

    @property
    def changed(self) -> bool:
        return any((self.added_files, self.updated_files, self.removed_files))


@dataclass(frozen=True)
class IndexIntegrityReport:
    status: str
    indexed_chunks: int
    indexed_sources: int
    expected_sources: int
    missing_sources: tuple[str, ...] = ()
    stale_sources: tuple[str, ...] = ()
    invalid_sources: tuple[str, ...] = ()
    invalid_chunks: int = 0
    duplicate_chunk_ids: int = 0
    issues: tuple[str, ...] = ()

    @property
    def issue_count(self) -> int:
        return (
            len(self.missing_sources)
            + len(self.stale_sources)
            + len(self.invalid_sources)
            + self.invalid_chunks
            + self.duplicate_chunk_ids
        )


def open_store(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
    )


def _fingerprint(document: Document, settings: Settings) -> str:
    return ":".join(
        (
            INDEX_VERSION,
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            str(document.metadata["content_hash"]),
        )
    )


def _indexed_fingerprints(store: Chroma) -> dict[str, set[str]]:
    result = store.get(include=["metadatas"])
    by_source: dict[str, set[str]] = defaultdict(set)
    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue
        source = metadata.get("source")
        fingerprint = metadata.get("index_fingerprint")
        if source and fingerprint:
            by_source[str(source)].add(str(fingerprint))
    return dict(by_source)


_REQUIRED_CHUNK_METADATA = (
    "source",
    "content_hash",
    "index_fingerprint",
    "chunk_index",
    "chunk_count",
    "chunk_id",
    "document_id",
    "parent_id",
    "prev_chunk_id",
    "next_chunk_id",
    "char_start",
    "char_end",
)


def _metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    try:
        return int(value) if value is not None and str(value) != "" else None
    except (TypeError, ValueError):
        return None


def _inspect_index_payload(
    payload: dict[str, object],
    expected_sources: set[str] | None = None,
) -> IndexIntegrityReport:
    raw_metadatas = payload.get("metadatas") or []
    raw_ids = payload.get("ids") or []
    metadatas = [dict(metadata or {}) for metadata in raw_metadatas if metadata]
    ids = [str(value) for value in raw_ids]
    rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    invalid_sources: set[str] = set()
    invalid_chunks = 0
    seen_ids: set[str] = set()
    duplicate_chunk_ids = 0

    for position, metadata in enumerate(metadatas):
        source = str(metadata.get("source") or "")
        if source:
            rows_by_source[source].append(metadata)
        missing = [key for key in _REQUIRED_CHUNK_METADATA if key not in metadata]
        chunk_index = _metadata_int(metadata, "chunk_index")
        chunk_count = _metadata_int(metadata, "chunk_count")
        char_start = _metadata_int(metadata, "char_start")
        char_end = _metadata_int(metadata, "char_end")
        if not source or missing or chunk_index is None or chunk_count is None or chunk_count <= 0:
            invalid_chunks += 1
            if source:
                invalid_sources.add(source)
        if char_start is None or char_end is None or char_start < 0 or char_end < char_start:
            invalid_chunks += 1
            if source:
                invalid_sources.add(source)
        chunk_id = str(metadata.get("chunk_id") or (ids[position] if position < len(ids) else ""))
        if chunk_id:
            if chunk_id in seen_ids:
                duplicate_chunk_ids += 1
                if source:
                    invalid_sources.add(source)
            seen_ids.add(chunk_id)

    for source, rows in rows_by_source.items():
        indexes = sorted(
            index for row in rows if (index := _metadata_int(row, "chunk_index")) is not None
        )
        counts = {
            count for row in rows if (count := _metadata_int(row, "chunk_count")) is not None
        }
        expected_count = max(counts) if counts else None
        if expected_count is None or len(rows) != expected_count or indexes != list(range(expected_count)):
            invalid_sources.add(source)
        by_index = {
            index: row
            for row in rows
            if (index := _metadata_int(row, "chunk_index")) is not None
        }
        for index, row in by_index.items():
            previous = by_index.get(index - 1)
            following = by_index.get(index + 1)
            expected_prev = str(previous.get("chunk_id", "")) if previous else ""
            expected_next = str(following.get("chunk_id", "")) if following else ""
            if str(row.get("prev_chunk_id", "")) != expected_prev:
                invalid_sources.add(source)
            if str(row.get("next_chunk_id", "")) != expected_next:
                invalid_sources.add(source)

    actual_sources = set(rows_by_source)
    expected = expected_sources or set()
    missing_sources = tuple(sorted(expected - actual_sources))
    stale_sources = tuple(sorted(actual_sources - expected)) if expected_sources is not None else ()
    invalid_sources.update(missing_sources)
    issues: list[str] = []
    if missing_sources:
        issues.append(f"missing_sources:{len(missing_sources)}")
    if stale_sources:
        issues.append(f"stale_sources:{len(stale_sources)}")
    if invalid_sources:
        issues.append(f"invalid_sources:{len(invalid_sources)}")
    if invalid_chunks:
        issues.append(f"invalid_chunks:{invalid_chunks}")
    if duplicate_chunk_ids:
        issues.append(f"duplicate_chunk_ids:{duplicate_chunk_ids}")
    return IndexIntegrityReport(
        status="error" if issues else "ok",
        indexed_chunks=len(raw_ids) if raw_ids else len(metadatas),
        indexed_sources=len(actual_sources),
        expected_sources=len(expected_sources) if expected_sources is not None else len(actual_sources),
        missing_sources=missing_sources,
        stale_sources=stale_sources,
        invalid_sources=tuple(sorted(invalid_sources)),
        invalid_chunks=invalid_chunks,
        duplicate_chunk_ids=duplicate_chunk_ids,
        issues=tuple(issues),
    )


def _invalid_index_sources(store: Chroma) -> set[str]:
    try:
        payload = store.get(include=["metadatas"])
    except Exception:  # pragma: no cover - backend-dependent
        return set()
    return set(_inspect_index_payload(payload).invalid_sources)


def check_index_integrity(
    settings: Settings,
    store: Chroma | None = None,
    documents: list[Document] | None = None,
) -> IndexIntegrityReport:
    """Validate chunk metadata, chains, and source coverage in Chroma."""
    active_store = store or open_store(settings)
    expected_documents = (
        documents
        if documents is not None
        else load_documents_report(settings.data_dir)[0]
    )
    expected_sources = {str(document.metadata.get("source", "")) for document in expected_documents}
    try:
        payload = active_store.get(include=["metadatas"])
    except Exception as error:  # pragma: no cover - backend-dependent
        return IndexIntegrityReport(
            status="error",
            indexed_chunks=0,
            indexed_sources=0,
            expected_sources=len(expected_sources),
            issues=(f"read_error:{type(error).__name__}",),
        )
    report = _inspect_index_payload(payload, expected_sources)
    expected_by_source = {
        str(document.metadata.get("source", "")): document
        for document in expected_documents
    }
    payload_rows = [dict(metadata or {}) for metadata in payload.get("metadatas") or [] if metadata]
    rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for metadata in payload_rows:
        source = str(metadata.get("source") or "")
        if source:
            rows_by_source[source].append(metadata)
    stale_content_sources = set(report.invalid_sources)
    for source, document in expected_by_source.items():
        rows = rows_by_source.get(source, [])
        if not rows:
            continue
        actual_hashes = {str(row.get("content_hash", "")) for row in rows}
        actual_fingerprints = {str(row.get("index_fingerprint", "")) for row in rows}
        expected_hash = str(document.metadata.get("content_hash", ""))
        expected_fingerprint = _fingerprint(document, settings)
        if actual_hashes != {expected_hash} or actual_fingerprints != {expected_fingerprint}:
            stale_content_sources.add(source)
    if not stale_content_sources.difference(report.invalid_sources):
        return report
    issues = list(report.issues)
    if "stale_content" not in ":".join(issues):
        issues.append(f"stale_content:{len(stale_content_sources.difference(report.invalid_sources))}")
    return IndexIntegrityReport(
        status="error",
        indexed_chunks=report.indexed_chunks,
        indexed_sources=report.indexed_sources,
        expected_sources=report.expected_sources,
        missing_sources=report.missing_sources,
        stale_sources=report.stale_sources,
        invalid_sources=tuple(sorted(stale_content_sources)),
        invalid_chunks=report.invalid_chunks,
        duplicate_chunk_ids=report.duplicate_chunk_ids,
        issues=tuple(issues),
    )


def _chunk_offsets(text: str, chunks: list[Document], overlap: int) -> list[tuple[int, int]]:
    """Find best-effort character ranges in the original document.

    RecursiveCharacterTextSplitter keeps chunk text intact in normal cases.
    The fallback range is still monotonic when a custom splitter normalizes a
    separator, so metadata remains useful for diagnostics rather than making
    indexing fail on one unusual document.
    """
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        content = chunk.page_content
        search_from = max(0, cursor - max(overlap, 0))
        start = text.find(content, search_from)
        if start < 0:
            start = min(cursor, len(text))
        end = min(len(text), start + len(content))
        offsets.append((start, end))
        cursor = max(cursor, end)
    return offsets


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _chunk_document(document: Document, settings: Settings) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    source = str(document.metadata["source"])
    document_id = str(
        document.metadata.get("document_id")
        or hashlib.sha256(f"source:{source}".encode("utf-8")).hexdigest()[:24]
    )
    fingerprint = _fingerprint(document, settings)
    chunks = splitter.split_documents([document])
    offsets = _chunk_offsets(document.page_content, chunks, settings.chunk_overlap)
    include_metadata = (os.getenv("INDEX_INCLUDE_METADATA") or "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    title = str(document.metadata.get("title") or "").strip()
    directory = str(document.metadata.get("directory") or "").strip()
    metadata_prefix = ""
    if include_metadata:
        metadata_lines = [f"Title: {title or source}", f"Source: {source}"]
        if directory:
            metadata_lines.append(f"Directory: {directory}")
        metadata_prefix = "\n".join(metadata_lines) + "\n\n"
    ids: list[str] = []
    for chunk_index, chunk in enumerate(chunks):
        stable_id = f"{source}:{document.metadata['content_hash']}:{chunk_index}:{INDEX_VERSION}"
        ids.append(hashlib.sha256(stable_id.encode("utf-8")).hexdigest())

    parent_size = max(_safe_int_env("PARENT_CHUNK_SIZE", 3), 1)
    parent_count = (len(chunks) + parent_size - 1) // parent_size
    chunk_count = len(chunks)
    for chunk_index, chunk in enumerate(chunks):
        if metadata_prefix:
            chunk.page_content = f"{metadata_prefix}{chunk.page_content}"
        parent_index = chunk_index // parent_size
        parent_id = hashlib.sha256(
            f"{document_id}:parent:{parent_index}:{INDEX_VERSION}".encode("utf-8")
        ).hexdigest()[:24]
        char_start, char_end = offsets[chunk_index] if chunk_index < len(offsets) else (0, 0)
        chunk.metadata = {
            **chunk.metadata,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "chunk_id": ids[chunk_index],
            "document_id": document_id,
            "parent_id": parent_id,
            "parent_chunk_index": parent_index,
            "parent_chunk_count": parent_count,
            "prev_chunk_id": ids[chunk_index - 1] if chunk_index else "",
            "next_chunk_id": ids[chunk_index + 1] if chunk_index + 1 < chunk_count else "",
            "char_start": char_start,
            "char_end": char_end,
            "index_fingerprint": fingerprint,
        }
    return chunks, ids


def sync_index(
    settings: Settings,
    embeddings_factory: Callable[[], Embeddings],
    *,
    force: bool = False,
    progress_callback: Callable[[str, int], None] | None = None,
) -> IndexReport:
    """仅写入新增或变化的文件，并从索引删除已经移除的文件。"""
    def report_progress(message: str, percent: int) -> None:
        if progress_callback:
            progress_callback(message, percent)

    report_progress("扫描 data 目录中的文档", 10)
    documents, data_integrity = load_documents_report(settings.data_dir)
    documents_by_source = {str(doc.metadata["source"]): doc for doc in documents}
    report_progress("读取当前 Chroma 索引元数据", 25)
    store = open_store(settings)
    indexed = _indexed_fingerprints(store)
    invalid_sources = _invalid_index_sources(store)

    changed: list[Document] = []
    added_files = 0
    updated_files = 0
    for source, document in documents_by_source.items():
        expected = _fingerprint(document, settings)
        if not force and source not in invalid_sources and indexed.get(source) == {expected}:
            continue
        changed.append(document)
        if source in indexed:
            updated_files += 1
        else:
            added_files += 1

    removed_sources = set(indexed) - set(documents_by_source)
    delete_sources = removed_sources | {str(doc.metadata["source"]) for doc in changed}
    report_progress("比较文件哈希，确定增量变更", 40)
    for source in delete_sources:
        store.delete(where={"source": source})

    total_new_chunks = 0
    if changed:
        report_progress("加载 Embedding 模型", 55)
        store = open_store(settings, embeddings_factory())
        chunks: list[Document] = []
        ids: list[str] = []
        for document in changed:
            document_chunks, document_ids = _chunk_document(document, settings)
            chunks.extend(document_chunks)
            ids.extend(document_ids)
        if chunks:
            for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                end = min(start + EMBEDDING_BATCH_SIZE, len(chunks))
                percent = 80 + int((end / len(chunks)) * 12)
                report_progress(f"写入向量片段 {end}/{len(chunks)}", percent)
                store.add_documents(chunks[start:end], ids=ids[start:end])
            total_new_chunks = len(chunks)

    report_progress("读取同步后的索引状态", 95)
    final_store = open_store(settings)
    total_chunks = len(final_store.get(include=[]).get("ids") or [])
    index_integrity = check_index_integrity(settings, final_store, documents=documents)
    report_progress("索引同步完成", 100)
    from .retrieval import clear_lexical_cache

    clear_lexical_cache()
    return IndexReport(
        scanned_files=len(documents),
        total_chunks=total_chunks,
        added_files=added_files,
        updated_files=updated_files,
        removed_files=len(removed_sources),
        skipped_files=len(documents) - len(changed),
        data_integrity_status=data_integrity.status,
        data_issue_count=data_integrity.issue_count,
        empty_files=data_integrity.empty_files,
        unreadable_files=data_integrity.unreadable_files,
        decode_warning_files=data_integrity.decode_warning_files,
        index_integrity_status=index_integrity.status,
        index_integrity_issue_count=index_integrity.issue_count,
    )


def index_status(settings: Settings) -> tuple[int, int]:
    store = open_store(settings)
    result = store.get(include=["metadatas"])
    sources = {
        metadata.get("source")
        for metadata in result.get("metadatas") or []
        if metadata and metadata.get("source")
    }
    return len(result.get("ids") or []), len(sources)
