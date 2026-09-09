from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .glossary import active_glossary, expansion_enabled


EvidenceStatus = Literal["sufficient", "review", "insufficient"]

_RERANKER_LOCK = threading.Lock()
_RERANKER_CACHE: dict[str, object] = {}
_RERANKER_ERRORS: dict[str, str] = {}
_LEXICAL_CACHE_LOCK = threading.Lock()
_LEXICAL_CACHE: dict[str, tuple[int, list[tuple[str, dict[str, object]]]]] = {}
_BM25_CACHE: dict[str, tuple[int, "_BM25Index"]] = {}
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "be", "by", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "what", "why", "with", "what's",
    "的", "了", "和", "是", "在", "对", "与", "为", "什么", "如何", "为什么", "什", "么",
}


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    elapsed_ms: float
    dense_distance: float | None = None
    rerank_score: float | None = None
    reranker_provider: str | None = None
    retrieval_method: str = "vector"
    bm25_score: float | None = None
    fusion_score: float | None = None


@dataclass(frozen=True)
class EvidenceAssessment:
    status: EvidenceStatus
    top_score: float | None
    score_gap: float | None
    reason_code: str
    reranker_enabled: bool
    reranker_provider: str
    candidate_count: int
    rerank_ms: float
    reranker_error: str | None = None


@dataclass(frozen=True)
class RetrievalReport:
    results: list[RetrievalResult]
    evidence: EvidenceAssessment
    vector_ms: float
    rerank_ms: float
    total_ms: float
    retry_count: int = 0
    retry_query: str | None = None
    # Primary results keep the retrieval metric honest. Context results may
    # include parent/neighbor material and are used only to build the answer.
    context_results: list[RetrievalResult] = field(default_factory=list)


@dataclass(frozen=True)
class _BM25Index:
    documents: list[tuple[str, dict[str, object]]]
    tokenized_documents: list[list[str]]
    document_frequency: dict[str, int]
    average_document_length: float


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    try:
        return int(value) if value is not None and str(value) != "" else None
    except (TypeError, ValueError):
        return None


def _bool_env(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    value = (os.getenv(name, fallback) or fallback).strip().lower()
    return value not in {"0", "false", "off", "no", "disabled"}


def _query_expansion_enabled() -> bool:
    return expansion_enabled()


def _query_aliases(question: str) -> list[str]:
    if not _query_expansion_enabled():
        return []
    lower_question = question.lower()
    aliases: list[str] = []
    for phrase, values in active_glossary().aliases:
        if phrase.lower() in lower_question:
            aliases.extend(values)
    return list(dict.fromkeys(aliases))


def _rule_topic_terms(question: str) -> list[str]:
    """Return English topic terms from corpus rewrite rules, if a glossary is loaded."""
    if not _query_expansion_enabled():
        return []
    lower_question = question.lower()
    terms: list[str] = []
    for signals, min_hits, values in active_glossary().rewrite_rules:
        hits = sum(1 for signal in signals if signal.lower() in lower_question)
        if hits >= max(min_hits, 1):
            terms.extend(values)
    return list(dict.fromkeys(terms))


def _retrieval_terms(question: str) -> list[str]:
    """Union of glossary aliases and rule-based English topics."""
    return list(dict.fromkeys([*_query_aliases(question), *_rule_topic_terms(question)]))


def _expand_query(question: str) -> str:
    """Append audited English concepts for lexical/hybrid candidate generation."""
    terms = _retrieval_terms(question)
    if not terms:
        return question
    return f"{question}\nRelevant concepts: {'; '.join(terms)}"


def _rule_rewrite_query(question: str) -> str | None:
    """Build a controlled second-pass retrieval query for hard cases.

    English topic terms lead so BM25 prefers essay titles/bodies; the original
    question is retained for multilingual dense retrieval. Terms never enter
    the answer prompt.
    """
    terms = _retrieval_terms(question)
    if not terms:
        return None
    english_lead = " ".join(terms)
    return f"{english_lead}\n{question}"


def _rerank_query(question: str) -> str:
    """Optionally expose domain terminology to the cross-encoder.

    Keeping this behind a flag lets evaluation compare original-language
    reranking with cross-language terminology assistance without changing the
    default evidence policy.
    """
    if not _bool_env("RERANKER_USE_EXPANDED_QUERY", True):
        return question
    return _expand_query(question)


def _token_list(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(token)
            tokens.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


def _terms(text: str) -> set[str]:
    return set(_token_list(text))


def _searchable_text(content: str, metadata: dict[str, object] | None = None) -> str:
    metadata = metadata or {}
    metadata_text = " ".join(
        str(metadata.get(key, "")).strip()
        for key in ("title", "source", "directory", "filename")
        if metadata.get(key)
    )
    return f"{metadata_text}\n{content}".strip()


def _lexical_score(question: str, content: str) -> float:
    query_terms = _terms(question)
    document_terms = _terms(content)
    if not query_terms:
        return 0.0
    overlap = len(query_terms & document_terms) / len(query_terms)
    compact_question = re.sub(r"\s+", "", question.lower())
    compact_content = re.sub(r"\s+", "", content.lower())
    phrase_bonus = 1.0 if len(compact_question) >= 4 and compact_question in compact_content else 0.0
    query_latin = re.findall(r"[a-z0-9]+", question.lower())
    content_latin = " ".join(re.findall(r"[a-z0-9]+", content.lower()))
    phrase_coverage = 0.0
    for size in range(min(6, len(query_latin)), 1, -1):
        if any(
            " ".join(query_latin[index:index + size]) in content_latin
            for index in range(len(query_latin) - size + 1)
        ):
            phrase_coverage = size / max(len(query_latin), 1)
            break
    return min(1.0, 0.55 * overlap + 0.15 * phrase_bonus + 0.30 * phrase_coverage)


def _normalise_dense_scores(results: list[RetrievalResult]) -> list[float]:
    scores = [result.score for result in results]
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [0.5 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _heuristic_rerank(question: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
    vector_results = [result for result in results if result.retrieval_method == "vector"]
    vector_dense_scores = _normalise_dense_scores(vector_results)
    vector_score_map = {id(result): score for result, score in zip(vector_results, vector_dense_scores)}
    scored: list[RetrievalResult] = []
    for result in results:
        # Lexical candidates are exact-term evidence, so they enter the second stage
        # with a neutral dense component instead of being penalized for lacking a vector distance.
        dense_score = (
            1.0
            if result.retrieval_method in {"lexical", "bm25"}
            else vector_score_map.get(id(result), 0.0)
        )
        lexical_score = _lexical_score(
            question,
            _searchable_text(result.document.page_content, result.document.metadata),
        )
        combined_score = min(1.0, 0.75 * lexical_score + 0.25 * dense_score)
        scored.append(
            replace(
                result,
                rerank_score=combined_score,
                reranker_provider="heuristic_fallback",
            )
        )
    return sorted(scored, key=lambda item: (item.rerank_score or 0.0, item.score), reverse=True)


def _cross_encoder(model_name: str) -> object | None:
    if model_name in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_name]
    if model_name in _RERANKER_ERRORS:
        return None
    with _RERANKER_LOCK:
        if model_name in _RERANKER_CACHE:
            return _RERANKER_CACHE[model_name]
        if model_name in _RERANKER_ERRORS:
            return None
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cache_dir = os.getenv("RERANKER_CACHE_DIR") or None
            encoder = TextCrossEncoder(model_name, cache_dir=cache_dir, lazy_load=True)
            _RERANKER_CACHE[model_name] = encoder
            return encoder
        except Exception as error:  # pragma: no cover - depends on optional model/runtime
            _RERANKER_ERRORS[model_name] = f"{type(error).__name__}: {error}"
            return None


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-clipped))


_RERANK_PAIR_CACHE: dict[tuple[str, str], float] = {}
_RERANK_PAIR_CACHE_LOCK = threading.Lock()
_RERANK_PAIR_CACHE_MAX = 4096


def _doc_cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _question_cache_key(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()


def clear_rerank_pair_cache() -> None:
    with _RERANK_PAIR_CACHE_LOCK:
        _RERANK_PAIR_CACHE.clear()


def _rerank(
    question: str,
    results: list[RetrievalResult],
) -> tuple[list[RetrievalResult], str, bool, float, str | None]:
    provider = (os.getenv("RERANKER_PROVIDER") or "local").strip().lower()
    if provider in {"off", "none", "disabled"} or not results:
        return results, "disabled", False, 0.0, None
    if provider in {"heuristic", "lexical", "hybrid"}:
        started = time.perf_counter()
        fallback = _heuristic_rerank(question, results)
        return fallback, "heuristic_fallback", False, (time.perf_counter() - started) * 1000, None

    model_name = os.getenv(
        "RERANKER_MODEL",
        "jinaai/jina-reranker-v2-base-multilingual",
    )
    started = time.perf_counter()
    encoder = _cross_encoder(model_name)
    reranker_error: str | None = None
    if encoder is not None:
        try:
            documents = [
                _searchable_text(result.document.page_content, result.document.metadata)
                for result in results
            ]
            q_key = _question_cache_key(question)
            pair_keys = [(q_key, _doc_cache_key(doc)) for doc in documents]
            cached_scores: list[float | None] = [None] * len(documents)
            missing_indices: list[int] = []
            if _bool_env("RERANK_PAIR_CACHE_ENABLED", True):
                with _RERANK_PAIR_CACHE_LOCK:
                    for index, key in enumerate(pair_keys):
                        cached = _RERANK_PAIR_CACHE.get(key)
                        if cached is None:
                            missing_indices.append(index)
                        else:
                            cached_scores[index] = cached
            else:
                missing_indices = list(range(len(documents)))

            if missing_indices:
                missing_docs = [documents[index] for index in missing_indices]
                raw_scores = list(encoder.rerank(question, missing_docs, batch_size=32))  # type: ignore[attr-defined]
                if len(raw_scores) != len(missing_docs):
                    raise RuntimeError("reranker returned unexpected score count")
                with _RERANK_PAIR_CACHE_LOCK:
                    for index, raw_score in zip(missing_indices, raw_scores):
                        score = _sigmoid(float(raw_score))
                        cached_scores[index] = score
                        if _bool_env("RERANK_PAIR_CACHE_ENABLED", True):
                            if len(_RERANK_PAIR_CACHE) >= _RERANK_PAIR_CACHE_MAX:
                                # Drop an arbitrary oldest-ish batch of keys.
                                for drop_key in list(_RERANK_PAIR_CACHE.keys())[:512]:
                                    _RERANK_PAIR_CACHE.pop(drop_key, None)
                            _RERANK_PAIR_CACHE[pair_keys[index]] = score
            elif not missing_indices:
                # Full cache hit — no model call.
                pass

            if any(score is None for score in cached_scores):
                raise RuntimeError("incomplete rerank cache materialization")

            reranked = [
                replace(
                    result,
                    rerank_score=float(score),
                    reranker_provider="cross_encoder",
                )
                for result, score in zip(results, cached_scores)
                if score is not None
            ]
            reranked.sort(
                key=lambda item: (item.rerank_score or 0.0, item.score),
                reverse=True,
            )
            return reranked, "cross_encoder", True, (time.perf_counter() - started) * 1000, None
        except Exception as error:  # pragma: no cover - depends on ONNX model/runtime
            reranker_error = f"{type(error).__name__}: {error}"
    else:
        reranker_error = _RERANKER_ERRORS.get(model_name)

    fallback = _heuristic_rerank(question, results)
    return (
        fallback,
        "heuristic_fallback",
        False,
        (time.perf_counter() - started) * 1000,
        reranker_error,
    )


def _deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[tuple[str, str]] = set()
    unique: list[RetrievalResult] = []
    for result in results:
        metadata = result.document.metadata
        source = str(metadata.get("source", "unknown"))
        chunk = str(metadata.get("chunk_index", ""))
        content_hash = hashlib.sha1(result.document.page_content.encode("utf-8")).hexdigest()
        key = (source, chunk or content_hash)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def clear_lexical_cache() -> None:
    with _LEXICAL_CACHE_LOCK:
        _LEXICAL_CACHE.clear()
        _BM25_CACHE.clear()


def _lexical_corpus(store: Chroma) -> list[tuple[str, dict[str, object]]]:
    collection = getattr(store, "_collection", None)
    collection_key = str(getattr(collection, "name", id(store)))
    try:
        count = int(collection.count()) if collection is not None else -1
    except Exception:  # pragma: no cover - depends on Chroma backend
        count = -1
    with _LEXICAL_CACHE_LOCK:
        cached = _LEXICAL_CACHE.get(collection_key)
        if cached and cached[0] == count:
            return cached[1]
    try:
        payload = store.get(include=["documents", "metadatas"])
    except Exception:  # pragma: no cover - depends on Chroma backend
        return []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    corpus = [
        (str(content), dict(metadata or {}))
        for content, metadata in zip(documents, metadatas)
        if content
    ]
    with _LEXICAL_CACHE_LOCK:
        _LEXICAL_CACHE[collection_key] = (count, corpus)
    return corpus


def _heuristic_lexical_candidates(store: Chroma, question: str, limit: int = 6) -> list[RetrievalResult]:
    """Legacy exact-term candidates retained for controlled comparisons."""
    corpus = _lexical_corpus(store)
    expanded_question = _expand_query(question)
    alias_query = expanded_question[len(question):].strip() if expanded_question != question else ""
    ranked: list[tuple[float, RetrievalResult]] = []
    for content, metadata in corpus:
        metadata = metadata or {}
        source = str(metadata.get("source", "unknown"))
        searchable_text = _searchable_text(content, metadata)
        lexical_score = _lexical_score(question, searchable_text)
        if alias_query:
            lexical_score = max(lexical_score, _lexical_score(alias_query, searchable_text))
        source_tokens = " ".join(re.findall(r"[a-z0-9]+", source.lower()))
        query_tokens = re.findall(r"[a-z0-9]+", expanded_question.lower())
        if len(query_tokens) >= 3:
            for size in range(min(6, len(query_tokens)), 2, -1):
                if any(
                    " ".join(query_tokens[index:index + size]) in source_tokens
                    for index in range(len(query_tokens) - size + 1)
                ):
                    lexical_score = max(lexical_score, 0.75)
                    break
        if lexical_score <= 0.25:
            continue
        ranked.append(
            (
                lexical_score,
                RetrievalResult(
                    document=Document(page_content=str(content), metadata=dict(metadata)),
                    score=0.0,
                    elapsed_ms=0.0,
                    reranker_provider="lexical_candidate",
                    retrieval_method="lexical",
                ),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked[:limit]]


def _bm25_index(store: Chroma) -> _BM25Index:
    collection = getattr(store, "_collection", None)
    collection_key = str(getattr(collection, "name", id(store)))
    try:
        count = int(collection.count()) if collection is not None else -1
    except Exception:  # pragma: no cover - depends on Chroma backend
        count = -1
    with _LEXICAL_CACHE_LOCK:
        cached = _BM25_CACHE.get(collection_key)
        if cached and cached[0] == count:
            return cached[1]

    corpus = _lexical_corpus(store)
    tokenized_documents: list[list[str]] = []
    document_frequency: dict[str, int] = {}
    for content, metadata in corpus:
        # Keep the BM25 body score about the chunk itself. Repeating the same
        # title/path tokens in every chunk makes short heading-only chunks
        # look more relevant than the paragraphs that contain the answer.
        # Metadata is added separately as a bounded title bonus below.
        tokens = _token_list(content)
        tokenized_documents.append(tokens)
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    average_length = (
        sum(len(tokens) for tokens in tokenized_documents) / len(tokenized_documents)
        if tokenized_documents
        else 1.0
    )
    index = _BM25Index(
        documents=corpus,
        tokenized_documents=tokenized_documents,
        document_frequency=document_frequency,
        average_document_length=max(average_length, 1.0),
    )
    with _LEXICAL_CACHE_LOCK:
        _BM25_CACHE[collection_key] = (count, index)
    return index


def _bm25_candidates(store: Chroma, question: str, limit: int = 6) -> list[RetrievalResult]:
    """Return BM25 candidates with title/path metadata included in the corpus."""
    index = _bm25_index(store)
    if not index.documents or limit <= 0:
        return []
    query_tokens = _token_list(_expand_query(question))
    query_terms = set(query_tokens)
    if not query_terms:
        return []
    k1 = max(_float_env("BM25_K1", 1.5), 0.1)
    b = min(max(_float_env("BM25_B", 0.75), 0.0), 1.0)
    title_boost = max(_float_env("BM25_TITLE_BOOST", 0.2), 0.0)
    total_documents = len(index.documents)
    scored: list[tuple[float, RetrievalResult]] = []
    for position, (content, metadata) in enumerate(index.documents):
        tokens = index.tokenized_documents[position]
        frequencies: dict[str, int] = {}
        for token in tokens:
            if token in query_terms:
                frequencies[token] = frequencies.get(token, 0) + 1
        document_length = max(len(tokens), 1)
        score = 0.0
        for term, frequency in frequencies.items():
            document_frequency = index.document_frequency.get(term, 0)
            idf = math.log(
                1.0 + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * document_length / index.average_document_length
            )
            score += idf * frequency * (k1 + 1.0) / max(denominator, 1e-9)

        metadata_terms = set(
            _token_list(
                " ".join(
                    str(metadata.get(key, ""))
                    for key in ("title", "source", "directory", "filename")
                )
            )
        )
        title_overlap = len(query_terms & metadata_terms) / max(len(query_terms), 1)
        score += title_boost * title_overlap
        if score <= 0.0:
            continue
        scored.append(
            (
                score,
                RetrievalResult(
                    document=Document(page_content=str(content), metadata=dict(metadata)),
                    score=0.0,
                    elapsed_ms=0.0,
                    reranker_provider="bm25_candidate",
                    retrieval_method="bm25",
                    bm25_score=round(score, 6),
                ),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in scored[:limit]]


def _lexical_candidates(store: Chroma, question: str, limit: int = 6) -> list[RetrievalResult]:
    provider = (os.getenv("LEXICAL_PROVIDER") or "bm25").strip().lower()
    if provider in {"heuristic", "legacy", "simple"}:
        return _heuristic_lexical_candidates(store, question, limit)
    return _bm25_candidates(store, question, limit)


def _result_key(result: RetrievalResult) -> tuple[str, str]:
    metadata = result.document.metadata
    source = str(metadata.get("source", "unknown"))
    chunk = str(metadata.get("chunk_index", ""))
    content_hash = hashlib.sha1(result.document.page_content.encode("utf-8")).hexdigest()
    return source, chunk or content_hash


def _rrf_fuse(
    dense_results: list[RetrievalResult],
    lexical_results: list[RetrievalResult],
    limit: int,
) -> list[RetrievalResult]:
    """Fuse dense and BM25 ranks without comparing incompatible raw scores."""
    rrf_k = max(_int_env("RRF_K", 60), 1)
    merged: dict[tuple[str, str], RetrievalResult] = {}
    scores: dict[tuple[str, str], float] = {}
    for ranked_results in (dense_results, lexical_results):
        for rank, result in enumerate(ranked_results, start=1):
            key = _result_key(result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            previous = merged.get(key)
            if previous is None:
                merged[key] = result
            elif previous.bm25_score is None and result.bm25_score is not None:
                merged[key] = replace(previous, bm25_score=result.bm25_score)
    fused = [
        replace(result, fusion_score=scores[key])
        for key, result in merged.items()
    ]
    fused.sort(
        key=lambda item: (item.fusion_score or 0.0, item.score, item.bm25_score or 0.0),
        reverse=True,
    )
    return fused[:max(limit, 1)]


def _source_intro_candidates(
    store: Chroma,
    candidates: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Add one substantive opening chunk for the strongest candidate sources.

    Long essays often have short section-heading chunks. If retrieval lands on
    such a heading, the opening paragraph of the same source usually contains
    the definition or thesis needed by the answer model. The supplement is
    deliberately capped before reranking so it does not become a full-document
    scan or a second unbounded retrieval pass.
    """
    if not _bool_env("SOURCE_INTRO_ENABLED", True) or not candidates:
        return candidates
    source_limit = max(_int_env("SOURCE_INTRO_SOURCE_LIMIT", 3), 0)
    min_chars = max(_int_env("SOURCE_INTRO_MIN_CHARS", 180), 1)
    if source_limit <= 0:
        return candidates
    source_order: list[str] = []
    for result in candidates:
        source = str(result.document.metadata.get("source", "unknown"))
        if source not in source_order:
            source_order.append(source)
        if len(source_order) >= source_limit:
            break
    existing = {_result_key(result) for result in candidates}
    corpus = _lexical_corpus(store)
    by_source: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for content, metadata in corpus:
        source = str(metadata.get("source", "unknown"))
        if source in source_order:
            by_source.setdefault(source, []).append((content, metadata))

    supplements: list[RetrievalResult] = []
    for source in source_order:
        rows = sorted(
            by_source.get(source, []),
            key=lambda item: int(item[1].get("chunk_index", 0) or 0),
        )
        for content, metadata in rows:
            key = (
                source,
                str(metadata.get("chunk_index", ""))
                or hashlib.sha1(content.encode("utf-8")).hexdigest(),
            )
            if len(content) < min_chars or key in existing:
                continue
            supplements.append(
                RetrievalResult(
                    document=Document(page_content=content, metadata=dict(metadata)),
                    score=0.0,
                    elapsed_ms=0.0,
                    reranker_provider="source_intro_candidate",
                    retrieval_method="source_intro",
                )
            )
            existing.add(key)
            break
    return candidates + supplements


def _context_text(rows: list[tuple[str, dict[str, object]]]) -> str:
    """Join ordered child chunks into one parent context without huge gaps."""
    parts: list[str] = []
    for content, _ in rows:
        text = str(content).strip()
        if not text:
            continue
        if parts:
            previous = parts[-1]
            overlap_limit = min(len(previous), len(text), 120)
            overlap = 0
            for size in range(overlap_limit, 19, -1):
                if previous[-size:] == text[:size]:
                    overlap = size
                    break
            text = text[overlap:]
        if text:
            parts.append(text)
    return "\n".join(parts)


def _context_result(
    content: str,
    metadata: dict[str, object],
    anchor: RetrievalResult,
    *,
    role: str,
    distance: int | None = None,
) -> RetrievalResult:
    context_metadata = {
        **metadata,
        "context_role": role,
        "context_anchor_chunk": anchor.document.metadata.get("chunk_index", ""),
    }
    if distance is not None:
        context_metadata["context_distance"] = distance
    return RetrievalResult(
        document=Document(page_content=content, metadata=context_metadata),
        score=0.0,
        elapsed_ms=anchor.elapsed_ms,
        reranker_provider="context_expansion",
        retrieval_method=role,
    )


def _expand_parent_neighbor_context(
    store: Chroma,
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Expand primary hits with one parent group and nearby child chunks.

    Expansion is deliberately separate from ``results``: vector/reranker
    scores and Recall@K continue to describe actual retrieval hits, while the
    answer model receives enough contiguous context to resolve split facts.
    """
    if not results or not _bool_env("CONTEXT_EXPANSION_ENABLED", True):
        return list(results)
    radius = max(_int_env("CONTEXT_NEIGHBOR_RADIUS", 1), 0)
    max_sources = max(_int_env("CONTEXT_MAX_SOURCES", 3), 1)
    max_context = max(_int_env("CONTEXT_MAX_CHUNKS", 10), len(results))
    corpus = _lexical_corpus(store)
    rows_by_key: dict[tuple[str, str, str], list[tuple[str, dict[str, object]]]] = {}
    rows_by_source: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for content, metadata in corpus:
        source = str(metadata.get("source", "unknown"))
        document_id = str(metadata.get("document_id") or source)
        parent_id = str(metadata.get("parent_id") or "")
        rows_by_key.setdefault((document_id, source, parent_id), []).append((content, metadata))
        rows_by_source.setdefault(source, []).append((content, metadata))

    context = list(results)
    existing = {_result_key(result) for result in context}
    source_order: list[str] = []
    for anchor in results:
        source = str(anchor.document.metadata.get("source", "unknown"))
        if source not in source_order:
            source_order.append(source)
        if len(source_order) >= max_sources:
            break

    for anchor in results:
        if len(context) >= max_context:
            break
        metadata = anchor.document.metadata
        source = str(metadata.get("source", "unknown"))
        if source not in source_order:
            continue
        document_id = str(metadata.get("document_id") or source)
        parent_id = str(metadata.get("parent_id") or "")
        parent_rows = rows_by_key.get((document_id, source, parent_id), [])
        parent_rows = sorted(
            parent_rows,
            key=lambda item: _metadata_int(item[1], "chunk_index")
            if _metadata_int(item[1], "chunk_index") is not None
            else 0,
        )
        # A parent is a bounded group of child chunks, not an entire essay.
        # It supplies continuity without making one hit consume the context.
        if len(parent_rows) > 1 and parent_id:
            parent_content = _context_text(parent_rows)
            parent_metadata = dict(parent_rows[0][1])
            parent_metadata["parent_id"] = parent_id
            parent_metadata["parent_chunk_index"] = parent_rows[0][1].get("parent_chunk_index", "")
            parent_metadata["chunk_index"] = metadata.get("chunk_index", "")
            parent_key = (source, f"parent:{parent_id}")
            if parent_content and parent_key not in existing and len(context) < max_context:
                context.append(
                    _context_result(parent_content, parent_metadata, anchor, role="parent")
                )
                existing.add(parent_key)

        if radius <= 0 or len(context) >= max_context:
            continue
        rows = sorted(
            rows_by_source.get(source, []),
            key=lambda item: _metadata_int(item[1], "chunk_index")
            if _metadata_int(item[1], "chunk_index") is not None
            else 0,
        )
        by_index = {
            index: row
            for row in rows
            if (index := _metadata_int(row[1], "chunk_index")) is not None
        }
        anchor_index = _metadata_int(metadata, "chunk_index")
        if anchor_index is None:
            continue
        for distance in range(1, radius + 1):
            if len(context) >= max_context:
                break
            for neighbor_index in (anchor_index - distance, anchor_index + distance):
                row = by_index.get(neighbor_index)
                if row is None:
                    continue
                content, neighbor_metadata = row
                key = _result_key(
                    RetrievalResult(
                        document=Document(page_content=content, metadata=neighbor_metadata),
                        score=0.0,
                        elapsed_ms=0.0,
                    )
                )
                if key in existing:
                    continue
                context.append(
                    _context_result(
                        content,
                        dict(neighbor_metadata),
                        anchor,
                        role="neighbor",
                        distance=distance,
                    )
                )
                existing.add(key)
                if len(context) >= max_context:
                    break
    return context


def _is_fragment_text(text: str) -> bool:
    """True for heading-only or ultra-short chunks that rarely hold an answer."""
    compact = " ".join(str(text or "").split())
    if not compact:
        return True
    max_chars = max(_int_env("FRAGMENT_MAX_CHARS", 100), 20)
    if len(compact) < max_chars:
        return True
    # Numbered section titles such as "9. Get ramen profitable."
    if re.match(r"^\d+\.\s+\S.{0,100}$", compact) and compact.count(".") <= 2:
        return True
    return False


def _corpus_rows_for_source(
    store: Chroma,
    source: str,
) -> list[tuple[str, dict[str, object]]]:
    rows = [
        (content, metadata)
        for content, metadata in _lexical_corpus(store)
        if str(metadata.get("source", "unknown")) == source
    ]
    return sorted(rows, key=lambda item: int(item[1].get("chunk_index", 0) or 0))


def _substantive_chunk_for_source(
    store: Chroma,
    source: str,
    *,
    prefer_after: int | None = None,
    min_chars: int | None = None,
    exclude_keys: set[tuple[str, str]] | None = None,
) -> tuple[str, dict[str, object]] | None:
    """Pick a non-fragment body chunk from the same source for repair/context."""
    min_len = max(min_chars or _int_env("SOURCE_INTRO_MIN_CHARS", 180), 40)
    exclude = exclude_keys or set()
    rows = _corpus_rows_for_source(store, source)
    if not rows:
        return None

    ordered: list[tuple[str, dict[str, object]]] = []
    if prefer_after is not None:
        ordered.extend(
            row
            for row in rows
            if int(row[1].get("chunk_index", -1) or -1) in {prefer_after + 1, prefer_after - 1}
        )
        ordered.extend(
            row
            for row in rows
            if int(row[1].get("chunk_index", -1) or -1) >= (prefer_after + 1)
        )
    ordered.extend(rows)

    seen: set[tuple[str, str]] = set()
    for content, metadata in ordered:
        key = (
            source,
            str(metadata.get("chunk_index", ""))
            or hashlib.sha1(str(content).encode("utf-8")).hexdigest(),
        )
        if key in seen or key in exclude:
            continue
        seen.add(key)
        if len(str(content).strip()) < min_len or _is_fragment_text(content):
            continue
        return str(content), dict(metadata)
    return None


def _repair_fragment_results(
    store: Chroma,
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Replace heading-only primary hits with a substantive same-source chunk.

    Ranking scores stay with the original hit so Recall@K remains honest about
    what retrieval ranked; only the answer-facing text is repaired.
    """
    if not results or not _bool_env("FRAGMENT_PRIMARY_REPAIR_ENABLED", True):
        return results
    repaired: list[RetrievalResult] = []
    used = {_result_key(result) for result in results}
    for result in results:
        text = result.document.page_content
        if not _is_fragment_text(text):
            repaired.append(result)
            continue
        source = str(result.document.metadata.get("source", "unknown"))
        chunk_index = _metadata_int(result.document.metadata, "chunk_index")
        alternative = _substantive_chunk_for_source(
            store,
            source,
            prefer_after=chunk_index,
            exclude_keys=used,
        )
        if alternative is None:
            repaired.append(result)
            continue
        content, metadata = alternative
        metadata = {
            **metadata,
            "repaired_from_fragment": True,
            "original_chunk_index": result.document.metadata.get("chunk_index", ""),
            "original_excerpt": " ".join(text.split())[:160],
        }
        key = (
            source,
            str(metadata.get("chunk_index", ""))
            or hashlib.sha1(content.encode("utf-8")).hexdigest(),
        )
        used.add(key)
        repaired.append(
            replace(
                result,
                document=Document(page_content=content, metadata=metadata),
                retrieval_method=(
                    f"{result.retrieval_method}+fragment_repair"
                    if result.retrieval_method
                    else "fragment_repair"
                ),
            )
        )
    return repaired


def assess_evidence(
    results: list[RetrievalResult],
    *,
    candidate_count: int,
    reranker_provider: str,
    reranker_enabled: bool,
    rerank_ms: float,
    reranker_error: str | None = None,
) -> EvidenceAssessment:
    if not results:
        return EvidenceAssessment(
            status="insufficient",
            top_score=None,
            score_gap=None,
            reason_code="no_results",
            reranker_enabled=reranker_enabled,
            reranker_provider=reranker_provider,
            candidate_count=candidate_count,
            rerank_ms=rerank_ms,
            reranker_error=reranker_error,
        )

    scores = [
        result.rerank_score if result.rerank_score is not None else result.score
        for result in results
    ]
    top_score = scores[0]
    score_gap = top_score - scores[1] if len(scores) > 1 else None
    # Calibrated on the development set (2026-08-12): knowledge hits sit at >=0.333 while
    # out-of-scope tops out near 0.21. Defaults keep a buffer above OOS.
    sufficient_threshold = _float_env("RERANKER_SUFFICIENT_THRESHOLD", 0.52)
    review_threshold = _float_env("RERANKER_REVIEW_THRESHOLD", 0.30)
    margin_threshold = _float_env("RERANKER_MIN_MARGIN", 0.04)
    soft_floor = _float_env("RERANKER_SOFT_REVIEW_FLOOR", 0.28)

    primary_source = str(results[0].document.metadata.get("source", ""))
    same_source_count = sum(
        1
        for result in results
        if str(result.document.metadata.get("source", "")) == primary_source
    )
    primary_is_fragment = _is_fragment_text(results[0].document.page_content)
    primary_repaired = bool(results[0].document.metadata.get("repaired_from_fragment"))

    if top_score < review_threshold:
        # A single essay dominating the top ranks is a stronger signal than a
        # lone weak match. Promote only into review (never sufficient) so OOS
        # cases with mixed sources and low scores stay refused.
        if (
            _bool_env("EVIDENCE_SAME_SOURCE_SUPPORT", True)
            and top_score >= soft_floor
            and same_source_count >= min(2, len(results))
            and len(results) >= 2
        ):
            status: EvidenceStatus = "review"
            reason_code = "same_source_support"
        else:
            status = "insufficient"
            reason_code = "low_relevance"
    elif top_score < sufficient_threshold:
        status = "review"
        reason_code = "borderline_relevance"
    elif (
        score_gap is not None
        and score_gap < margin_threshold
        and str(results[0].document.metadata.get("source", ""))
        != str(results[1].document.metadata.get("source", ""))
    ):
        status = "review"
        reason_code = "ambiguous_top_results"
    else:
        status = "sufficient"
        reason_code = "strong_relevance"

    # Heading-only evidence is not enough to claim full sufficiency even when
    # the reranker score is high; keep it in review until fragment repair or
    # neighbor context supplies a real body paragraph.
    if status == "sufficient" and primary_is_fragment and not primary_repaired:
        status = "review"
        reason_code = "fragment_primary"

    # A heuristic fallback is useful for ordering, but it is not calibrated
    # enough to claim full evidence sufficiency on its own.
    if status == "sufficient" and not reranker_enabled:
        status = "review"
        reason_code = "heuristic_only"

    return EvidenceAssessment(
        status=status,
        top_score=round(top_score, 4),
        score_gap=round(score_gap, 4) if score_gap is not None else None,
        reason_code=reason_code,
        reranker_enabled=reranker_enabled,
        reranker_provider=reranker_provider,
        candidate_count=candidate_count,
        rerank_ms=round(rerank_ms, 1),
        reranker_error=reranker_error,
    )


def _diversify_results(results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    if not _bool_env("SOURCE_DIVERSITY_ENABLED", True):
        return results[:top_k]
    max_per_source = max(_int_env("MAX_CHUNKS_PER_SOURCE", 2), 1)
    selected: list[RetrievalResult] = []
    selected_keys: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}

    # When the strongest source is clearly ahead, keep one additional chunk
    # from it before enforcing cross-source diversity. This preserves the
    # definition plus its supporting paragraph without letting near-ties from
    # one source crowd out independent evidence.
    if results and top_k > 1 and max_per_source > 1:
        primary = results[0]
        primary_source = str(primary.document.metadata.get("source", "unknown"))
        primary_score = primary.rerank_score if primary.rerank_score is not None else primary.score
        continuation_margin = max(_float_env("PRIMARY_SOURCE_SCORE_MARGIN", 0.08), 0.0)
        source_advantage = max(_float_env("PRIMARY_SOURCE_ADVANTAGE_MARGIN", 0.15), 0.0)
        other_scores = [
            result.rerank_score if result.rerank_score is not None else result.score
            for result in results[1:]
            if str(result.document.metadata.get("source", "unknown")) != primary_source
        ]
        primary_is_clear_leader = not other_scores or primary_score - max(other_scores) >= source_advantage
        selected.append(primary)
        selected_keys.add(_result_key(primary))
        counts[primary_source] = 1
        for result in results[1:]:
            source = str(result.document.metadata.get("source", "unknown"))
            score = result.rerank_score if result.rerank_score is not None else result.score
            if source != primary_source or _result_key(result) in selected_keys:
                continue
            if primary_is_clear_leader and score >= primary_score - continuation_margin:
                selected.append(result)
                selected_keys.add(_result_key(result))
                counts[primary_source] = 2
            break

    # First pass gives every available source a chance to enter the answer context.
    for result in results:
        source = str(result.document.metadata.get("source", "unknown"))
        key = _result_key(result)
        if key in selected_keys or source in counts:
            continue
        selected.append(result)
        selected_keys.add(key)
        counts[source] = 1
        if len(selected) >= top_k:
            return selected[:top_k]

    # Then allow adjacent chunks from the strongest sources.
    for result in results:
        source = str(result.document.metadata.get("source", "unknown"))
        key = _result_key(result)
        if key in selected_keys or counts.get(source, 0) >= max_per_source:
            continue
        selected.append(result)
        selected_keys.add(key)
        counts[source] = counts.get(source, 0) + 1
        if len(selected) >= top_k:
            return selected[:top_k]

    # If the corpus has only one relevant source, do not return fewer than top_k.
    for result in results:
        key = _result_key(result)
        if key in selected_keys:
            continue
        selected.append(result)
        selected_keys.add(key)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def _rewrite_mode() -> str:
    """Return retry rewrite mode: rule (default), alias, or off."""
    return (os.getenv("RETRIEVAL_REWRITE_MODE") or "rule").strip().lower()


def _retry_query(question: str) -> str | None:
    if not _bool_env("RETRIEVAL_RETRY_ENABLED", True):
        return None
    mode = _rewrite_mode()
    if mode in {"0", "off", "false", "disabled", "none"}:
        return None
    if mode in {"alias", "expand"}:
        expanded = _expand_query(question)
        return expanded if expanded != question else None
    # Default: rule rewrite (aliases + multi-signal English topics).
    rewritten = _rule_rewrite_query(question)
    if rewritten:
        return rewritten
    expanded = _expand_query(question)
    return expanded if expanded != question else None


def _should_retry(report: RetrievalReport, retry_query: str | None) -> bool:
    """Decide whether a second retrieval pass is worth the latency.

    Default policy (latency-first after development-set calibration):
    - never retry when already ``sufficient``
    - retry ``insufficient`` only when the score is not hopelessly low
    - do **not** retry ordinary ``review`` cases (they are already answerable);
      set ``RETRIEVAL_RETRY_ON_REVIEW=1`` to restore the old low-score review retry
    """
    if not retry_query:
        return False
    if report.evidence.status == "sufficient":
        return False

    top_score = report.evidence.top_score
    hopeless = _float_env("RETRIEVAL_RETRY_HOPELESS", 0.12)
    if top_score is not None and top_score < hopeless:
        # Extremely weak matches (typical OOS) almost never improve enough to
        # justify a full second hybrid + rerank pass.
        return False

    if report.evidence.status == "insufficient":
        return True

    if report.evidence.status != "review":
        return False

    # Review is already in the answerable band after threshold calibration.
    if not _bool_env("RETRIEVAL_RETRY_ON_REVIEW", False):
        return False
    threshold = _float_env("RETRIEVAL_RETRY_SCORE", 0.35)
    return top_score is not None and top_score < threshold


def _retrieve_once(
    store: Chroma,
    question: str,
    top_k: int,
    *,
    rerank_question: str | None = None,
    widen: bool = False,
) -> RetrievalReport:
    started = time.perf_counter()
    candidate_multiplier = max(_int_env("RETRIEVAL_CANDIDATE_MULTIPLIER", 3), 1)
    candidate_min = max(_int_env("RETRIEVAL_CANDIDATE_MIN", 9), top_k)
    candidate_max = max(_int_env("RETRIEVAL_CANDIDATE_MAX", 18), candidate_min)
    if widen and _bool_env("RETRIEVAL_RETRY_WIDEN", True):
        candidate_min = max(candidate_min, _int_env("RETRIEVAL_RETRY_CANDIDATE_MIN", 12))
        candidate_max = max(candidate_max, _int_env("RETRIEVAL_RETRY_CANDIDATE_MAX", 28))
    candidate_k = min(max(top_k * candidate_multiplier, candidate_min), candidate_max)
    vector_started = time.perf_counter()
    matches = store.similarity_search_with_score(question, k=candidate_k)
    vector_ms = (time.perf_counter() - vector_started) * 1000
    candidates = [
        RetrievalResult(
            document=document,
            score=1 / (1 + max(float(distance), 0.0)),
            elapsed_ms=0.0,
            dense_distance=float(distance),
        )
        for document, distance in matches
    ]
    dense_candidates = _deduplicate(candidates)
    lexical_limit = max(_int_env("BM25_CANDIDATE_LIMIT", 6), 0)
    if widen and _bool_env("RETRIEVAL_RETRY_WIDEN", True):
        lexical_limit = max(lexical_limit, _int_env("RETRIEVAL_RETRY_BM25_LIMIT", 10))
    lexical_candidates = _lexical_candidates(store, question, limit=lexical_limit)
    if _bool_env("HYBRID_RETRIEVAL_ENABLED", True):
        hybrid_pool = max(
            _int_env("HYBRID_CANDIDATE_POOL", candidate_max),
            top_k,
        )
        if widen and _bool_env("RETRIEVAL_RETRY_WIDEN", True):
            hybrid_pool = max(hybrid_pool, _int_env("RETRIEVAL_RETRY_HYBRID_POOL", 28))
        candidates = _rrf_fuse(dense_candidates, lexical_candidates, hybrid_pool)
    else:
        candidates = _deduplicate(dense_candidates + lexical_candidates)
    candidates = _source_intro_candidates(store, candidates)
    reranked, provider, enabled, rerank_ms, reranker_error = _rerank(
        rerank_question or question,
        candidates,
    )
    results = _diversify_results(reranked, top_k)
    results = _repair_fragment_results(store, results)
    total_ms = (time.perf_counter() - started) * 1000
    results = [replace(result, elapsed_ms=total_ms) for result in results]
    evidence = assess_evidence(
        results,
        candidate_count=len(candidates),
        reranker_provider=provider,
        reranker_enabled=enabled,
        rerank_ms=rerank_ms,
        reranker_error=reranker_error,
    )
    context_results = _expand_parent_neighbor_context(store, results)
    return RetrievalReport(
        results=results,
        evidence=evidence,
        vector_ms=round(vector_ms, 1),
        rerank_ms=round(rerank_ms, 1),
        total_ms=round(total_ms, 1),
        context_results=context_results,
    )


def _merge_retry_reports(
    store: Chroma,
    first: RetrievalReport,
    retry: RetrievalReport,
    question: str,
    retry_query: str,
    top_k: int,
) -> RetrievalReport:
    merged = _deduplicate(first.results + retry.results)
    merged.sort(
        key=lambda item: (
            item.rerank_score if item.rerank_score is not None else item.score,
            item.fusion_score or 0.0,
        ),
        reverse=True,
    )
    keep_original = min(
        max(_int_env("RETRIEVAL_RETRY_KEEP_ORIGINAL", 1), 0),
        top_k,
        len(first.results),
    )
    preserved = first.results[:keep_original]
    preserved_keys = {_result_key(result) for result in preserved}
    supplemental = [result for result in merged if _result_key(result) not in preserved_keys]
    results = preserved + _diversify_results(supplemental, top_k - len(preserved))
    results.sort(
        key=lambda item: (
            item.rerank_score if item.rerank_score is not None else item.score,
            item.fusion_score or 0.0,
        ),
        reverse=True,
    )
    results = _repair_fragment_results(store, results)
    total_ms = first.total_ms + retry.total_ms
    results = [replace(result, elapsed_ms=total_ms) for result in results]
    reranker_enabled = first.evidence.reranker_enabled and retry.evidence.reranker_enabled
    reranker_provider = (
        "cross_encoder"
        if reranker_enabled
        else retry.evidence.reranker_provider or first.evidence.reranker_provider
    )
    evidence = assess_evidence(
        results,
        candidate_count=len(merged),
        reranker_provider=reranker_provider,
        reranker_enabled=reranker_enabled,
        rerank_ms=first.rerank_ms + retry.rerank_ms,
        reranker_error=retry.evidence.reranker_error or first.evidence.reranker_error,
    )
    context_results = _expand_parent_neighbor_context(store, results)
    return RetrievalReport(
        results=results,
        evidence=evidence,
        vector_ms=round(first.vector_ms + retry.vector_ms, 1),
        rerank_ms=round(first.rerank_ms + retry.rerank_ms, 1),
        total_ms=round(total_ms, 1),
        retry_count=1,
        retry_query=retry_query,
        context_results=context_results,
    )


def retrieve_report(store: Chroma, question: str, top_k: int) -> RetrievalReport:
    first = _retrieve_once(store, question, top_k, rerank_question=_rerank_query(question))
    retry_query = _retry_query(question)
    if not _should_retry(first, retry_query):
        return first
    retry_top_k = max(_int_env("RETRIEVAL_RETRY_TOP_K", max(top_k * 2, 6)), top_k)
    # Search with the rewritten English-heavy query; rerank with original + terms
    # so the cross-encoder still sees the user's language and intent.
    retry = _retrieve_once(
        store,
        retry_query or question,
        retry_top_k,
        rerank_question=_rerank_query(question),
        widen=True,
    )
    return _merge_retry_reports(store, first, retry, question, retry_query or question, top_k)


def retrieve(store: Chroma, question: str, top_k: int) -> list[RetrievalResult]:
    return retrieve_report(store, question, top_k).results


def format_context(results: list[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[Source: {result.document.metadata.get('source', 'unknown')} | "
        f"chunk {result.document.metadata.get('chunk_index', '?')} | "
        f"context {result.document.metadata.get('context_role', 'primary')}]\n"
        f"{result.document.page_content}"
        for result in results
    )


def print_retrieval_diagnostics(results: list[RetrievalResult]) -> None:
    print("\nRetrieval results")
    if not results:
        print("  No matching chunks")
        return
    for rank, result in enumerate(results, start=1):
        metadata = result.document.metadata
        excerpt = re.sub(r"\s+", " ", result.document.page_content).strip()
        if len(excerpt) > 220:
            excerpt = f"{excerpt[:220]}..."
        rerank = "-" if result.rerank_score is None else f"{result.rerank_score:.3f}"
        print(
            f"  [{rank}] vector {result.score:.3f} | reranker {rerank} | "
            f"{metadata.get('source', 'unknown')} | chunk {metadata.get('chunk_index', '?')}"
        )
        print(f"      {excerpt}")
    print(f"  Retrieval time: {results[0].elapsed_ms:.0f} ms")
