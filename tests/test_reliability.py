from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.documents import Document

from src import cli, indexing, retrieval, service, web
from src.config import project_settings
from src.loaders import DataIntegrityReport
from tests.test_indexing import FakeEmbeddings


@pytest.fixture
def knowledge_base(tmp_path):
    settings = project_settings(tmp_path)
    settings.data_dir.mkdir()
    source = settings.data_dir / "guide.md"
    source.write_text("Original reliable evidence. " * 30, encoding="utf-8")
    indexing.sync_index(settings, FakeEmbeddings)
    try:
        yield settings, source
    finally:
        retrieval.clear_lexical_cache()
        SharedSystemClient.clear_system_cache()


def snapshot(settings):
    payload = indexing.open_store(settings).get(include=["documents", "metadatas"])
    return {key: (text, metadata) for key, text, metadata in zip(
        payload["ids"], payload["documents"], payload["metadatas"]
    )}


def test_embedding_failure_preserves_existing_index(knowledge_base):
    settings, source = knowledge_base
    before = snapshot(settings)
    source.write_text("Updated content. " * 30, encoding="utf-8")
    factory = Mock(side_effect=RuntimeError("model unavailable"))
    with pytest.raises(RuntimeError, match="model unavailable"):
        indexing.sync_index(settings, factory)
    assert snapshot(settings) == before


def test_partial_write_rolls_back_source(knowledge_base):
    settings, source = knowledge_base
    before = snapshot(settings)
    source.write_text("Updated content. " * 30, encoding="utf-8")
    collection_type = type(indexing.open_store(settings)._collection)
    original = collection_type.upsert
    calls = 0

    def fail_once(collection, **kwargs):
        nonlocal calls
        calls += 1
        original(collection, **kwargs)
        if calls == 1:
            raise RuntimeError("interrupted write")

    with patch.object(collection_type, "upsert", fail_once):
        with pytest.raises(RuntimeError, match="interrupted write"):
            indexing.sync_index(settings, FakeEmbeddings)
    assert calls >= 2
    assert snapshot(settings) == before


def test_unreadable_source_is_not_deleted(knowledge_base):
    settings, _ = knowledge_base
    before = snapshot(settings)
    report = DataIntegrityReport(scanned_files=1, loaded_files=0, unreadable_files=("guide.md",))
    with patch.object(indexing, "load_documents_report", return_value=([], report)):
        result = indexing.sync_index(settings, FakeEmbeddings)
    assert result.removed_files == 0
    assert result.data_integrity_status == "error"
    assert snapshot(settings) == before


@pytest.mark.parametrize("name,value", [
    ("INDEX_INCLUDE_METADATA", "1"), ("PARENT_CHUNK_SIZE", "7"),
    ("LOCAL_EMBED_MODEL", "different-model"),
])
def test_index_configuration_changes_trigger_rebuild(knowledge_base, monkeypatch, name, value):
    settings, _ = knowledge_base
    before = snapshot(settings)
    monkeypatch.setenv(name, value)
    report = indexing.sync_index(settings, FakeEmbeddings)
    assert report.updated_files == 1
    assert report.index_integrity_status == "ok"
    assert set(snapshot(settings)).isdisjoint(before)


def test_rerank_cache_is_scoped_to_model(monkeypatch):
    retrieval.clear_rerank_pair_cache()
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANK_PAIR_CACHE_ENABLED", "1")
    first = Mock(); first.rerank.return_value = [0.0]
    second = Mock(); second.rerank.return_value = [5.0]
    result = retrieval.RetrievalResult(Document(page_content="evidence"), 0.5, 0.0)
    with patch.object(retrieval, "_cross_encoder", side_effect=lambda name: {"a": first, "b": second}[name]):
        monkeypatch.setenv("RERANKER_MODEL", "a")
        score_a = retrieval._rerank("question", [result])[0][0].rerank_score
        monkeypatch.setenv("RERANKER_MODEL", "b")
        score_b = retrieval._rerank("question", [result])[0][0].rerank_score
        retrieval._rerank("question", [result])
    assert score_b > score_a
    first.rerank.assert_called_once()
    second.rerank.assert_called_once()
    retrieval.clear_rerank_pair_cache()


def test_equal_size_collections_do_not_share_lexical_cache():
    retrieval.clear_lexical_cache()
    def store(text):
        return SimpleNamespace(
            _collection=SimpleNamespace(name="same", count=lambda: 1),
            get=lambda **kwargs: {"documents": [text], "metadatas": [{"source": "a.md"}]},
        )
    a, b = store("first database"), store("second database")
    assert retrieval._lexical_corpus(a)[0][0] == "first database"
    assert retrieval._lexical_corpus(b)[0][0] == "second database"


def test_external_equal_count_update_invalidates_cache(knowledge_base):
    settings, _ = knowledge_base
    store = indexing.open_store(settings)
    before = retrieval._lexical_corpus(store)
    chunk_id = store.get(include=[])["ids"][0]
    store._collection.update(ids=[chunk_id], documents=["Externally updated text"],
        embeddings=FakeEmbeddings().embed_documents(["Externally updated text"]))
    after = retrieval._lexical_corpus(store)
    assert before != after
    assert any(text == "Externally updated text" for text, _ in after)


def test_postings_bm25_matches_full_scan(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "0")
    monkeypatch.setenv("BM25_K1", "1.5")
    monkeypatch.setenv("BM25_B", "0.75")
    monkeypatch.setenv("BM25_TITLE_BOOST", "0.2")
    corpus = [
        ("alpha beta beta", {"source": "a.md"}),
        ("gamma gamma", {"source": "b.md", "title": "alpha"}),
        ("unrelated", {"source": "c.md"}),
        ("beta alpha alpha", {"source": "d.md"}),
    ]
    retrieval.clear_lexical_cache()
    store = SimpleNamespace(_collection=SimpleNamespace(name="bm25", count=lambda: len(corpus)),
        get=lambda **kwargs: {"documents": [x[0] for x in corpus], "metadatas": [x[1] for x in corpus]})
    index = retrieval._bm25_index(store)
    expected = {}
    for position, (text, metadata) in enumerate(corpus):
        frequencies = Counter(index.tokenized_documents[position])
        score = 0.0
        for term in ("alpha", "beta"):
            frequency = frequencies[term]
            df = index.document_frequency.get(term, 0)
            denominator = frequency + 1.5 * (0.25 + 0.75 * sum(frequencies.values()) / index.average_document_length)
            score += math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5)) * frequency * 2.5 / denominator
        metadata_terms = set(retrieval._token_list(" ".join(str(metadata.get(key, ""))
            for key in ("title", "source", "directory", "filename"))))
        score += 0.2 * len({"alpha", "beta"} & metadata_terms) / 2
        if score > 0:
            expected[metadata["source"]] = round(score, 6)
    actual = retrieval._bm25_candidates(store, "alpha beta", limit=10)
    assert {item.document.metadata["source"]: item.bm25_score for item in actual} == expected
    assert [item.bm25_score for item in actual] == sorted(expected.values(), reverse=True)


def report(status="insufficient"):
    evidence = retrieval.EvidenceAssessment(status=status, top_score=0.1, score_gap=None,
        reason_code="test", reranker_enabled=True, reranker_provider="cross_encoder",
        candidate_count=0, rerank_ms=0)
    return retrieval.RetrievalReport(results=[], evidence=evidence, vector_ms=0, rerank_ms=0, total_ms=0)


@pytest.mark.parametrize("question,status", [("Explain indexing", "insufficient"), ("今天的天气", "sufficient")])
def test_rejected_questions_never_construct_chat_client(question, status):
    with patch.object(service, "build_chat_model") as model:
        answer, elapsed, _ = service.guarded_answer(question, report(status))
    assert answer
    assert elapsed == 0
    model.assert_not_called()


def test_cli_uses_shared_gate():
    with patch.object(cli, "build_embeddings"), patch.object(cli, "open_store"), \
         patch.object(cli, "retrieve_report", return_value=report()), \
         patch.object(cli, "guarded_answer", return_value=("refused", 0, None)) as gate:
        cli.answer_once(project_settings(), "question", 3)
    gate.assert_called_once()


def test_local_embedding_instance_reused_by_model(monkeypatch):
    cli._local_embeddings.cache_clear()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBED_MODEL", "a")
    with patch.object(cli, "FastEmbedEmbeddings", side_effect=lambda model_name: object()) as factory:
        first = cli.build_embeddings()
        assert cli.build_embeddings() is first
        monkeypatch.setenv("LOCAL_EMBED_MODEL", "b")
        assert cli.build_embeddings() is not first
        assert factory.call_count == 2
    cli._local_embeddings.cache_clear()


def test_web_releases_index_lock_before_generation():
    handler = object.__new__(web.RAGDashboardHandler)
    handler.path = "/api/ask"
    handler.server = SimpleNamespace(settings=project_settings())
    handler._read_json = lambda: {"question": "question"}
    handler._send_json = Mock()

    def generate(*args):
        def acquire_from_other_thread():
            acquired = web.RAG_LOCK.acquire(timeout=0.1)
            if acquired:
                web.RAG_LOCK.release()
            return acquired
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(acquire_from_other_thread).result(timeout=2)
        return "answer", 0, SimpleNamespace(status="allowed", reason_code="test")

    with patch.object(web, "build_embeddings"), patch.object(web, "open_store"), \
         patch.object(web, "retrieve_report", return_value=report()), \
         patch.object(web, "guarded_answer", side_effect=generate) as gate, \
         patch.object(web, "keyword_groups_for_question", return_value=()):
        handler.do_POST()
    gate.assert_called_once()
    response = handler._send_json.call_args.args[0]
    assert "error" not in response
    assert response["answer"] == "answer"


def test_retrieval_total_includes_entire_pipeline():
    with patch.object(retrieval, "_retrieve_report", return_value=report()), \
         patch.object(retrieval.time, "perf_counter", side_effect=[1.0, 1.25]):
        result = retrieval.retrieve_report(Mock(), "question", 3)
    assert result.total_ms == 250


def test_warm_persistent_bm25_reuses_tokenization(knowledge_base):
    settings, _ = knowledge_base
    store = indexing.open_store(settings)
    first = retrieval._bm25_index(store)
    with patch.object(retrieval, "_token_list", side_effect=AssertionError("unexpected rebuild")):
        assert retrieval._bm25_index(indexing.open_store(settings)) is first


def test_allowed_question_generates_from_context_snapshot():
    from dataclasses import replace
    document = retrieval.RetrievalResult(Document(page_content="Evidence"), 0.5, 0)
    evidence = replace(report("sufficient"), context_results=[document])
    with patch.object(service, "build_chat_model", return_value="client"), \
         patch.object(service, "answer_question", return_value=("answer", 12)) as generate:
        answer, elapsed, boundary = service.guarded_answer("Explain indexing", evidence)
    generate.assert_called_once_with("client", "Explain indexing", [document])
    assert (answer, elapsed, boundary.blocked) == ("answer", 12, False)
