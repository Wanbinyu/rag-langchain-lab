from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path


GLOSSARY_FILENAME = "retrieval_glossary.json"


@dataclass(frozen=True)
class RetrievalGlossary:
    """Corpus-side lexical aids. Empty means the generic retrieval path."""

    aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    rewrite_rules: tuple[tuple[tuple[str, ...], int, tuple[str, ...]], ...] = ()
    path: Path | None = None

    @property
    def empty(self) -> bool:
        return not self.aliases and not self.rewrite_rules


EMPTY_GLOSSARY = RetrievalGlossary()

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, RetrievalGlossary]] = {}
_DATA_DIR: Path | None = None


def configure_glossary(data_dir: Path | None = None) -> RetrievalGlossary:
    """Remember the active corpus directory used to auto-discover a glossary."""
    global _DATA_DIR
    with _LOCK:
        _DATA_DIR = Path(data_dir) if data_dir is not None else None
        if _DATA_DIR is not None:
            os.environ["RAG_DATA_DIR"] = str(_DATA_DIR)
        else:
            os.environ.pop("RAG_DATA_DIR", None)
        _CACHE.clear()
    return active_glossary()


def reset_glossary_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def expansion_enabled() -> bool:
    value = (os.getenv("QUERY_EXPANSION_ENABLED") or "1").strip().lower()
    return value not in {"0", "false", "off", "no", "disabled"}


def resolve_glossary_path(data_dir: Path | None = None) -> Path | None:
    explicit = (os.getenv("QUERY_GLOSSARY_PATH") or "").strip()
    if explicit.lower() in {"0", "false", "off", "no", "none", "disabled"}:
        return None
    if explicit:
        return Path(explicit)
    root = data_dir
    if root is None:
        root = _DATA_DIR
    if root is None:
        env_dir = (os.getenv("RAG_DATA_DIR") or "").strip()
        root = Path(env_dir) if env_dir else None
    if root is None:
        return None
    candidate = Path(root) / GLOSSARY_FILENAME
    return candidate if candidate.is_file() else None


def load_glossary(path: Path) -> RetrievalGlossary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"glossary must be a JSON object: {path}")
    aliases: list[tuple[str, tuple[str, ...]]] = []
    for item in payload.get("aliases") or []:
        phrase = str(item.get("phrase") or "").strip()
        terms = tuple(
            str(term).strip()
            for term in (item.get("terms") or [])
            if str(term).strip()
        )
        if phrase and terms:
            aliases.append((phrase, terms))
    rewrite_rules: list[tuple[tuple[str, ...], int, tuple[str, ...]]] = []
    for item in payload.get("rewrite_rules") or []:
        signals = tuple(
            str(signal).strip()
            for signal in (item.get("signals") or [])
            if str(signal).strip()
        )
        terms = tuple(
            str(term).strip()
            for term in (item.get("terms") or [])
            if str(term).strip()
        )
        try:
            min_hits = int(item.get("min_hits") or 1)
        except (TypeError, ValueError):
            min_hits = 1
        if signals and terms:
            rewrite_rules.append((signals, max(min_hits, 1), terms))
    return RetrievalGlossary(tuple(aliases), tuple(rewrite_rules), path.resolve())


def active_glossary() -> RetrievalGlossary:
    path = resolve_glossary_path()
    if path is None:
        return EMPTY_GLOSSARY
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"QUERY_GLOSSARY_PATH does not exist: {resolved}")
    key = str(resolved)
    mtime = resolved.stat().st_mtime
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        glossary = load_glossary(resolved)
        _CACHE[key] = (mtime, glossary)
        return glossary
