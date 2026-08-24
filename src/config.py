from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    chroma_dir: Path
    collection_name: str = "rag_lab_zh_en_v1"
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 3


def project_settings(root: Path | None = None) -> Settings:
    project_root = (root or Path(__file__).resolve().parents[1]).resolve()
    return Settings(
        root=project_root,
        data_dir=project_root / "data",
        chroma_dir=project_root / "chroma_db_zh_en",
    )
