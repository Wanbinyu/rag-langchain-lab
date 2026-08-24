from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.embeddings import Embeddings

from src.config import project_settings
from src.indexing import check_index_integrity, index_status, open_store, sync_index
from src.loaders import load_documents


class FakeEmbeddings(Embeddings):
    """确定性向量，用于测试索引流程而不下载模型或调用 API。"""

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 997), 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class IncrementalIndexTests(unittest.TestCase):
    def test_add_skip_update_and_remove_a_file(self) -> None:
        # HNSW 文件在 Windows 上可能会在进程退出前保持短暂锁定。
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            root = Path(temp_dir.name)
            data_dir = root / "data"
            data_dir.mkdir()
            source = data_dir / "guide.md"
            source.write_text("按钮组件用于触发操作。" * 30, encoding="utf-8")
            settings = project_settings(root)

            first = sync_index(settings, FakeEmbeddings)
            self.assertEqual((first.added_files, first.updated_files), (1, 0))
            self.assertGreater(first.total_chunks, 0)
            payload = open_store(settings, FakeEmbeddings()).get(include=["metadatas"])
            metadata = payload["metadatas"][0]
            for key in (
                "document_id",
                "chunk_id",
                "chunk_count",
                "parent_id",
                "prev_chunk_id",
                "next_chunk_id",
                "char_start",
                "char_end",
            ):
                self.assertIn(key, metadata)
            integrity = check_index_integrity(settings, documents=load_documents(data_dir))
            self.assertEqual(integrity.status, "ok")

            unchanged = sync_index(settings, FakeEmbeddings)
            self.assertEqual((unchanged.added_files, unchanged.updated_files), (0, 0))
            self.assertEqual(unchanged.skipped_files, 1)

            source.write_text("标签组件用于显示文字。" * 30, encoding="utf-8")
            stale = check_index_integrity(settings, documents=load_documents(data_dir))
            self.assertEqual(stale.status, "error")
            self.assertIn("guide.md", stale.invalid_sources)
            updated = sync_index(settings, FakeEmbeddings)
            self.assertEqual((updated.added_files, updated.updated_files), (0, 1))

            source.unlink()
            removed = sync_index(settings, FakeEmbeddings)
            self.assertEqual(removed.removed_files, 1)
            self.assertEqual(index_status(settings), (0, 0))
        finally:
            # Chroma 在 Windows 上持有 HNSW 文件句柄，需要先清除进程级客户端。
            SharedSystemClient.clear_system_cache()
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
