from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.loaders import list_directories, load_documents_report


class DirectoryDiscoveryTests(unittest.TestCase):
    def test_lists_nested_and_empty_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            (data_dir / "组件" / "按钮").mkdir(parents=True)
            (data_dir / "待整理").mkdir()
            self.assertEqual(list_directories(data_dir), ["待整理", "组件", "组件/按钮"])


    def test_scan_report_keeps_empty_file_as_an_explicit_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()
            (data_dir / "empty.md").write_text("", encoding="utf-8")
            (data_dir / "guide.md").write_text("# Guide\nUseful text", encoding="utf-8")
            documents, report = load_documents_report(data_dir)
            self.assertEqual(len(documents), 1)
            self.assertEqual(report.status, "warning")
            self.assertEqual(report.empty_files, ("empty.md",))

    def test_scan_skips_hidden_files_and_corpus_glossary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()
            (data_dir / "guide.md").write_text("# Guide\nUseful text", encoding="utf-8")
            (data_dir / ".secret.md").write_text("# Hidden\nShould not index", encoding="utf-8")
            (data_dir / "retrieval_glossary.json").write_text(
                '{"aliases": []}', encoding="utf-8"
            )
            documents, report = load_documents_report(data_dir)
            self.assertEqual([item.metadata["source"] for item in documents], ["guide.md"])
            self.assertEqual(report.scanned_files, 1)


if __name__ == "__main__":
    unittest.main()
