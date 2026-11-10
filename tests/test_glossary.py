from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.glossary import (
    GLOSSARY_FILENAME,
    configure_glossary,
    load_glossary,
    reset_glossary_cache,
    resolve_glossary_path,
)


class GlossaryLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_glossary_cache()

    def tearDown(self) -> None:
        reset_glossary_cache()
        configure_glossary(None)

    def test_tracked_graham_glossary_loads_aliases_and_rewrite_rules(self) -> None:
        path = Path(__file__).resolve().parents[1] / "eval" / "glossaries" / "graham.json"
        glossary = load_glossary(path)
        self.assertGreater(len(glossary.aliases), 50)
        self.assertGreater(len(glossary.rewrite_rules), 5)
        phrases = {phrase for phrase, _ in glossary.aliases}
        self.assertIn("创业想法", phrases)

    def test_data_dir_auto_discovers_corpus_side_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()
            (data_dir / GLOSSARY_FILENAME).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "aliases": [{"phrase": "索引同步", "terms": ["incremental index"]}],
                        "rewrite_rules": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            configure_glossary(data_dir)
            self.assertEqual(resolve_glossary_path(), data_dir / GLOSSARY_FILENAME)

    def test_explicit_off_disables_auto_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / GLOSSARY_FILENAME).write_text("{}", encoding="utf-8")
            configure_glossary(data_dir)
            with patch.dict("os.environ", {"QUERY_GLOSSARY_PATH": "off"}, clear=False):
                reset_glossary_cache()
                self.assertIsNone(resolve_glossary_path())


if __name__ == "__main__":
    unittest.main()
