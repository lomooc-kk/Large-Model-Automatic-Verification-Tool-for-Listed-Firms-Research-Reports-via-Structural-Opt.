import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pdfparse/src"))

from yjparse.corpus import catalog_summary, import_jsonl, load_catalog  # noqa: E402
from yjparse.retrieval import load_index  # noqa: E402


class CorpusCatalogTest(unittest.TestCase):
    def test_repository_catalog_is_valid_and_classified(self):
        catalog = load_catalog(ROOT / "config/financial_corpus_catalog.json")
        summary = catalog_summary(catalog)
        self.assertGreaterEqual(summary["sources"], 15)
        self.assertEqual(set(catalog["categories"]), set(summary["by_category"]))
        self.assertGreaterEqual(summary["by_role"]["production"], 10)

    def test_import_jsonl_builds_existing_kb_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "records.jsonl"
            source.write_text(json.dumps({"document_id": "a-1", "title": "Annual report",
                                          "content": "Revenue was 10.\n\nProfit was 2."}) + "\n",
                              encoding="utf-8")
            manifest = import_jsonl([source], root / "kb", source_id="official-demo",
                                    category="company_disclosure", role="production",
                                    project="financial-evidence", language="en")
            self.assertEqual(len(manifest["documents"]), 1)
            rows = load_index(root / "kb/kb_index.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["corpus_role"], "production")
            self.assertTrue(Path(manifest["documents"][0]["markdown"]).exists())

    def test_evaluation_requires_opt_in_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "inputs.test.jsonl"
            source.write_text('{"sample_id":"x","context":"safe input"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow-evaluation"):
                import_jsonl([source], Path(tmp) / "kb", source_id="bench", category="evaluation",
                             role="evaluation", project="financial-evidence")
            with self.assertRaisesRegex(ValueError, "eval-."):
                import_jsonl([source], Path(tmp) / "kb", source_id="bench", category="evaluation",
                             role="evaluation", project="financial-evidence", allow_evaluation=True)

    def test_gold_files_are_never_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "gold.test.jsonl"
            source.write_text('{"sample_id":"x","context":"answer"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gold/answer"):
                import_jsonl([source], Path(tmp) / "kb", source_id="bench", category="evaluation",
                             role="evaluation", project="eval-bench", allow_evaluation=True)


if __name__ == "__main__":
    unittest.main()
