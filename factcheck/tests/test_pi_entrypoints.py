"""Real DOCX parsing, evidence-rule recheck, and persisted Pi entrypoint artifacts.

The scheduler and retrieval responses are fixtures; this does not exercise a live
OpenViking vector index. Parsing, hash checks, fact rules, and output are real.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "factcheck/src"), str(ROOT / "pdfparse/src")]

from yjcheck.adapters import file_hash
from yjcheck.agent_workflow import run_agent_check
from yjcheck.model import ModelConfig
from yjcheck.model_runtime import RuntimeSettings
from yjcheck.pipeline import check_documents, verify_artifacts


def write_docx(path, paragraphs):
    document = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body>' + ''.join('<w:p><w:r><w:t>' + escape(text) + '</w:t></w:r></w:p>'
                                     for text in paragraphs) + '<w:sectPr/></w:body></w:document>')
    with ZipFile(path, "w", ZIP_DEFLATED) as package:
        package.writestr("word/document.xml", document)
        package.writestr("[Content_Types].xml", '<?xml version="1.0"?>'
                         '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                         '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                         '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                         '</Types>')
        package.writestr("_rels/.rels", '<?xml version="1.0"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                         '</Relationships>')


class PiEntrypointIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report = self.root / "report.docx"
        self.source = self.root / "source.docx"
        write_docx(self.report, ["测试公司2024年度研报", "2024年营业收入200万元。"])
        write_docx(self.source, ["测试公司2024年度合并利润表", "单位：万元",
                                 "项目 2024年度 2023年度", "营业收入 100 90"])
        self.digest = file_hash(self.source)
        self.doc_id = "sha256:" + self.digest
        self.kb = self.root / "kb"
        self.kb.mkdir()
        (self.kb / "ingest_manifest.json").write_text(json.dumps({"documents": [{
            "doc_id": self.doc_id, "sha256": self.digest, "source_path": str(self.source)}]}), encoding="utf-8")
        self.hit = {"doc_id": self.doc_id, "block_id": "paragraph_4", "text": "营业收入 100 90"}
        self.evidence = {"status": "ok", "provider": "controlled_retrieval_fixture", "evidence": {
            **self.hit, "verified": True, "source_sha256": self.digest}}
        self.settings = RuntimeSettings(ledger=self.root / "shared.sqlite3", budget_cny="0")
        self.config = ModelConfig("http://127.0.0.1:1/v1", "unused-local-fixture")

    def runtime(self, status="completed", reason="agent_finished"):
        return {"engine": "pi", "protocol_version": 1, "request_id": "entrypoint-fixture",
                "status": status, "stop_reason": reason, "model_calls": 0, "tool_calls": 4}

    def test_hash_bound_source_resolves_real_missing_source_finding_and_writes_valid_manifest(self):
        views = []

        def runner(task, workflow, config, *, settings, limits):
            self.assertIs(settings, self.settings)
            self.assertIs(config, self.config)
            initial = workflow("detect_document", {})
            views.append(initial)
            self.assertEqual(initial["status"], "needs_review")
            self.assertGreater(initial["summary"]["needs_review"], 0)
            self.assertEqual(initial["summary"]["source_facts"], 0)
            hits = workflow("search_evidence", {"query": "测试公司2024年度营业收入", "limit": 1})
            eid = hits["hits"][0]["evidence_id"]
            read = workflow("read_evidence", {"evidence_id": eid})
            self.assertEqual(read["evidence"]["source_sha256"], self.digest)
            views.append(workflow("recheck", {"evidence_ids": [eid]}))
            return self.runtime()

        with patch("yjparse.openviking.search_kb", return_value={
                "status": "ok", "provider": "controlled_retrieval_fixture", "hits": [self.hit]}), \
                patch("yjparse.openviking.read_evidence", return_value=self.evidence), \
                patch("yjcheck.agent_workflow.check_documents", wraps=check_documents) as checks, \
                patch("urllib.request.urlopen", side_effect=AssertionError("unexpected_model_request")):
            result, directory = run_agent_check(self.report, [], self.root / "out", company="测试公司",
                                                model_config=self.config, runtime_settings=self.settings,
                                                kb_dir=self.kb, runner=runner)
        self.assertIs(checks.call_args_list[0].kwargs["runtime_settings"], self.settings)
        self.assertEqual(checks.call_count, 2)
        self.assertEqual(views[-1]["status"], "completed")
        self.assertEqual(result["summary"]["confirmed_error"], 1)
        self.assertEqual(result["summary"]["needs_review"], 0)
        self.assertTrue(result["summary"]["complete"])
        finding = next(f for f in result["findings"] if f["status"] == "confirmed_error")
        self.assertEqual(finding["claim"]["value"], "200")
        self.assertEqual(finding["suggested_value"], "100")
        source_evidence = finding["evidence"][0]["evidence"][0]
        self.assertEqual(source_evidence["sha256"], self.digest)
        self.assertEqual(Path(source_evidence["file"]), self.source)
        self.assertIsNotNone(source_evidence["paragraph"])
        self.assertIsNone(source_evidence["page"])
        self.assertEqual([e["tool"] for e in result["agent_runtime"]["workflow_events"]],
                         ["detect_document", "search_evidence", "read_evidence", "recheck"])
        self.assertTrue(verify_artifacts(directory))
        saved = json.loads((directory / "check_result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], result["run_id"])
        self.assertFalse(self.settings.ledger.exists())

    def test_stopped_runtime_preserves_real_detection_but_marks_run_incomplete(self):
        def runner(task, workflow, config, *, settings, limits):
            detected = workflow("detect_document", {})
            self.assertEqual(detected["status"], "completed")
            self.assertEqual(detected["summary"]["confirmed_error"], 1)
            return self.runtime("stopped", "max_rounds")

        result, directory = run_agent_check(self.report, [self.source], self.root / "out",
                                            company="测试公司", model_config=self.config,
                                            runtime_settings=self.settings, runner=runner)
        self.assertFalse(result["summary"]["complete"])
        self.assertEqual(result["summary"]["confirmed_error"], 1)
        self.assertTrue(result["agent_runtime"]["incomplete"])
        self.assertEqual(result["agent_runtime"]["stop_reason"], "max_rounds")
        self.assertTrue(verify_artifacts(directory))


if __name__ == "__main__":
    unittest.main()
