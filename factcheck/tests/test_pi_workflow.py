"""Business boundaries are enforced independently of the agent's plan."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yjcheck.agent_workflow import EvidenceWorkflow, _retain_previous_candidates
from yjcheck.span_normalization import normalize_candidate_spans
from yjcheck.text_review import detect_text


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.kb = Path(self.temp.name)
        self.source = self.kb / "source.pdf"
        self.source.write_bytes(b"frozen-source")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.manifest = {"documents": [{"doc_id": "source", "sha256": self.digest,
                                         "source_path": str(self.source)}]}
        self.write_manifest()
        self.report = {"document_id": "doc", "errors": [{"id": "err", "status": "needs_review",
                        "error_type": "数值单位错误", "spans": [], "reason": "verify source"}],
                       "coverage": {"complete": True}}
        self.seen = []
        self.workflow = EvidenceWorkflow(lambda: deepcopy(self.report),
                                         self.recheck, kb_dir=self.kb)
        self.hit = {"doc_id": "source", "block_id": "b1", "text": "单位：万元", "page": 1}
        self.evidence = {"status": "ok", "provider": "openviking", "evidence": {
            **self.hit, "verified": True, "source_sha256": self.digest}}

    def write_manifest(self):
        (self.kb / "ingest_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def recheck(self, paths, prior):
        self.seen.extend(paths)
        return prior

    def search(self):
        self.workflow("detect_document", {})
        with patch("yjparse.openviking.search_kb", return_value={"status": "ok", "hits": [self.hit]}):
            return self.workflow("search_evidence", {"query": "利润单位"})["hits"][0]["evidence_id"]

    def read(self, eid, evidence=None):
        with patch("yjparse.openviking.read_evidence", return_value=evidence or self.evidence):
            return self.workflow("read_evidence", {"evidence_id": eid})

    def test_agent_cannot_skip_detection_or_supply_unsearched_evidence(self):
        with self.assertRaises(ValueError):
            self.workflow("search_evidence", {"query": "x"})
        self.workflow("detect_document", {})
        with self.assertRaises(ValueError):
            self.workflow("read_evidence", {"evidence_id": "invented"})

    def test_recheck_requires_read_and_retains_python_verdict(self):
        eid = self.search()
        with self.assertRaises(ValueError):
            self.workflow("recheck", {"evidence_ids": [eid]})
        self.read(eid)
        result = self.workflow("recheck", {"evidence_ids": [eid]})
        self.assertEqual(self.seen, [str(self.source.resolve())])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(self.workflow.result["errors"], self.report["errors"])

    def test_transport_ok_is_not_verified_evidence(self):
        eid = self.search()
        self.read(eid, {"status": "ok", "text": "everything confirmed"})
        with self.assertRaisesRegex(ValueError, "identity"):
            self.workflow("recheck", {"evidence_ids": [eid]})
        self.assertFalse(self.seen)

    def test_changed_source_file_is_blocked(self):
        eid = self.search()
        self.read(eid)
        self.source.write_bytes(b"changed after indexing")
        with self.assertRaisesRegex(ValueError, "source_changed"):
            self.workflow("recheck", {"evidence_ids": [eid]})

    def test_wrong_block_identity_and_duplicate_manifest_are_blocked(self):
        eid = self.search()
        changed = deepcopy(self.evidence)
        changed["evidence"]["block_id"] = "invented"
        self.read(eid, changed)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.workflow("recheck", {"evidence_ids": [eid]})
        self.read(eid)
        self.manifest["documents"] *= 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.workflow("recheck", {"evidence_ids": [eid]})

    def test_empty_incomplete_detection_is_not_success(self):
        self.workflow = EvidenceWorkflow(lambda: {"errors": [], "coverage": {"complete": False}}, self.recheck)
        self.assertEqual(self.workflow("detect_document", {})["status"], "needs_review")

    def test_text_recheck_reports_external_findings_to_pi_separately(self):
        external = {"summary": {"confirmed_error": 1}, "findings": [
            {"id": "external", "status": "confirmed_error", "rule_id": "number", "message": "原文数值不一致"}]}
        self.report["external_fact_check"] = external
        view = self.workflow("detect_document", {})
        self.assertEqual(view["external_fact_check"]["summary"]["confirmed_error"], 1)
        self.assertTrue(view["external_fact_check"]["scored_separately"])
        self.assertEqual(view["pending_checks_total"], 1)
        self.assertEqual(self.workflow.result["errors"][0]["status"], "needs_review")

    def test_recheck_keeps_model_claim_denominator_and_failure_visible(self):
        current = {"findings": [{"id": "rule", "claim": {"metric": "income"}, "status": "no_issue"}],
                   "summary": {"complete": True, "claims": 1, "automatic_claims": 1, "review_claims": 0}}
        previous = {"findings": [{"id": "model", "claim": {"metric": "profit"}, "status": "needs_review",
                                 "rule_id": "MODEL_CANDIDATE", "decision": "ask", "evidence_request": [{}]}],
                    "summary": {"model_candidates_aligned": 2}, "model_traces": [{"status": "error"}]}
        result = _retain_previous_candidates(current, previous)
        self.assertEqual(result["summary"]["claims"], 2)
        self.assertEqual(result["summary"]["automatic_claims"], 1)
        self.assertEqual(result["summary"]["review_claims"], 1)
        self.assertEqual(result["summary"]["needs_review_by_rule"], {"MODEL_CANDIDATE": 1})
        self.assertEqual(result["summary"]["model_candidates_aligned"], 2)
        self.assertEqual(result["summary"]["evidence_request_items"], 1)
        self.assertFalse(result["summary"]["complete"])
        self.assertEqual(result["model_traces"], previous["model_traces"])


class SpanOutputTests(unittest.TestCase):
    def test_actual_detection_merges_quotes_without_promoting_status(self):
        content = "重复原文。重复原文。"
        candidate = {"error_type": "冗余语句", "reason": "重复", "spans": [
            {"start": 0, "end": 5, "text": content[:5]}, {"start": 5, "end": 10, "text": content[5:]}]}
        chat = lambda messages, purpose: {"content": json.dumps({"errors": [candidate]}, ensure_ascii=False)}
        original = detect_text(content, document_id="doc", detector="model_direct", chat=chat, normalize_spans=False)
        changed = detect_text(content, document_id="doc", detector="model_direct", chat=chat)
        self.assertEqual(len(original["errors"][0]["spans"]), 2)
        self.assertEqual(changed["errors"][0]["spans"], [{"start": 0, "end": 10, "text": content}])
        self.assertEqual(changed["errors"][0]["status"], "needs_review")
        self.assertEqual(original["errors"][0]["id"], changed["errors"][0]["id"])

    def test_missing_words_and_invalid_quote_are_never_filled(self):
        for content, spans in [("A.B", [{"start": 0, "end": 1, "text": "A"}, {"start": 2, "end": 3, "text": "B"}]),
                               ("AB", [{"start": 0, "end": 1, "text": "wrong"}, {"start": 1, "end": 2, "text": "B"}])]:
            candidate = {"spans": spans, "status": "needs_review"}
            self.assertEqual(normalize_candidate_spans(candidate, content), candidate)


if __name__ == "__main__":
    unittest.main()
