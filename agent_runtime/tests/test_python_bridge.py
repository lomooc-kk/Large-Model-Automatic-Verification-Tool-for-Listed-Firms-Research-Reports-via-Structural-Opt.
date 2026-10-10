"""Actual Python -> Node Pi SDK -> local SSE -> Python workflow integration.

Only retrieval responses and business detection/recheck are controlled fixtures.
The real Pi SDK, JSONL bridge, EvidenceWorkflow state machine, source hashes,
and SQLite shared budget reservations/settlements all execute.
No remote models, API charges, or existing budget database are touched.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "factcheck/src"), str(ROOT / "pdfparse/src")]

from yjcheck.agent_workflow import EvidenceWorkflow
from yjcheck.model import ModelConfig
from yjcheck.model_runtime import RuntimeSettings
from yjcheck.pi_bridge import AgentLimits, run_pi, runtime_command


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.request_bodies.append(body)
        index = len(self.server.request_bodies) - 1
        with closing(sqlite3.connect(self.server.ledger_path)) as db:
            states = [row[0] for row in db.execute("SELECT status FROM calls")]
        # Prove the Python shared ledger committed this reservation before HTTP.
        self.server.reservation_checks.append(states.count("reserved") == 1 and len(states) == index + 1)
        steps = [
            ("detect_document", {}),
            ("search_evidence", {"query": "收入 2025", "limit": 3}),
            ("read_evidence", {"evidence_id": self.server.evidence_id}),
            ("recheck", {"evidence_ids": [self.server.evidence_id]}),
            (None, None),
        ]
        if index >= len(steps):
            self.send_error(500)
            return
        name, args = steps[index]
        base = {"id": f"fixture-{index}", "object": "chat.completion.chunk", "created": 1700000000, "model": "fixture-model"}
        packets = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
        delta = {"tool_calls": [{"index": 0, "id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]} if name else {"content": "工具已完成核查。"}
        packets.extend([
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if name else "stop"}]},
            {**base, "choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}},
        ])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for packet in packets:
            self.wfile.write(("data: " + json.dumps(packet, ensure_ascii=False) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class RealPiPythonBridgeTests(unittest.TestCase):
    def setUp(self):
        runtime_command()  # Fail clearly if actual runtime has not been built.
        self.temp = tempfile.TemporaryDirectory(prefix="yjcheck-pi-integration-")
        self.directory = Path(self.temp.name)
        self.ledger = self.directory / "usage.sqlite3"
        self.eid = sha256(json.dumps(["source-1", "block-1"]).encode()).hexdigest()[:24]
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        self.server.ledger_path = self.ledger
        self.server.evidence_id = self.eid
        self.server.request_bodies = []
        self.server.reservation_checks = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.model = ModelConfig(f"http://127.0.0.1:{self.server.server_port}/v1", "fixture-model", "fixture-secret-never-in-traces")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def settings(self, budget="1"):
        return RuntimeSettings(input_cny_per_mtok="1", output_cny_per_mtok="4",
                               price_source="local-test-fixture", price_date="2026-10-09",
                               context_tokens=20000, max_output_tokens=1000,
                               budget_cny=budget, ledger=self.ledger, max_retries=0)

    def test_actual_sdk_bridge_workflow_and_shared_ledger(self):
        source = self.directory / "source.txt"
        source.write_text("公司2025年收入120万元。", encoding="utf-8")
        digest = sha256(source.read_bytes()).hexdigest()
        (self.directory / "ingest_manifest.json").write_text(json.dumps({"documents": [
            {"doc_id": "source-1", "source_path": str(source), "sha256": digest}
        ]}), encoding="utf-8")
        pending = {"run_id": "business-result", "summary": {"complete": False}, "findings": [
            {"id": "claim-1", "status": "needs_review", "message": "缺少原始来源", "claim": {"text": "收入100万元"}}
        ]}
        verified = deepcopy(pending)
        verified["summary"]["complete"] = True
        verified["findings"][0]["status"] = "confirmed_error"
        seen_sources = []

        def recheck(paths, previous):
            seen_sources.extend(paths)
            self.assertEqual(previous["findings"][0]["status"], "needs_review")
            return deepcopy(verified)

        workflow = EvidenceWorkflow(lambda: deepcopy(pending), recheck, kb_dir=self.directory)
        hit = {"doc_id": "source-1", "block_id": "block-1", "page": 1, "text": source.read_text(encoding="utf-8")}
        read = {"status": "ok", "evidence": {**hit, "verified": True, "source_sha256": digest}}
        with patch("yjparse.openviking.search_kb", return_value={"status": "ok", "provider": "controlled-test-fixture", "hits": [hit]}), patch("yjparse.openviking.read_evidence", return_value=read):
            result = run_pi({"documentId": "fixed-doc", "question": "检测并按需补证"}, workflow,
                            self.model, settings=self.settings(),
                            limits=AgentLimits(max_rounds=6, max_tool_calls=8, timeout_seconds=15,
                                               tool_timeout_seconds=3, max_total_tokens=100000, max_cost_cny="1"))
        self.assertEqual(result["engine"], "pi")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual([e["tool"] for e in workflow.events], ["detect_document", "search_evidence", "read_evidence", "recheck"])
        self.assertEqual(workflow.result["findings"][0]["status"], "confirmed_error")
        self.assertEqual(seen_sources, [str(source.resolve())])
        self.assertEqual(self.server.reservation_checks, [True] * 5)
        self.assertEqual(result["budget"]["calls"], 5)
        self.assertEqual(result["budget"]["uncertain_calls"], 0)
        self.assertAlmostEqual(result["budget"]["accounted_cny"], 0.0014)
        with closing(sqlite3.connect(self.ledger)) as db:
            rows = db.execute("SELECT status, charge_micro, trace FROM calls").fetchall()
        self.assertEqual([r[0] for r in rows], ["ok"] * 5)
        self.assertEqual([r[1] for r in rows], [280] * 5)
        self.assertNotIn(self.model.api_key, json.dumps([result, rows, self.server.request_bodies]))

    def test_empty_shared_budget_blocks_sdk_before_http(self):
        tools = []
        result = run_pi({"documentId": "fixed-doc", "question": "检测"}, lambda name, args: tools.append(name),
                        self.model, settings=self.settings("0"),
                        limits=AgentLimits(timeout_seconds=10, tool_timeout_seconds=2, max_total_tokens=100000))
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["budget"]["calls"], 0)
        self.assertEqual(self.server.request_bodies, [])
        self.assertEqual(tools, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
