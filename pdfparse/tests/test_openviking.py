"""Real localhost HTTP contract tests; no SDK, external model, or paid requests."""

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yjparse.openviking import (OpenVikingClient, OpenVikingConfig, OpenVikingError,
                               config_for_kb, health, ingest_kb, read_evidence, search_kb)


class TestTransportConfig(unittest.TestCase):
    """Pure configuration checks: no server, DNS resolution, or HTTP request."""

    def test_http_is_allowed_only_for_explicit_loopback_hosts(self):
        for endpoint in ["http://localhost:1933", "http://127.0.0.1:1933",
                         "http://[::1]:1933", "http://LOCALHOST:1933/v1"]:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(OpenVikingConfig(base_url=endpoint).base_url, endpoint)

    def test_remote_http_rejected_with_or_without_api_key(self):
        for endpoint in ["http://ov.example:1933", "http://192.168.1.20:1933",
                         "http://0.0.0.0:1933", "http://[2001:db8::1]:1933"]:
            for key in ["", "fixture-tenant-key"]:
                with self.subTest(endpoint=endpoint, key_present=bool(key)):
                    with self.assertRaisesRegex(ValueError, "HTTPS outside loopback"):
                        OpenVikingConfig(base_url=endpoint, api_key=key)

    def test_loopback_lookalikes_cannot_bypass_https(self):
        for endpoint in ["http://localhost.example", "http://127.0.0.1.example",
                         "http://127.0.0.1%2eexample", "http://2130706433",
                         "http://[::ffff:127.0.0.1]"]:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "HTTPS outside loopback"):
                    OpenVikingConfig(base_url=endpoint, api_key="fixture-tenant-key")

    def test_remote_https_retains_explicit_path_and_key(self):
        config = OpenVikingConfig(base_url="https://ov.example:8443/api",
                                 api_key="fixture-tenant-key")
        self.assertEqual(config.base_url, "https://ov.example:8443/api")
        self.assertEqual(config.api_key, "fixture-tenant-key")
        self.assertNotIn("fixture-tenant-key", repr(config))

    def test_environment_uses_same_guard_without_opening_transport(self):
        with patch.dict(os.environ, {"OPENVIKING_URL": "http://ov.example",
                                    "OPENVIKING_API_KEY": "fixture-tenant-key"}, clear=True):
            with patch("urllib.request.OpenerDirector.open") as request:
                with self.assertRaisesRegex(ValueError, "HTTPS outside loopback"):
                    OpenVikingConfig.from_env()
                request.assert_not_called()


class TestConfigForKB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.kb = Path(self.tmp.name)
        self.manifest = self.kb / "ingest_manifest.json"
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def write_manifest(self, payload):
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")

    def test_project_changes_only_scope_not_server_settings(self):
        self.write_manifest({"project": "pi-local-smoke", "base_url": "https://ignored.invalid",
                             "api_key": "ignored", "timeout": 100,
                             "openviking": [{"uri": "viking://resources/ignored"}]})
        with patch.dict(os.environ, {"OPENVIKING_URL": "http://127.0.0.1:12345", "OPENVIKING_TIMEOUT": "2.5"}):
            config = config_for_kb(self.kb)
        self.assertEqual(config.target_uri, "viking://resources/pi-local-smoke")
        self.assertEqual(config.base_url, "http://127.0.0.1:12345")
        self.assertEqual(config.timeout, 2.5)
        self.assertEqual(config.api_key, "")

    def test_explicit_config_precedes_environment_and_manifest(self):
        self.write_manifest({"project": "../invalid"})
        explicit = OpenVikingConfig(target_uri="viking://resources/explicit")
        with patch.dict(os.environ, {"OPENVIKING_TARGET_URI": "viking://resources/environment"}):
            self.assertIs(config_for_kb(self.kb, explicit), explicit)

    def test_explicit_environment_scope_precedes_manifest(self):
        self.write_manifest({"project": "../invalid"})
        with patch.dict(os.environ, {"OPENVIKING_TARGET_URI": "viking://resources/environment"}):
            self.assertEqual(config_for_kb(self.kb).target_uri, "viking://resources/environment")

    def test_missing_manifest_or_project_preserves_default(self):
        self.assertEqual(config_for_kb(self.kb).target_uri, OpenVikingConfig.target_uri)
        self.write_manifest({"documents": []})
        self.assertEqual(config_for_kb(self.kb).target_uri, OpenVikingConfig.target_uri)

    def test_invalid_project_fails_instead_of_changing_scope(self):
        for project in ["", None, 1, {}, ".", "..", "a/b", "a\\b", "%2e%2e", "https://elsewhere", "a?b", "a#b", "a b", "a\nb"]:
            with self.subTest(project=project):
                self.write_manifest({"project": project})
                with self.assertRaisesRegex(ValueError, "ingest_manifest.project"):
                    config_for_kb(self.kb)


class ContractHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, result, status=200, envelope=True):
        data = {"status": "ok", "result": result} if envelope else result
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        state = self.server.state
        state["requests"].append(("GET", self.path, self.headers.get("X-API-Key")))
        path = urlsplit(self.path).path
        if path == "/health":
            time.sleep(state.get("delay", 0))
            if state.get("invalid_json"):
                self.send_response(200)
                self.end_headers()
                try:
                    self.wfile.write(b"not-json")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return
            self.respond({"status": "ok", "healthy": True, "version": "0.4.23"}, envelope=False)
        elif path.startswith("/api/v1/tasks/"):
            self.respond({"task_id": "task-1", "status": state.get("task_status", "completed")})
        elif path == "/api/v1/content/read":
            uri = parse_qs(urlsplit(self.path).query)["uri"][0]
            if state.get("read_error"):
                self.respond({"message": "do not expose secret-token"}, status=503)
            else:
                self.respond(state.get("content_override", state["content"].get(uri, "")))
        else:
            self.respond({}, status=404)

    def do_POST(self):
        state = self.server.state
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        state["requests"].append(("POST", self.path, self.headers.get("X-API-Key")))
        if self.path == "/api/v1/resources/temp_upload":
            state["upload_content_type"] = self.headers.get("Content-Type")
            state["uploaded"] = body.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
            self.respond({"temp_file_id": "temp-1"})
            return
        payload = json.loads(body)
        state["payloads"].append((self.path, payload))
        if self.path == "/api/v1/resources":
            state["content"][payload["to"] + "/document.md"] = state["uploaded"].decode("utf-8")
            result = {"root_uri": payload["to"], "task_id": "task-1"}
            if state.get("missing_task"):
                del result["task_id"]
            self.respond(result)
        elif self.path == "/api/v1/search/find":
            if state.get("search_error"):
                self.respond({"error": "secret-token"}, status=503)
                return
            resources = state.get("candidates", [{
                "uri": "viking://resources/research-reports/doc-1/document.md",
                "score": 0.95, "abstract": "FAKE SUMMARY: revenue is 999 billion",
                "content": "<!-- block:invented page:1 --> also untrusted"}])
            self.respond({"resources": resources, "memories": [], "skills": []})
        else:
            self.respond({}, status=404)


class TestOpenVikingHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ContractHandler)
        cls.thread = threading.Thread(target=lambda: cls.server.serve_forever(poll_interval=0.02), daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.kb = Path(self.tmp.name)
        (self.kb / "markdown").mkdir()
        self.text = "营业收入为100亿元，证据原文。" * 35
        rows = [{"doc_id": "doc-1", "block_id": "p1-b1", "page": 1,
                 "page_status": "pass", "bbox": [1, 2, 90, 30], "text": self.text,
                 "cells": [], "type": "text", "run_id": "run-1", "source_path": "report.pdf"},
                {"doc_id": "doc-1", "block_id": "p1-b2", "page": 1,
                 "page_status": "pass", "bbox": [1, 40, 90, 90], "text": "利润表",
                 "cells": [{"row": 0, "col": 0, "text": "利润为20亿元"}], "type": "table"},
                {"doc_id": "doc-1", "block_id": "p2-b1", "page": 2,
                 "page_status": "fail", "text": "营业收入错误页", "type": "text"}]
        self.index = self.kb / "kb_index.jsonl"
        self.index.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        self.markdown = ("<!-- source:report.pdf sha256:abc engine:test@1 run:run-1 -->\n"
                         "<!-- Page 1 -->\n<!-- block:p1-b1 page:1 type:text bbox:[1,2,90,30] -->\n"
                         + self.text + "\n<!-- block:p1-b2 page:1 type:table bbox:[1,40,90,90] -->\n利润表\n")
        self.md = self.kb / "markdown/doc-1.md"
        self.md.write_text(self.markdown, encoding="utf-8")
        (self.kb / "ingest_manifest.json").write_text(json.dumps({"documents": [{
            "doc_id": "doc-1", "sha256": "abc", "markdown": str(self.md),
            "markdown_sha256": hashlib.sha256(self.md.read_bytes()).hexdigest()}]}), encoding="utf-8")
        self.server.state = {"requests": [], "payloads": [], "content": {}}
        self.config = OpenVikingConfig(base_url=f"http://127.0.0.1:{self.server.server_port}",
                                       api_key="secret-token", timeout=0.5)
        self.uri = "viking://resources/research-reports/doc-1/document.md"

    def ingest(self, **kwargs):
        result = ingest_kb(self.kb, self.config, **kwargs)
        self.assertEqual(result["status"], "ok", result)
        return result

    def test_health_uses_real_http_and_does_not_claim_model_readiness(self):
        result = health(self.config)
        self.assertTrue(result["available"])
        self.assertEqual(result["server_version"], "0.4.23")
        self.assertEqual(result["model_readiness"], "not_checked")
        self.assertEqual(self.server.state["requests"][0][2], "secret-token")
        self.assertNotIn("secret-token", repr(self.config))

    def test_implicit_smoke_project_scope_is_shared_by_ingest_search_and_read(self):
        manifest_path = self.kb / "ingest_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["project"] = "pi-local-smoke"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        uri = "viking://resources/pi-local-smoke/doc-1/document.md"
        self.server.state["candidates"] = [{"uri": uri, "score": 0.9}]
        with patch.dict(os.environ, {"OPENVIKING_URL": self.config.base_url}, clear=True):
            ingested = ingest_kb(self.kb)
            found = search_kb(self.kb, "营业收入")
            evidence = read_evidence(self.kb, "doc-1", "p1-b1", uri=uri)
        self.assertEqual(ingested["status"], "ok", ingested)
        self.assertEqual(found["status"], "ok", found)
        self.assertEqual(found["hits"][0]["uri"], uri)
        self.assertEqual(evidence["status"], "ok", evidence)
        self.assertTrue(evidence["openviking_used"])
        search_payload = next(value for path, value in self.server.state["payloads"] if path.endswith("/find"))
        self.assertEqual(search_payload["target_uri"], "viking://resources/pi-local-smoke")

    def test_multipart_ingest_preserves_bytes_anchors_and_polls_task(self):
        result = self.ingest()
        self.assertEqual(self.server.state["uploaded"], self.md.read_bytes())
        self.assertIn("multipart/form-data", self.server.state["upload_content_type"])
        payload = self.server.state["payloads"][0][1]
        self.assertEqual(payload["temp_file_id"], "temp-1")
        self.assertEqual(payload["args"], {"parse_mode": "no_split"})
        self.assertEqual(payload["processing_mode"], "vectors_only")
        self.assertNotIn("path", payload)
        self.assertEqual(result["documents"][0]["status"], "completed")
        self.assertTrue(any("/tasks/task-1" in p for _, p, _ in self.server.state["requests"]))

    def test_search_reads_l2_and_returns_full_original_text_not_summary(self):
        self.ingest()
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(result["status"], "ok")
        hit = result["hits"][0]
        self.assertEqual(hit["text"], self.text)
        self.assertGreater(len(hit["text"]), 300)
        self.assertEqual(hit["block_id"], "p1-b1")
        self.assertEqual(hit["bbox"], [1, 2, 90, 30])
        self.assertEqual(hit["evidence_source"], "kb_index")
        self.assertTrue(hit["verified"])
        self.assertFalse(hit["source_file_verified"])
        self.assertEqual(hit["source_sha256"], "abc")
        self.assertNotIn("999 billion", json.dumps(result))
        request = [p for path, p in self.server.state["payloads"] if path.endswith("/find")][0]
        self.assertEqual(request["level"], 2)
        self.assertEqual(request["context_type"], "resource")
        self.assertTrue(any("raw=true" in p for _, p, _ in self.server.state["requests"]))

    def test_unknown_block_remains_unconfirmed(self):
        self.ingest()
        self.server.state["content_override"] = "<!-- block:unknown page:1 --> fabricated"
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["status"], "unconfirmed")
        self.assertEqual(result["unconfirmed"][0]["reason"], "UNKNOWN_BLOCK")

    def test_failed_page_and_wrong_page_do_not_become_evidence(self):
        self.ingest()
        self.server.state["content_override"] = "<!-- block:p2-b1 page:2 --> x\n<!-- block:p1-b1 page:9 --> x"
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertFalse(result["hits"])
        self.assertTrue(all(x["reason"] == "INVALID_PAGE_ANCHOR" for x in result["unconfirmed"]))

    def test_summary_without_anchors_is_unconfirmed_even_when_fallback_allowed(self):
        self.ingest()
        self.server.state["content_override"] = "Revenue summary: 100 billion"
        result = search_kb(self.kb, "营业收入", config=self.config, allow_fallback=True)
        self.assertEqual(result["status"], "unconfirmed")
        self.assertEqual(result["provider"], "openviking")
        self.assertFalse(result["hits"])

    def test_resource_outside_bound_document_is_not_read(self):
        self.ingest()
        self.server.state["candidates"] = [{"uri": "viking://resources/other/doc/document.md"}]
        before = len(self.server.state["requests"])
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertFalse(result["hits"])
        self.assertEqual(result["unconfirmed"][0]["reason"], "UNBOUND_RESOURCE")
        self.assertFalse(any("content/read" in p for _, p, _ in self.server.state["requests"][before:]))

    def test_local_index_change_invalidates_stale_remote_binding(self):
        self.ingest()
        self.index.write_text(self.index.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(result["unconfirmed"][0]["reason"], "STALE_LOCAL_INDEX")
        self.assertFalse(result["hits"])

    def test_server_identity_is_part_of_binding(self):
        self.ingest()
        p = self.kb / "openviking_bindings.json"
        bindings = json.loads(p.read_text(encoding="utf-8"))
        bindings["documents"][0]["server"] = "http://another-server:1933"
        p.write_text(json.dumps(bindings), encoding="utf-8")
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(result["unconfirmed"][0]["reason"], "SERVER_MISMATCH")

    def test_service_failure_does_not_silently_use_bm25(self):
        self.server.state["search_error"] = True
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["hits"], [])
        self.assertFalse(result["openviking_used"])
        self.assertNotIn("secret-token", json.dumps(result))

    def test_explicit_bm25_fallback_has_provider_and_original_evidence(self):
        self.server.state["search_error"] = True
        result = search_kb(self.kb, "营业收入", config=self.config, allow_fallback=True)
        self.assertEqual(result["provider"], "bm25")
        self.assertEqual(result["status"], "fallback")
        self.assertFalse(result["openviking_used"])
        self.assertEqual(result["hits"][0]["text"], self.text)
        self.assertEqual(result["errors"][0]["code"], "HTTP_503")

    def test_empty_doc_filter_does_not_search_all_documents_in_fallback(self):
        self.server.state["search_error"] = True
        result = search_kb(self.kb, "营业收入", doc_ids=[], config=self.config, allow_fallback=True)
        self.assertEqual(result["hits"], [])

    def test_read_evidence_distinguishes_local_and_remote_sources(self):
        local = read_evidence(self.kb, "doc-1", "p1-b1")
        self.assertEqual(local["provider"], "kb_index")
        self.assertFalse(local["openviking_used"])
        self.ingest()
        remote = read_evidence(self.kb, "doc-1", "p1-b1", self.config, self.uri)
        self.assertTrue(remote["openviking_used"])
        self.assertEqual(remote["evidence"]["text"], self.text)
        self.assertIsNone(read_evidence(self.kb, "doc-1", "unknown")["evidence"])

    def test_read_failure_is_explicit_not_a_confirmed_summary(self):
        self.ingest()
        self.server.state["read_error"] = True
        result = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(result["status"], "unconfirmed")
        self.assertFalse(result["hits"])
        self.assertEqual(result["errors"][0]["code"], "HTTP_503")

    def test_explicit_read_fallback_preserves_local_provider_and_failure(self):
        self.ingest()
        self.server.state["read_error"] = True
        result = read_evidence(self.kb, "doc-1", "p1-b1", self.config, self.uri)
        self.assertEqual(result["status"], "unavailable")
        result = read_evidence(self.kb, "doc-1", "p1-b1", self.config, self.uri, allow_fallback=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "kb_index")
        self.assertFalse(result["openviking_used"])
        self.assertTrue(result["evidence"]["verified"])
        self.assertEqual(result["evidence"]["verification_scope"], "local_index")
        self.assertEqual(result["errors"][0]["code"], "HTTP_503")
        self.assertEqual(result["evidence"]["text"], self.text)

    def test_evidence_identity_is_stable_across_remote_and_local_reads(self):
        local = read_evidence(self.kb, "doc-1", "p1-b1")
        self.ingest()
        remote = read_evidence(self.kb, "doc-1", "p1-b1", self.config, self.uri)
        search = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(local["evidence"]["evidence_id"], remote["evidence"]["evidence_id"])
        self.assertEqual(search["hits"][0]["evidence_id"], remote["evidence"]["evidence_id"])

    def test_cancellation_after_find_prevents_content_requests_and_fallback(self):
        self.ingest()
        before = len(self.server.state["requests"])
        checks = []

        def cancel_check():
            checks.append(True)
            if len(checks) == 2:
                raise RuntimeError("execution_cancelled")

        with self.assertRaisesRegex(RuntimeError, "execution_cancelled"):
            search_kb(self.kb, "营业收入", config=self.config, allow_fallback=True,
                      cancel_check=cancel_check)
        subsequent = self.server.state["requests"][before:]
        self.assertEqual(len(subsequent), 1)
        self.assertEqual(subsequent[0][1], "/api/v1/search/find")

    def test_cancelled_ingest_does_not_send_upload(self):
        def cancel_check():
            raise RuntimeError("execution_cancelled")

        with self.assertRaisesRegex(RuntimeError, "execution_cancelled"):
            ingest_kb(self.kb, config=self.config, cancel_check=cancel_check)
        self.assertEqual(self.server.state["requests"], [])

    def test_submitted_ingest_is_not_claimed_completed_or_searchable(self):
        result = ingest_kb(self.kb, self.config, wait=False)
        self.assertEqual(result["status"], "submitted")
        found = search_kb(self.kb, "营业收入", config=self.config)
        self.assertEqual(found["unconfirmed"][0]["reason"], "INGEST_NOT_COMPLETED")

    def test_processing_failure_does_not_write_completed_binding(self):
        self.server.state["task_status"] = "failed"
        result = ingest_kb(self.kb, self.config)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"][0]["code"], "PROCESSING_FAILED")
        self.assertEqual(json.loads((self.kb / "openviking_bindings.json").read_text())["documents"], [])

    def test_processing_timeout_is_bounded_and_explicit(self):
        self.server.state["task_status"] = "running"
        started = time.monotonic()
        result = ingest_kb(self.kb, self.config, processing_timeout=0.06)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(result["errors"][0]["code"], "PROCESSING_TIMEOUT")

    def test_upload_success_without_task_id_is_not_completed(self):
        self.server.state["missing_task"] = True
        result = ingest_kb(self.kb, self.config)
        self.assertEqual(result["errors"][0]["code"], "MISSING_TASK_ID")

    def test_local_export_integrity_error_prevents_network_upload(self):
        self.md.write_text(self.markdown + "tampered", encoding="utf-8")
        result = ingest_kb(self.kb, self.config)
        self.assertEqual(result["errors"][0]["code"], "EXPORT_INDEX_MISMATCH")
        self.assertEqual(self.server.state["requests"], [])

    def test_transport_timeout_and_bad_json_are_visible(self):
        self.server.state["delay"] = 0.12
        config = OpenVikingConfig(base_url=self.config.base_url, timeout=0.02)
        result = health(config)
        self.assertEqual(result["errors"][0]["code"], "TIMEOUT")
        self.server.state["delay"] = 0
        self.server.state["invalid_json"] = True
        self.assertEqual(health(self.config)["errors"][0]["code"], "INVALID_JSON")

    def test_configuration_rejects_credentials_in_url_and_unsafe_uri(self):
        with self.assertRaises(ValueError):
            OpenVikingConfig(base_url="http://name:secret@localhost")
        with self.assertRaises(ValueError):
            OpenVikingConfig(target_uri="viking://resources/project/%2e%2e/other")
        with self.assertRaises(ValueError):
            OpenVikingConfig(timeout=0)


if __name__ == "__main__":
    unittest.main()
