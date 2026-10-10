"""Boundary checks plus real, offline Chinese embedding/HTTP regression tests."""

import base64
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy/local_embedding_server.py"
SPEC = importlib.util.spec_from_file_location("local_embedding_server", SCRIPT)
embedding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(embedding)


class TestEmbeddingBoundaries(unittest.TestCase):
    def test_rejects_invalid_inputs_and_dimensions(self):
        invalid = [None, [], {"input": ""}, {"input": [1]}, {"input": ["ok", None]},
                   {"input": "ok", "dimensions": True}, {"input": "ok", "dimensions": 384},
                   {"input": "ok", "encoding_format": "unknown"}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(embedding.RequestError):
                embedding.validate_request(payload)

    def test_limits_batch_and_input_characters(self):
        for value in [["a"] * 17, "a" * (embedding.MAX_INPUT_CHARACTERS + 1)]:
            with self.subTest(length=len(value)), self.assertRaises(embedding.RequestError):
                embedding.validate_request({"input": value})

    def test_rejects_non_loopback_and_excess_threads(self):
        with self.assertRaises(ValueError):
            embedding.make_server(None, "0.0.0.0", 0)
        with self.assertRaises(ValueError):
            embedding.EmbeddingEngine(Path("unused"), threads=5)

    def test_corrupt_model_cannot_load(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "onnx/model_quantized.onnx"
            model.parent.mkdir()
            model.write_bytes(b"invalid weights")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                embedding.verify_model_files(Path(folder))


class TestRealLocalEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all((embedding.DEFAULT_MODEL_DIR / name).is_file() for name in embedding.MODEL_FILES):
            raise unittest.SkipTest("Pinned offline model is absent; follow README-local-embedding.md")
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"Local embedding runtime dependency is absent: {exc.name}") from exc
        cls.engine = embedding.EmbeddingEngine(embedding.DEFAULT_MODEL_DIR, threads=2)
        cls.server = embedding.make_server(cls.engine, port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, payload=None, method="POST", path="/v1/embeddings", raw=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=10)
        if method.upper() == "GET":
            # Health GET has no entity. Sending JSON null left unread bytes when
            # do_GET immediately replied and closed, which can reset on Windows.
            body = None
            headers = {key: value for key, value in (headers or {}).items()
                       if key.lower() not in {"content-type", "content-length"}}
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if raw is None else raw
            headers = {"Content-Type": "application/json"} if headers is None else headers
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_actual_chinese_retrieval_ranks_relevant_passage_first(self):
        status, result = self.request({"model": embedding.MODEL_ID, "input": [
            "公司营业收入同比增长百分之十。",
            "本年度公司的收入比上年增加百分之十。",
            "今天公园里有很多游客在散步。",
        ]})
        self.assertEqual(status, 200)
        np = self.engine.np
        vectors = np.asarray([item["embedding"] for item in result["data"]])
        self.assertEqual(vectors.shape, (3, 512))
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        scores = vectors[1:] @ vectors[0]
        self.assertGreater(scores[0], scores[1] + 0.1)
        self.assertEqual(result["truncation"]["truncated_inputs"], 0)

    def test_base64_is_little_endian_float32_with_same_embedding(self):
        plain = self.engine.embed(["中文财务证据"])["data"][0]["embedding"]
        status, result = self.request({"input": "中文财务证据", "encoding_format": "base64"})
        self.assertEqual(status, 200)
        blob = base64.b64decode(result["data"][0]["embedding"], validate=True)
        self.assertEqual(len(blob), 512 * 4)
        self.engine.np.testing.assert_allclose(self.engine.np.frombuffer(blob, dtype="<f4"), plain, atol=1e-7)

    def test_truncation_counts_exclude_padding_and_reset_between_calls(self):
        status, before = self.request(method="GET", path="/health")
        self.assertEqual(status, 200)
        result = self.engine.embed(["财务收入增长。" * 150, "短句"])
        long, short = result["data"]
        self.assertTrue(long["truncated"])
        self.assertEqual(long["processed_tokens"], 512)
        self.assertEqual(long["input_tokens"] - 512, long["truncated_tokens"])
        self.assertFalse(short["truncated"])
        self.assertLess(short["processed_tokens"], 512)
        self.assertEqual(result["usage"]["total_tokens"], 512 + short["processed_tokens"])
        status, after = self.request(method="GET", path="/health")
        self.assertEqual(status, 200)
        self.assertEqual(after["completed_requests"] - before["completed_requests"], 1)
        self.assertEqual(after["total_input_tokens"] - before["total_input_tokens"], result["truncation"]["input_tokens"])
        self.assertEqual(after["processed_tokens"] - before["processed_tokens"], result["usage"]["total_tokens"])
        self.assertEqual(after["truncated_inputs"] - before["truncated_inputs"], 1)
        again = self.engine.embed(["短句"])["data"][0]
        self.assertFalse(again["truncated"])
        self.assertEqual(short["processed_tokens"], again["processed_tokens"])
        _, final = self.request(method="GET", path="/health")
        self.assertEqual(final["truncated_inputs"], after["truncated_inputs"])
        self.assertEqual(final["total_input_tokens"] - after["total_input_tokens"], again["input_tokens"])
        self.assertEqual(final["processed_tokens"] - after["processed_tokens"], again["processed_tokens"])

    def test_concurrent_inference_is_explicitly_rejected(self):
        before = self.engine.completed_requests
        with self.engine._inference_lock:
            status, result = self.request({"input": "中文财务证据"})
        self.assertEqual(status, 503)
        self.assertEqual(result["error"]["code"], "server_busy")
        self.assertEqual(self.engine.completed_requests, before)

    def test_health_reports_real_model_and_runtime(self):
        status, result = self.request(method="GET", path="/health")
        self.assertEqual(status, 200)
        self.assertEqual(result["model"], embedding.MODEL_ID)
        self.assertEqual(result["revision"], embedding.MODEL_REVISION)
        self.assertEqual(result["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(result["max_concurrency"], 1)
        self.assertLessEqual(result["threads"], 4)

    def test_bad_http_requests_do_not_reach_inference(self):
        before = self.engine.health()
        cases = [
            ({"input": "x", "model": "unknown"}, None, None, 404),
            ({"input": ["x"] * 17}, None, None, 400),
            ({"input": "x"}, b"invalid json", None, 400),
            ({"input": "x"}, b"x", {"Content-Length": str(embedding.MAX_BODY_BYTES + 1)}, 413),
            ({"input": "x"}, b"{}", {"Content-Type": "text/plain"}, 415),
            ({"input": "x"}, b"{}", {"Transfer-Encoding": "chunked"}, 400),
        ]
        for payload, raw, headers, expected in cases:
            with self.subTest(expected=expected, headers=headers):
                status, result = self.request(payload, raw=raw, headers=headers)
                self.assertEqual(status, expected, result)
                self.assertIn("error", result)
        self.assertEqual(self.engine.health(), before)


if __name__ == "__main__":
    unittest.main()
