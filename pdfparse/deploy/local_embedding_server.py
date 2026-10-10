"""Offline Chinese BGE embeddings over a loopback-only OpenAI-compatible API.

The model and tokenizer are data files verified against a fixed HF revision.
No model download, remote code, API call, or paid inference occurs at runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading


MODEL_ID = "bge-small-zh-v1.5-onnx"
MODEL_REPOSITORY = "Xenova/bge-small-zh-v1.5"
MODEL_REVISION = "75c43b069aac4d136ba6bc1122f995fedcfd2781"
MODEL_FILES = {
    "onnx/model_quantized.onnx": "15b717c382bcb518ba457b93ea6850ede7f4f1cd8937454aa06972366cd19bcc",
    "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
}
DIMENSION = 512
MAX_TOKENS = 512
MAX_BATCH = 16
MAX_INPUT_CHARACTERS = 16384
MAX_BODY_BYTES = 262144
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data/local-openviking/models/bge-small-zh-v1.5"


class RequestError(ValueError):
    def __init__(self, message: str, code: str = "invalid_request", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def validate_request(payload: object) -> tuple[list[str], str]:
    if not isinstance(payload, dict):
        raise RequestError("The JSON body must be an object")
    if payload.get("model", MODEL_ID) != MODEL_ID:
        raise RequestError(f"Only model {MODEL_ID} is loaded", "model_not_found", 404)
    dimensions = payload.get("dimensions", DIMENSION)
    if type(dimensions) is not int or dimensions != DIMENSION:
        raise RequestError(f"This model produces exactly {DIMENSION} dimensions")
    encoding = payload.get("encoding_format", "float")
    if encoding not in ("float", "base64"):
        raise RequestError("encoding_format must be float or base64")
    values = payload.get("input")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_BATCH:
        raise RequestError(f"input must be a string or 1..{MAX_BATCH} strings")
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise RequestError("Each input must be a non-empty string; token-ID inputs are unsupported")
        if len(value) > MAX_INPUT_CHARACTERS:
            raise RequestError(f"Each input is limited to {MAX_INPUT_CHARACTERS} characters", "input_too_large", 413)
    return values, encoding


def verify_model_files(model_dir: Path) -> None:
    for relative, expected in MODEL_FILES.items():
        path = model_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Missing offline model file: {path}; see README-local-embedding.md")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for offline model file: {path}")


class EmbeddingEngine:
    def __init__(self, model_dir: Path, threads: int = 4):
        if not 1 <= threads <= 4:
            raise ValueError("threads must be between 1 and 4")
        self.model_dir = Path(model_dir).resolve()
        verify_model_files(self.model_dir)
        # The tokenizer otherwise uses an independent all-core Rayon worker pool.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        self.threads = threads
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.model_dir / "onnx/model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.input_names = {item.name for item in self.session.get_inputs()}
        required = {"input_ids", "attention_mask", "token_type_ids"}
        if not {"input_ids", "attention_mask"}.issubset(self.input_names) or not self.input_names.issubset(required):
            raise RuntimeError("Unexpected ONNX input contract")
        self._inference_lock = threading.Lock()
        self._stats = {
            "completed_requests": 0, "total_input_tokens": 0,
            "processed_tokens": 0, "truncated_inputs": 0,
        }

    @property
    def completed_requests(self) -> int:
        return self._stats["completed_requests"]

    def health(self) -> dict:
        return {
            "status": "ok", "model": MODEL_ID, "dimension": DIMENSION,
            "provider": "onnxruntime", "execution_provider": "CPUExecutionProvider",
            "model_repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
            "max_input_tokens": MAX_TOKENS, "max_batch_size": MAX_BATCH,
            "max_input_characters": MAX_INPUT_CHARACTERS, "max_body_bytes": MAX_BODY_BYTES,
            "threads": self.threads, "max_concurrency": 1,
            "network_access": "none", **self._stats,
        }

    def embed(self, texts: list[str], encoding: str = "float") -> dict:
        # Validate even direct callers; the HTTP boundary additionally bounds body bytes.
        texts, encoding = validate_request({"input": texts, "encoding_format": encoding})
        if not self._inference_lock.acquire(blocking=False):
            raise RequestError("Local CPU embedding worker is busy; retry later", "server_busy", 503)
        try:
            np = self.np
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
            full = self.tokenizer.encode_batch(texts, add_special_tokens=True)
            original_lengths = [len(item.ids) for item in full]
            self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            tokens = self.tokenizer.encode_batch(texts, add_special_tokens=True)
            input_data = {
                "input_ids": np.asarray([item.ids for item in tokens], dtype=np.int64),
                "attention_mask": np.asarray([item.attention_mask for item in tokens], dtype=np.int64),
                "token_type_ids": np.asarray([item.type_ids for item in tokens], dtype=np.int64),
            }
            outputs = self.session.run(None, {name: input_data[name] for name in self.input_names})
            hidden = np.asarray(outputs[0])
            if hidden.ndim != 3 or hidden.shape[0] != len(texts) or hidden.shape[2] != DIMENSION:
                raise RuntimeError("Unexpected ONNX output contract")
            # BGE uses the final CLS hidden state, not mean pooling.
            vectors = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any():
                raise RuntimeError("Embedding contains non-finite or zero-norm values")
            vectors = vectors / norms
            processed_lengths = [sum(item.attention_mask) for item in tokens]
            data = []
            for index, vector in enumerate(vectors):
                value = vector.tolist() if encoding == "float" else base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")
                data.append({
                    "object": "embedding", "index": index, "embedding": value,
                    "input_tokens": original_lengths[index],
                    "processed_tokens": processed_lengths[index],
                    "truncated": original_lengths[index] > processed_lengths[index],
                    "truncated_tokens": original_lengths[index] - processed_lengths[index],
                })
            used_tokens = sum(processed_lengths)
            original_tokens = sum(original_lengths)
            truncated_inputs = sum(item["truncated"] for item in data)
            result = {
                "object": "list", "model": MODEL_ID, "data": data,
                "usage": {"prompt_tokens": used_tokens, "total_tokens": used_tokens},
                "truncation": {
                    "max_input_tokens": MAX_TOKENS,
                    "input_tokens": original_tokens, "processed_tokens": used_tokens,
                    "truncated_inputs": truncated_inputs,
                    "policy": "right_truncate_including_special_tokens",
                },
            }
            # Replace one numeric-only snapshot after success. Concurrent health reads
            # see a consistent set of totals without waiting for the inference lock.
            previous = self._stats
            self._stats = {
                "completed_requests": previous["completed_requests"] + 1,
                "total_input_tokens": previous["total_input_tokens"] + original_tokens,
                "processed_tokens": previous["processed_tokens"] + used_tokens,
                "truncated_inputs": previous["truncated_inputs"] + truncated_inputs,
            }
            return result
        finally:
            self._inference_lock.release()


def make_server(engine: EmbeddingEngine, host: str = "127.0.0.1", port: int = 1934) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Only the IPv4 loopback host 127.0.0.1 (or localhost) is allowed")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            # Do not log input text, request bodies, URL parameters, or credentials.
            pass

        def respond(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def error(self, status: int, code: str, message: str):
            self.respond(status, {"error": {"message": message, "type": "local_embedding_error", "code": code}})

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, engine.health())
            else:
                self.error(404, "not_found", "Unknown endpoint")

        def do_POST(self):
            if self.path != "/v1/embeddings":
                self.error(404, "not_found", "Unknown endpoint")
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise RequestError("Chunked request bodies are unsupported")
                content_lengths = self.headers.get_all("Content-Length", [])
                if not content_lengths:
                    raise RequestError("Content-Length is required", "length_required", 411)
                if len(content_lengths) != 1:
                    raise RequestError("Exactly one Content-Length is required")
                try:
                    length = int(content_lengths[0])
                except ValueError:
                    raise RequestError("Invalid Content-Length") from None
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise RequestError(f"Request body must be 1..{MAX_BODY_BYTES} bytes", "request_too_large", 413)
                if self.headers.get_content_type() != "application/json":
                    raise RequestError("Content-Type must be application/json", "unsupported_media_type", 415)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise RequestError("Incomplete request body")
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise RequestError("Malformed UTF-8 JSON") from None
                texts, encoding = validate_request(payload)
                self.respond(200, engine.embed(texts, encoding))
            except RequestError as exc:
                self.error(exc.status, exc.code, str(exc))
            except TimeoutError:
                self.error(408, "request_timeout", "Timed out reading the request")
            except Exception as exc:
                print(f"Local embedding request failed: {type(exc).__name__}", file=sys.stderr, flush=True)
                self.error(500, "inference_failed", "Local embedding inference failed")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1934)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        engine = EmbeddingEngine(args.model_dir, args.threads)
        server = make_server(engine, args.host, args.port)
    except Exception as exc:
        print(f"Cannot start local embedding server: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}", **engine.health()}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
