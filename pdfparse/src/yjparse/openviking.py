"""OpenViking 0.4.23 HTTP integration with locally anchored evidence.

OpenViking ranks resources; only kb_index.jsonl supplies evidence text.  No
generated abstract/overview is promoted to original evidence.  This module uses
the Python standard library and never starts a model or installs a server.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .retrieval import Bm25Index, load_index

API_CONTRACT_VERSION = "0.4.23"
ANCHOR = re.compile(r"<!--\s*block:([^\s>]+)\s+page:(\d+)\b[^>]*-->")
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class OpenVikingError(RuntimeError):
    def __init__(self, code: str, operation: str):
        self.code, self.operation = code, operation
        # Never include remote response bodies, API keys, or URL query strings.
        super().__init__(f"OpenViking {operation}: {code}")

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "operation": self.operation, "message": str(self)}


@dataclass(frozen=True)
class OpenVikingConfig:
    base_url: str = "http://127.0.0.1:1933"
    api_key: str = field(default="", repr=False)
    timeout: float = 10.0
    target_uri: str = "viking://resources/research-reports"

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or
                url.username or url.password or url.query or url.fragment):
            raise ValueError("base_url must be an HTTP(S) server URL without credentials/query")
        # Match the Pi transport boundary: HTTP is only for explicit loopback
        # development hosts, never a remote endpoint carrying source text/keys.
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("base_url must use HTTPS outside loopback")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        _validate_uri(self.target_uri)

    @classmethod
    def from_env(cls) -> "OpenVikingConfig":
        return cls(base_url=os.getenv("OPENVIKING_URL", cls.base_url),
                   api_key=os.getenv("OPENVIKING_API_KEY", ""),
                   timeout=float(os.getenv("OPENVIKING_TIMEOUT", "10")),
                   target_uri=os.getenv("OPENVIKING_TARGET_URI", cls.target_uri))


def _validate_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    path = unquote(parsed.path)
    if (parsed.scheme != "viking" or parsed.netloc != "resources" or
            parsed.query or parsed.fragment or "\\" in path or
            any(part in {".", ".."} for part in path.split("/"))):
        raise ValueError("Expected a canonical viking://resources/... URI")
    return uri.rstrip("/")


def _within(uri: str, root: str) -> bool:
    try:
        uri, root = _validate_uri(uri), _validate_uri(root)
    except (ValueError, TypeError):
        return False
    return uri == root or uri.startswith(root + "/")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward a tenant API key to another server.


class OpenVikingClient:
    def __init__(self, config: Optional[OpenVikingConfig] = None, cancel_check=None):
        self.config = config or OpenVikingConfig.from_env()
        self.cancel_check = cancel_check
        self._opener = build_opener(_NoRedirect())

    def request(self, method: str, path: str, payload=None, *, body=None,
                content_type="application/json", timeout=None):
        if self.cancel_check is not None:
            self.cancel_check()
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else body
        headers = {"Accept": "application/json", "Content-Type": content_type}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        request = Request(self.config.base_url.rstrip("/") + path,
                          data=data, headers=headers, method=method)
        operation = method + " " + path.split("?", 1)[0]
        try:
            with self._opener.open(request, timeout=timeout or self.config.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise OpenVikingError(f"HTTP_{exc.code}", operation) from None
        except (TimeoutError, socket.timeout):
            raise OpenVikingError("TIMEOUT", operation) from None
        except URLError as exc:
            code = "TIMEOUT" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "UNAVAILABLE"
            raise OpenVikingError(code, operation) from None
        except OSError:
            raise OpenVikingError("CONNECTION_ERROR", operation) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise OpenVikingError("RESPONSE_TOO_LARGE", operation)
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError):
            raise OpenVikingError("INVALID_JSON", operation) from None
        if not isinstance(data, dict) or data.get("status") != "ok":
            raise OpenVikingError("API_ERROR", operation)
        if self.cancel_check is not None:
            self.cancel_check()
        return data.get("result", data)

    def health(self):
        return self.request("GET", "/health")

    def upload(self, markdown: bytes):
        boundary = "yjparse-" + uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                "filename=\"document.md\"\r\nContent-Type: text/markdown; charset=utf-8\r\n\r\n").encode()
        body += markdown + f"\r\n--{boundary}--\r\n".encode()
        result = self.request("POST", "/api/v1/resources/temp_upload", body=body,
                              content_type=f"multipart/form-data; boundary={boundary}")
        if not isinstance(result, dict) or not result.get("temp_file_id"):
            raise OpenVikingError("MISSING_TEMP_FILE_ID", "upload")
        return result["temp_file_id"]

    def read(self, uri: str) -> str:
        _validate_uri(uri)
        result = self.request("GET", "/api/v1/content/read?" + urlencode({"uri": uri, "raw": "true"}))
        if not isinstance(result, str):
            raise OpenVikingError("INVALID_CONTENT", "read")
        return result


def _base(status="ok", used=False, provider="openviking", **extra):
    return {"provider": provider, "status": status, "openviking_used": used,
            "api_contract_version": API_CONTRACT_VERSION, "errors": [], **extra}


def health(config: Optional[OpenVikingConfig] = None, cancel_check=None) -> Dict[str, Any]:
    """A healthy process does not establish embedding/VLM availability."""
    try:
        result = OpenVikingClient(config, cancel_check=cancel_check).health()
        if not isinstance(result, dict) or result.get("healthy") is not True:
            raise OpenVikingError("UNHEALTHY", "health")
        return _base(used=True, available=True, server_version=result.get("version"),
                     model_readiness="not_checked")
    except OpenVikingError as exc:
        return _base("unavailable", available=False, errors=[exc.as_dict()])


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def config_for_kb(kb_dir: Path, config: Optional[OpenVikingConfig] = None) -> OpenVikingConfig:
    """Resolve only the resource scope from a trusted local export manifest.

    Explicit configuration and an explicit target environment variable take
    precedence. The manifest never controls the server, credentials, or timeout.
    """
    if config is not None:
        return config
    resolved = OpenVikingConfig.from_env()
    if "OPENVIKING_TARGET_URI" in os.environ:
        return resolved
    manifest_path = Path(kb_dir) / "ingest_manifest.json"
    if not manifest_path.exists():
        return resolved
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("ingest_manifest.json must contain an object")
    if "project" not in manifest:
        return resolved
    project = manifest["project"]
    if (not isinstance(project, str) or project in {".", ".."} or
            re.fullmatch(r"[\w.-]{1,128}", project) is None):
        raise ValueError("ingest_manifest.project must be a single 1..128 character name using letters, digits, underscore, hyphen, or dot")
    return replace(resolved, target_uri="viking://resources/" + project)


def _local(kb_dir: Path):
    kb_dir = Path(kb_dir)
    rows = load_index(kb_dir / "kb_index.jsonl")
    lookup = {}
    for row in rows:
        key = (row.get("doc_id"), row.get("block_id"))
        if key in lookup or not all(key):
            raise ValueError("kb_index requires unique non-empty (doc_id, block_id)")
        lookup[key] = row
    bindings_path = kb_dir / "openviking_bindings.json"
    bindings = _read_json(bindings_path).get("documents", []) if bindings_path.exists() else []
    return rows, lookup, bindings, _sha((kb_dir / "kb_index.jsonl").read_bytes())


def ingest_kb(kb_dir: Path, config: Optional[OpenVikingConfig] = None,
              wait: bool = True, processing_timeout: float = 60.0,
              cancel_check=None) -> Dict[str, Any]:
    """Upload exported Markdown and explicitly track processing completion.

    This operation may invoke the server's configured embedding model. Calling
    health does not invoke it. Ingest uses vectors_only and preserves Markdown
    body anchors (no_split); no VLM summary generation is requested.
    """
    if not math.isfinite(processing_timeout) or processing_timeout <= 0:
        raise ValueError("processing_timeout must be positive and finite")
    kb_dir = Path(kb_dir)
    client = OpenVikingClient(config_for_kb(kb_dir, config), cancel_check=cancel_check)
    rows, _, bindings, index_sha = _local(kb_dir)
    manifest = _read_json(kb_dir / "ingest_manifest.json")
    docs = manifest.get("documents", [])
    result = _base(documents=[])
    for doc in docs:
        doc_id = doc["doc_id"]
        if not isinstance(doc_id, str) or any(x in doc_id for x in ["/", "\\"]) or doc_id in {".", ".."}:
            raise ValueError("doc_id cannot contain path separators")
        markdown_path = kb_dir / "markdown" / f"{doc_id}.md"
        raw = markdown_path.read_bytes()
        expected = doc.get("markdown_sha256")
        anchors = {(block, int(page)) for block, page in ANCHOR.findall(raw.decode("utf-8"))}
        local_anchors = {(r["block_id"], r["page"]) for r in rows
                         if r["doc_id"] == doc_id and r.get("page_status") != "fail"}
        if not expected or _sha(raw) != expected or not anchors or anchors != local_anchors:
            result["errors"].append({"code": "EXPORT_INDEX_MISMATCH", "doc_id": doc_id})
            continue
        target = client.config.target_uri.rstrip("/") + "/" + quote(doc_id, safe="")
        item = {"doc_id": doc_id, "uri": target, "status": "failed"}
        try:
            upload_id = client.upload(raw)
            added = client.request("POST", "/api/v1/resources", {
                "temp_file_id": upload_id, "to": target, "create_parent": True,
                "wait": False, "processing_mode": "vectors_only",
                "args": {"parse_mode": "no_split"},
                "reason": "B parser export with page/block anchors"})
            result["openviking_used"] = True
            if not isinstance(added, dict):
                raise OpenVikingError("INVALID_INGEST_RESULT", "ingest")
            root_uri = added.get("root_uri", target)
            if not _within(root_uri, target):
                raise OpenVikingError("UNEXPECTED_RESOURCE_URI", "ingest")
            task_id = added.get("task_id")
            item.update(uri=root_uri, task_id=task_id, status="submitted")
            if wait:
                if not task_id:
                    raise OpenVikingError("MISSING_TASK_ID", "ingest")
                deadline = time.monotonic() + processing_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise OpenVikingError("PROCESSING_TIMEOUT", "ingest")
                    task = client.request("GET", "/api/v1/tasks/" + quote(str(task_id), safe=""),
                                          timeout=min(client.config.timeout, remaining))
                    if not isinstance(task, dict):
                        raise OpenVikingError("INVALID_TASK", "ingest")
                    state = task.get("status")
                    if state == "completed":
                        item["status"] = "completed"
                        break
                    if state in {"failed", "cancelled"}:
                        raise OpenVikingError("PROCESSING_" + state.upper(), "ingest")
                    if state not in {"pending", "running", "cancelling"}:
                        raise OpenVikingError("INVALID_TASK_STATUS", "ingest")
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            binding = {"doc_id": doc_id, "root_uri": root_uri,
                       "markdown_sha256": expected, "kb_index_sha256": index_sha,
                       "source_sha256": doc.get("sha256"), "status": item["status"],
                       "server": client.config.base_url.rstrip("/")}
            bindings = [b for b in bindings if b.get("doc_id") != doc_id] + [binding]
        except OpenVikingError as exc:
            item["status"] = "failed"
            item["error"] = exc.as_dict()
            result["errors"].append({**exc.as_dict(), "doc_id": doc_id})
            # An unsuccessful replacement must not leave an old trusted binding.
            bindings = [b for b in bindings if b.get("doc_id") != doc_id]
        result["documents"].append(item)
    (kb_dir / "openviking_bindings.json").write_text(
        json.dumps({"api_contract_version": API_CONTRACT_VERSION, "documents": bindings},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if result["errors"]:
        result["status"] = "partial" if any(d["status"] == "completed" for d in result["documents"]) else "failed"
    elif not docs:
        result["status"] = "empty"
    elif not wait:
        result["status"] = "submitted"
    return result


def _binding_for(uri, bindings, client, index_sha):
    candidates = [b for b in bindings if _within(uri, b.get("root_uri", ""))]
    if len(candidates) != 1:
        return None, "UNBOUND_RESOURCE"
    binding = candidates[0]
    if binding.get("status") != "completed":
        return None, "INGEST_NOT_COMPLETED"
    if binding.get("server") != client.config.base_url.rstrip("/"):
        return None, "SERVER_MISMATCH"
    if binding.get("kb_index_sha256") != index_sha:
        return None, "STALE_LOCAL_INDEX"
    if not _within(uri, client.config.target_uri):
        return None, "OUTSIDE_SCOPE"
    return binding, None


def _document_hash(kb_dir, doc_id):
    path = Path(kb_dir) / "ingest_manifest.json"
    if not path.exists():
        return None
    documents = _read_json(path).get("documents", [])
    return next((d.get("sha256") for d in documents if d.get("doc_id") == doc_id), None)


def _evidence(row, uri=None, source_sha256=None):
    # Full original text/cells, never remote abstract and never a 300-char snippet.
    identity = (str(row["doc_id"]) + "\0" + str(row["block_id"])).encode("utf-8")
    return {**row, "evidence_status": "confirmed", "evidence_source": "kb_index",
            "evidence_id": "kb-" + _sha(identity)[:24], "verified": True,
            "verification_scope": "openviking_anchor_and_local_index" if uri else "local_index",
            "source_file_verified": False, "sha256": source_sha256,
            "source_sha256": source_sha256, "uri": uri,
            "text": row.get("text") or "", "cells": row.get("cells") or []}


def _resolve_content(content, binding, lookup):
    evidence, rejected = [], []
    seen = set()
    for block_id, page in ANCHOR.findall(content):
        key = (binding["doc_id"], block_id)
        if key in seen:
            continue
        seen.add(key)
        row = lookup.get(key)
        if row is None:
            rejected.append({"doc_id": key[0], "block_id": block_id, "reason": "UNKNOWN_BLOCK"})
        elif row.get("page_status") == "fail" or row.get("page") != int(page):
            rejected.append({"doc_id": key[0], "block_id": block_id, "reason": "INVALID_PAGE_ANCHOR"})
        else:
            evidence.append(row)
    return evidence, rejected


def search_kb(kb_dir: Path, query: str, top_k: int = 5,
              doc_ids: Optional[Iterable[str]] = None,
              config: Optional[OpenVikingConfig] = None,
              allow_fallback: bool = False, cancel_check=None) -> Dict[str, Any]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    rows, lookup, bindings, index_sha = _local(Path(kb_dir))
    client = OpenVikingClient(config_for_kb(kb_dir, config), cancel_check=cancel_check)
    allowed = set(doc_ids) if doc_ids is not None else None
    result = _base(hits=[], unconfirmed=[])
    if not query.strip():
        return result
    try:
        found = client.request("POST", "/api/v1/search/find", {
            "query": query, "target_uri": client.config.target_uri,
            "limit": top_k, "context_type": "resource", "level": 2})
        result["openviking_used"] = True
        if not isinstance(found, dict) or not isinstance(found.get("resources"), list):
            raise OpenVikingError("INVALID_SEARCH_RESULT", "search")
        seen = set()
        for candidate in found["resources"][:top_k]:
            if not isinstance(candidate, dict):
                result["unconfirmed"].append({"reason": "INVALID_CANDIDATE"})
                continue
            uri = candidate.get("uri", "")
            binding, reason = _binding_for(uri, bindings, client, index_sha)
            if reason:
                result["unconfirmed"].append({"uri": uri, "reason": reason})
                continue
            if allowed is not None and binding["doc_id"] not in allowed:
                continue
            try:
                original = client.read(uri)
            except OpenVikingError as exc:
                result["errors"].append(exc.as_dict())
                result["unconfirmed"].append({"uri": uri, "reason": "CONTENT_READ_FAILED"})
                continue
            evidence, rejected = _resolve_content(original, binding, lookup)
            result["unconfirmed"].extend({**r, "uri": uri} for r in rejected)
            if not evidence and not rejected:
                result["unconfirmed"].append({"uri": uri, "reason": "NO_ORIGINAL_BLOCK_ANCHORS"})
            # OV ranks documents; BM25 only selects blocks inside the raw document.
            selected = Bm25Index(evidence).search(query, top_k=top_k)
            ranked = [lookup[(h["doc_id"], h["block_id"])] for h in selected] or evidence[:top_k]
            for row in ranked:
                key = (row["doc_id"], row["block_id"])
                if key in seen:
                    continue
                seen.add(key)
                hit = _evidence(row, uri, binding.get("source_sha256"))
                hit.update(provider="openviking", score=candidate.get("score"),
                           block_selection="local_bm25" if selected else "source_order")
                result["hits"].append(hit)
        result["hits"] = result["hits"][:top_k]
        if result["unconfirmed"] or result["errors"]:
            result["status"] = "partial" if result["hits"] else "unconfirmed"
    except OpenVikingError as exc:
        result["errors"].append(exc.as_dict())
        result["status"] = "unavailable"
        if allow_fallback:
            eligible = [r for r in rows if allowed is None or r.get("doc_id") in allowed]
            hits = Bm25Index(eligible).search(query, top_k)
            result["hits"] = [{**_evidence(lookup[(h["doc_id"], h["block_id"])],
                                                   source_sha256=_document_hash(kb_dir, h["doc_id"])),
                               "score": h["score"], "provider": "bm25"} for h in hits]
            result.update(provider="bm25", status="fallback", openviking_used=False,
                          fallback_from="openviking")
    return result


def read_evidence(kb_dir: Path, doc_id: str, block_id: str,
                  config: Optional[OpenVikingConfig] = None,
                  uri: Optional[str] = None,
                  allow_fallback: bool = False, cancel_check=None) -> Dict[str, Any]:
    """Read full local evidence; with uri, require a live OV raw-content backlink."""
    config = config_for_kb(kb_dir, config)
    _, lookup, bindings, index_sha = _local(Path(kb_dir))
    row = lookup.get((doc_id, block_id))
    if row is None or row.get("page_status") == "fail":
        return _base("unconfirmed", provider="kb_index", evidence=None,
                     errors=[{"code": "UNKNOWN_OR_REJECTED_BLOCK"}])
    if uri is None:
        return _base(provider="kb_index", evidence=_evidence(
            row, source_sha256=_document_hash(kb_dir, doc_id)))
    client = OpenVikingClient(config, cancel_check=cancel_check)
    binding, reason = _binding_for(uri, bindings, client, index_sha)
    if reason or binding["doc_id"] != doc_id:
        return _base("unconfirmed", evidence=None, errors=[{"code": reason or "DOCUMENT_MISMATCH"}])
    try:
        content = client.read(uri)
        evidence, _ = _resolve_content(content, binding, lookup)
        if not any(r["block_id"] == block_id for r in evidence):
            return _base("unconfirmed", used=True, evidence=None, errors=[{"code": "BLOCK_NOT_IN_REMOTE_CONTENT"}])
        return _base(used=True, evidence=_evidence(row, uri, binding.get("source_sha256")))
    except OpenVikingError as exc:
        if allow_fallback:
            return _base(provider="kb_index", evidence=_evidence(
                row, source_sha256=_document_hash(kb_dir, doc_id)),
                fallback_from="openviking", errors=[exc.as_dict()])
        return _base("unavailable", evidence=None, errors=[exc.as_dict()])
