"""Real OpenViking retrieval over benchmark-provided excerpts, not original PDFs.

Only input evidence_text is indexed. Source identity is a content hash; no gold,
claim labels, sample IDs, company labels, or invented PDF coordinates are used.
This separate adapter deliberately does not weaken B's PDF evidence contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote

from yjparse.openviking import OpenVikingClient, OpenVikingConfig


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunks(text, tokenizer, limit=384, overlap=48):
    """Token-aware original character slices; never silently truncate embeddings."""
    if not 16 <= limit <= 480 or not 0 <= overlap < limit:
        raise ValueError("invalid_chunk_limits")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    offsets = [(a, b) for a, b in encoded.offsets if b > a]
    if not offsets:
        raise ValueError("evidence_has_no_indexable_tokens")
    result = []
    for i in range(0, len(offsets), limit - overlap):
        end_index = min(i + limit, len(offsets))
        a = 0 if i == 0 else offsets[i][0]
        b = len(text) if end_index == len(offsets) else offsets[end_index - 1][1]
        piece = text[a:b]
        # Tokenization at a sliced boundary can differ; reject, never truncate.
        if len(tokenizer.encode(piece).ids) > 512:
            raise ValueError("chunk_exceeds_embedding_limit")
        result.append({"start": a, "end": b, "text": piece})
        if end_index == len(offsets):
            break
    return result


class ExcerptStore:
    def __init__(self, directory, config=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.client = OpenVikingClient(config or OpenVikingConfig(
            target_uri=self.manifest["target_uri"], timeout=20))
        if (self.client.config.target_uri != self.manifest["target_uri"] or
                self.client.config.base_url != self.manifest["server"]):
            raise ValueError("retrieval_store_binding_mismatch")
        self.sources = {r["id"]: r for r in self.manifest["sources"]}
        self.bindings = self.manifest["chunks"]
        if (len(self.sources) != len(self.manifest["sources"]) or not self.sources or
                any(not re.fullmatch(r"[a-f0-9]{64}", key) or row.get("sha256") != key
                    for key, row in self.sources.items())):
            raise ValueError("invalid_source_manifest")
        for chunk in self.bindings:
            if (chunk.get("source_id") not in self.sources or
                    not isinstance(chunk.get("root_uri"), str) or
                    not chunk["root_uri"].startswith(self.manifest["target_uri"] + "/") or
                    type(chunk.get("start")) is not int or type(chunk.get("end")) is not int or
                    not 0 <= chunk["start"] < chunk["end"] <= self.sources[chunk["source_id"]]["characters"]):
                raise ValueError("invalid_chunk_manifest")

    @classmethod
    def create(cls, directory, evidence_texts, tokenizer_path, namespace):
        from tokenizers import Tokenizer
        if not namespace or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in namespace):
            raise ValueError("invalid_namespace")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config = OpenVikingConfig(target_uri="viking://resources/" + namespace, timeout=20)
        client = OpenVikingClient(config)
        client.health()
        manifest = {"schema_version": 1, "role": "benchmark_provided_excerpt_pool",
                    "source_pdf_available": False, "server": config.base_url,
                    "target_uri": config.target_uri, "sources": [], "chunks": []}
        for text in sorted(set(evidence_texts), key=digest):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty_evidence")
            sid = digest(text)
            (directory / (sid + ".txt")).write_text(text, encoding="utf-8", newline="")
            manifest["sources"].append({"id": sid, "sha256": sid, "characters": len(text)})
            for index, chunk in enumerate(chunks(text, tokenizer)):
                target = config.target_uri + "/" + sid + "-" + str(index)
                upload = client.upload(chunk["text"].encode("utf-8"))
                added = client.request("POST", "/api/v1/resources", {
                    "temp_file_id": upload, "to": target, "create_parent": True,
                    "wait": False, "processing_mode": "vectors_only",
                    "args": {"parse_mode": "no_split"},
                    "reason": "Benchmark input excerpt; original PDF unavailable; no gold"})
                uri = added.get("root_uri", target)
                if uri != target and not uri.startswith(target + "/"):
                    raise ValueError("unexpected_ingest_uri")
                task_id = added.get("task_id")
                if not task_id:
                    raise ValueError("missing_ingest_task")
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    task = client.request("GET", "/api/v1/tasks/" + quote(str(task_id), safe=""))
                    if task.get("status") == "completed":
                        break
                    if task.get("status") not in {"pending", "running"}:
                        raise ValueError("ingest_failed")
                    time.sleep(0.1)
                else:
                    raise TimeoutError("ingest_timeout")
                manifest["chunks"].append({"source_id": sid, "root_uri": uri,
                                           "start": chunk["start"], "end": chunk["end"],
                                           "sha256": digest(chunk["text"])})
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(directory, config)

    def source(self, source_id):
        if source_id not in self.sources:
            raise ValueError("unknown_evidence_source")
        text = (self.directory / (source_id + ".txt")).read_bytes().decode("utf-8")
        if digest(text) != source_id:
            raise ValueError("evidence_source_changed")
        return text

    def search(self, query, limit=5):
        if not isinstance(query, str) or not 0 < len(query) <= 2000 or type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("invalid_search")
        found = self.client.request("POST", "/api/v1/search/find", {
            "query": query, "target_uri": self.manifest["target_uri"],
            "limit": limit, "context_type": "resource", "level": 2})
        hits = []
        for row in found.get("resources", [])[:limit]:
            uri = row.get("uri", "")
            binding = next((b for b in self.bindings if uri == b["root_uri"] or uri.startswith(b["root_uri"] + "/")), None)
            if binding is None:
                continue
            source = self.source(binding["source_id"])
            piece = source[binding["start"]:binding["end"]]
            raw = self.client.read(uri)
            if digest(piece) != binding["sha256"] or piece not in raw:
                raise ValueError("remote_excerpt_does_not_match_source")
            hits.append({"evidence_id": digest(uri), "source_id": binding["source_id"],
                         "uri": uri, "text": piece, "start": binding["start"], "end": binding["end"],
                         "score": row.get("score"), "provider": "openviking"})
        return hits

    def read(self, hit):
        # No model-selected path or arbitrary resource can be read.
        binding = next((b for b in self.bindings if hit.get("source_id") == b["source_id"] and
                       (hit.get("uri") == b["root_uri"] or str(hit.get("uri", "")).startswith(b["root_uri"] + "/"))), None)
        if binding is None:
            raise ValueError("unbound_evidence_read")
        source = self.source(binding["source_id"])
        raw = self.client.read(hit["uri"])
        piece = source[binding["start"]:binding["end"]]
        if digest(piece) != binding["sha256"] or piece not in raw:
            raise ValueError("remote_excerpt_changed")
        return {"text": source, "source_id": binding["source_id"], "verified": True,
                "source_kind": "benchmark_provided_excerpt", "original_pdf_verified": False,
                "read_scope": "matched_remote_chunk_expanded_to_hash_bound_local_excerpt"}
