"""External financial corpus registry and safe text-to-KB ingestion.

The corpus layer deliberately separates production evidence from evaluation data.
It creates the same KB contract as :mod:`yjparse.kb_export`, so BM25,
OpenViking and Pi can consume the result without a second adapter.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .utils import ensure_dir, sha256_file, write_json


TEXT_FIELDS = ("content", "context", "text", "evidence", "evidence_text",
               "context_full_page", "corrupted_text", "wrong_statement", "claim", "input")
TITLE_FIELDS = ("title", "document_title", "company", "question")
ID_FIELDS = ("document_id", "doc_id", "sample_id", "id", "accession_number")
ROLE_VALUES = {"production", "evaluation", "development", "quarantine"}
SAFE_NAME = re.compile(r"[^0-9A-Za-z._-]+")


def load_catalog(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported corpus catalog schema")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("catalog.sources must be a non-empty list")
    seen = set()
    for source in sources:
        sid = source.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("catalog source ids must be unique non-empty strings")
        seen.add(sid)
        if source.get("role") not in ROLE_VALUES:
            raise ValueError(f"invalid corpus role for {sid}")
        if not source.get("category") or not source.get("official_url"):
            raise ValueError(f"category and official_url are required for {sid}")
        if source.get("role") == "production" and source.get("license_status") == "unknown":
            raise ValueError(f"production source {sid} cannot have unknown license status")
    return payload


def catalog_summary(catalog: Dict[str, Any]) -> Dict[str, Any]:
    by_category: Dict[str, int] = {}
    by_role: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    for source in catalog["sources"]:
        for target, key in ((by_category, "category"), (by_role, "role"),
                            (by_status, "integration_status")):
            value = source.get(key, "unspecified")
            target[value] = target.get(value, 0) + 1
    return {"sources": len(catalog["sources"]), "by_category": by_category,
            "by_role": by_role, "by_status": by_status}


def _open_jsonl(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def iter_jsonl(paths: Iterable[Path]) -> Iterator[tuple[Path, int, Dict[str, Any]]]:
    for path in paths:
        with _open_jsonl(path) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: object required")
                yield path, line_number, value


def _first(record: Dict[str, Any], names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _chunks(text: str, limit: int = 1800) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result: List[str] = []
    for paragraph in paragraphs or [text.strip()]:
        while len(paragraph) > limit:
            cut = max(paragraph.rfind(mark, 0, limit) for mark in ("。", ". ", "; ", "\n"))
            cut = cut + 1 if cut >= limit // 3 else limit
            result.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            result.append(paragraph)
    return result


def import_jsonl(paths: Iterable[Path], kb_dir: Path, *, source_id: str,
                 category: str, role: str, project: str,
                 language: str = "unknown", allow_evaluation: bool = False,
                 max_records: Optional[int] = None) -> Dict[str, Any]:
    """Normalize JSONL records into the existing OpenViking KB contract.

    Evaluation/development corpora require an explicit opt-in and are exported
    to an isolated project namespace. Gold files are always rejected.
    """
    if role not in ROLE_VALUES:
        raise ValueError("invalid corpus role")
    if role != "production" and not allow_evaluation:
        raise ValueError("non-production corpus requires --allow-evaluation")
    resolved = [Path(p).resolve() for p in paths]
    if not resolved:
        raise ValueError("at least one input is required")
    if any("gold." in p.name.lower() or p.name.lower().startswith("gold") for p in resolved):
        raise ValueError("gold/answer files cannot be imported into a retrieval corpus")
    if role != "production" and not project.startswith("eval-"):
        raise ValueError("evaluation/development corpora require an eval-* project namespace")

    kb_dir = ensure_dir(Path(kb_dir))
    markdown_dir = ensure_dir(kb_dir / "markdown")
    documents, rows, skipped = [], [], []
    for ordinal, (path, line_number, record) in enumerate(iter_jsonl(resolved), 1):
        if max_records is not None and len(documents) >= max_records:
            break
        text = _first(record, TEXT_FIELDS)
        if not text:
            skipped.append({"path": str(path), "line": line_number, "reason": "NO_TEXT"})
            continue
        external_id = _first(record, ID_FIELDS) or f"line-{line_number}"
        identity = f"{source_id}\0{path}\0{external_id}".encode("utf-8")
        doc_id = SAFE_NAME.sub("-", source_id).strip("-") + "-" + hashlib.sha256(identity).hexdigest()[:20]
        title = _first(record, TITLE_FIELDS) or external_id
        record_sha = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
        chunks = _chunks(text)
        markdown = [f"<!-- source:{source_id} record_sha256:{record_sha} role:{role} -->",
                    "", f"# {title}", ""]
        for index, chunk in enumerate(chunks, 1):
            block_id = f"corpus-{index:04d}"
            markdown.extend([f"<!-- block:{block_id} page:1 type:paragraph bbox:[] -->", chunk, ""])
            rows.append({"doc_id": doc_id, "run_id": "corpus-import-v1", "source_path": str(path),
                         "page": 1, "page_size": None, "page_status": "ok", "block_id": block_id,
                         "type": "paragraph", "level": None, "bbox": [], "text": chunk,
                         "cells": [], "sentences": [], "caption": "", "source_note": "",
                         "confidence": None, "corpus_source": source_id, "corpus_category": category,
                         "corpus_role": role, "corpus_language": language,
                         "external_id": external_id, "record_sha256": record_sha})
        markdown_path = markdown_dir / f"{doc_id}.md"
        markdown_path.write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")
        documents.append({"doc_id": doc_id, "source_path": str(path), "sha256": record_sha,
                          "markdown": str(markdown_path), "markdown_sha256": sha256_file(markdown_path),
                          "pages": 1, "status": "ok", "corpus_source": source_id,
                          "corpus_category": category, "corpus_role": role,
                          "external_id": external_id})

    index_path = kb_dir / "kb_index.jsonl"
    with open(index_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"schema_version": "corpus-kb-1.0", "project": project,
                "corpus": {"source_id": source_id, "category": category, "role": role,
                           "language": language, "input_files": [str(p) for p in resolved]},
                "kb_index": str(index_path), "blocks": len(rows), "documents": documents,
                "skipped": skipped,
                "viking_targets": [{"doc_id": d["doc_id"], "markdown": d["markdown"],
                                    "uri": f"viking://resources/{project}/{d['doc_id']}"}
                                   for d in documents]}
    manifest_path = write_json(kb_dir / "ingest_manifest.json", manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest
