"""Pi-controlled business workflow with source-bound OpenViking evidence tools."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import uuid

from .adapters import bind_company, file_hash, load_document
from .pi_bridge import run_pi
from .pipeline import check_documents, write_result
from .execution_scope import check_active, remaining_timeout


class EvidenceWorkflow:
    """An agent may select evidence, but cannot supply source paths or verdicts."""

    def __init__(self, detect, recheck, *, kb_dir=None, ov_config=None, allow_fallback=False):
        self.detect_callback, self.recheck_callback = detect, recheck
        self.kb_dir = Path(kb_dir).resolve() if kb_dir else None
        self.ov_config = ov_config
        self.allow_fallback = allow_fallback
        self.result = None
        self.hits, self.reads = {}, {}
        self.events = []

    def _view(self):
        paired = "findings" in self.result
        candidates = self.result.get("findings" if paired else "errors", [])
        pending = [c for c in candidates if c.get("status") == "needs_review"]
        complete = self.result.get("summary" if paired else "coverage", {}).get("complete", False)
        view = {"status": "needs_review" if pending or not complete else "completed", "summary": self.result.get("summary", {
                    "candidates": len(candidates), "needs_review": len(pending)}),
                "pending_checks": [{"id": c.get("id"), "type": c.get("error_type"),
                                    "reason": (c.get("reason") or c.get("message") or "")[:600],
                                    "text": (c.get("claim", {}).get("text") or
                                             " ".join(s.get("text", "") for s in c.get("spans", [])))[:800],
                                    "evidence_request": c.get("evidence_request", [])}
                                   for c in pending[:12]],
                "pending_checks_total": len(pending), "pending_checks_truncated": len(pending) > 12,
                "result_ref": self.result.get("run_id", self.result.get("document_id")),
                "coverage": self.result.get("coverage", self.result.get("summary", {}).get("coverage"))}
        if self.result.get("external_fact_check") is not None:
            external = self.result["external_fact_check"]
            view["external_fact_check"] = {
                "summary": external.get("summary", {}),
                "findings": [{"id": f.get("id"), "status": f.get("status"),
                              "rule_id": f.get("rule_id"), "message": str(f.get("message", ""))[:600]}
                             for f in external.get("findings", [])[:12]],
                "findings_total": len(external.get("findings", [])),
                "scored_separately": True,
            }
        return view

    def __call__(self, name, args):
        check_active()
        if name == "detect_document":
            if args:
                raise ValueError("detect_uses_bound_document")
            if self.result is None:
                detected = self.detect_callback()
                check_active()
                self.result = detected
                self.events.append({"tool": name, "status": "ok"})
            return self._view()
        if self.result is None:
            raise ValueError("detect_document_required_first")
        if not self.kb_dir:
            raise ValueError("evidence_store_not_configured")
        from yjparse.openviking import config_for_kb, search_kb, read_evidence
        ov_config = config_for_kb(self.kb_dir, self.ov_config)
        ov_config = replace(ov_config, timeout=remaining_timeout(ov_config.timeout))
        if name == "search_evidence":
            if set(args) - {"query", "limit"} or not isinstance(args.get("query"), str) or not 0 < len(args["query"]) <= 2000:
                raise ValueError("invalid_evidence_query")
            limit = args.get("limit", 5)
            if type(limit) is not int or not 1 <= limit <= 10:
                raise ValueError("invalid_evidence_limit")
            response = search_kb(self.kb_dir, args["query"], top_k=limit, config=ov_config,
                                 allow_fallback=self.allow_fallback, cancel_check=check_active)
            check_active()
            result = deepcopy(response)
            result["hits"] = []
            for hit in response.get("hits", []):
                if not hit.get("doc_id") or not hit.get("block_id"):
                    continue
                eid = sha256(json.dumps([hit["doc_id"], hit["block_id"]]).encode()).hexdigest()[:24]
                self.hits[eid] = hit
                result["hits"].append({"evidence_id": eid, "doc_id": hit["doc_id"],
                                       "block_id": hit["block_id"], "page": hit.get("page"),
                                       "text": str(hit.get("text", ""))[:1200]})
            self.events.append({"tool": name, "status": response.get("status"),
                                "provider": response.get("provider"), "hits": len(result["hits"])})
            return result
        if name == "read_evidence":
            if set(args) != {"evidence_id"} or args.get("evidence_id") not in self.hits:
                raise ValueError("evidence_must_come_from_search")
            eid = args["evidence_id"]
            hit = self.hits[eid]
            response = read_evidence(self.kb_dir, hit["doc_id"], hit["block_id"],
                                     config=ov_config, uri=hit.get("uri"), allow_fallback=self.allow_fallback,
                                     cancel_check=check_active)
            check_active()
            self.reads[eid] = response
            self.events.append({"tool": name, "status": response.get("status"), "evidence_id": eid})
            return {**response, "evidence_id": eid}
        if name == "recheck":
            ids = args.get("evidence_ids")
            if (set(args) != {"evidence_ids"} or not isinstance(ids, list) or not ids or len(ids) > 10
                    or any(not isinstance(eid, str) or eid not in self.reads for eid in ids)):
                raise ValueError("read_evidence_required_before_recheck")
            paths = self._verified_source_paths(ids)
            checked = self.recheck_callback(paths, self.result)
            check_active()
            self.result = checked
            self.events.append({"tool": name, "status": "ok", "evidence_ids": ids,
                                "source_documents": len(paths)})
            return self._view()
        raise ValueError("unsupported_agent_tool")

    def _verified_source_paths(self, ids):
        manifest = json.loads((self.kb_dir / "ingest_manifest.json").read_text(encoding="utf-8"))
        documents = {d["doc_id"]: d for d in manifest.get("documents", [])}
        if len(documents) != len(manifest.get("documents", [])):
            raise ValueError("duplicate_source_manifest_identity")
        paths = []
        for eid in ids:
            hit, response = self.hits[eid], self.reads[eid]
            # Provider statuses are transport results, not business verdicts.
            if response.get("status") not in {"ok", "ready"}:
                raise ValueError("evidence_read_not_verified")
            evidence = response.get("evidence")
            if (not isinstance(evidence, dict) or evidence.get("verified") is not True
                    or evidence.get("doc_id") != hit["doc_id"] or evidence.get("block_id") != hit["block_id"]):
                raise ValueError("evidence_identity_not_verified")
            doc = documents.get(hit["doc_id"])
            if not doc or not doc.get("sha256") or not doc.get("source_path"):
                raise ValueError("source_manifest_missing")
            if evidence.get("source_sha256") != doc["sha256"]:
                raise ValueError("evidence_source_hash_mismatch")
            path = Path(doc["source_path"])
            if not path.is_absolute():
                path = self.kb_dir / path
            path = path.resolve()
            if not path.is_file() or file_hash(path) != doc["sha256"]:
                raise ValueError("source_changed_or_unavailable")
            paths.append(str(path))
        return list(dict.fromkeys(paths))


def _retain_previous_candidates(current, previous):
    """A source-only rerun cannot silently remove model-only claims or failures."""
    current_ids = {f["id"] for f in current["findings"]}
    retained = [deepcopy(f) for f in previous["findings"] if f["id"] not in current_ids]
    current["findings"].extend(retained)
    findings, summary = current["findings"], current["summary"]
    for status in ("confirmed_error", "needs_review", "no_issue"):
        summary[status] = sum(f["status"] == status for f in findings)
    fact_ids = {f["id"] for f in findings
                if f.get("claim", {}).get("metric") != "document_coverage"
                and f.get("claim", {}).get("attributes", {}).get("extraction") != "intrinsic"}
    review_ids = {f["id"] for f in findings if f["status"] == "needs_review"}
    summary["claims"] = len(fact_ids)
    summary["automatic_claims"] = len(fact_ids - review_ids)
    summary["review_claims"] = len(fact_ids & review_ids)
    summary["intrinsic_candidates"] = sum(f.get("claim", {}).get("attributes", {}).get("extraction") == "intrinsic" for f in findings)
    summary["model_candidates_aligned"] = previous.get("summary", {}).get("model_candidates_aligned", 0)
    summary["needs_review_by_rule"] = dict(sorted(Counter(f.get("rule_id", "") for f in findings if f["status"] == "needs_review").items()))
    summary["evidence_requests"] = sum(f.get("decision") == "ask" for f in findings)
    summary["evidence_request_items"] = sum(len(f.get("evidence_request", [])) for f in findings)
    summary["complete"] = summary["complete"] and not summary["needs_review"]
    if previous.get("text_review"):
        current["text_review"] = deepcopy(previous["text_review"])
        for key in ("text_candidates", "text_confirmed", "text_needs_review", "text_rejected_hints", "text_review_total_hints"):
            if key in previous["summary"]:
                summary[key] = previous["summary"][key]
    current["model_traces"] = deepcopy(previous.get("model_traces", []))
    summary["complete"] = summary["complete"] and all(t.get("status") == "ok" for t in current["model_traces"])
    return current


def run_agent_check(report_path, source_paths, out, *, company=None, engine="pdfplumber",
                    model_config, runtime_settings=None, limits=None, kb_dir=None,
                    ov_config=None, allow_fallback=False, detect_with_model=False, runner=run_pi):
    """Parse once, then expose existing C checks as bounded Pi business tools."""
    out = Path(out)
    work = out / "work" / uuid.uuid4().hex
    report = load_document(report_path, "report", work, engine)
    initial_sources = [load_document(p, "source", work, engine) for p in source_paths]
    bind_company(report, initial_sources, company)
    source_map = {str(Path(d.path).resolve()): d for d in initial_sources}

    def detect():
        check_active()
        return check_documents(report, list(source_map.values()), model_config if detect_with_model else None,
                               runtime_settings=runtime_settings)

    def recheck(paths, previous):
        for path in paths:
            check_active()
            if path not in source_map:
                source_map[path] = load_document(path, "source", work, engine)
        check_active()
        bind_company(report, list(source_map.values()), company)
        current = check_documents(report, list(source_map.values()))
        return _retain_previous_candidates(current, previous)

    workflow = EvidenceWorkflow(detect, recheck, kb_dir=kb_dir, ov_config=ov_config, allow_fallback=allow_fallback)
    runtime = runner({"documentId": "document", "question": "检测绑定的研报；仅在需要外部证据时检索、读原文并重新核查。"},
                     workflow, model_config, settings=runtime_settings, limits=limits)
    if workflow.result is None:
        dest = out / "agent_failures"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / (runtime.get("request_id", uuid.uuid4().hex) + ".json")).write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("Pi did not complete initial detection: " + str(runtime.get("stop_reason")))
    result = workflow.result
    result["agent_runtime"] = {**runtime, "workflow_events": workflow.events,
                               "verdict_authority": "python_evidence_rules",
                               "openviking_required": bool(kb_dir), "fallback_allowed": allow_fallback}
    if runtime.get("status") in {"failed", "stopped"}:
        result["summary"]["complete"] = False
        result["agent_runtime"]["incomplete"] = True
    return result, write_result(result, out)


def run_agent_text(content, *, document_id, model_config, runtime_settings=None, limits=None,
                   kb_dir=None, ov_config=None, allow_fallback=False, examples=(),
                   company=None, out=None, runner=run_pi):
    """Text detection stays FinED-compatible; source checks are a separate track."""
    from .model_runtime import BudgetedChatClient
    from .models import Block, Document
    from .text_review import detect_text
    client = BudgetedChatClient(model_config, runtime_settings)
    out = Path(out or ROOT_OUTPUT)
    session_dir = out / uuid.uuid4().hex
    session_dir.mkdir(parents=True, exist_ok=False)
    original = session_dir / "input.txt"
    original.write_text(content, encoding="utf-8")
    report_doc = Document("text-" + sha256(content.encode()).hexdigest()[:24], file_hash(original),
                          session_dir.name, str(original.resolve()), "report",
                          blocks=[Block("text", content, paragraph=1)], metadata={"format": "text"})

    def detect():
        check_active()
        return detect_text(content, document_id=document_id, detector="hybrid", chat=client, examples=examples,
                           max_input_tokens=client.settings.context_tokens - client.settings.max_output_tokens)

    def recheck(paths, previous):
        sources = []
        for path in paths:
            check_active()
            sources.append(load_document(path, "source", session_dir / "work", "pdfplumber"))
        check_active()
        bind_company(report_doc, sources, company)
        facts = check_documents(report_doc, sources)
        result = deepcopy(previous)
        result["external_fact_check"] = facts
        result["coverage"]["external_source_documents_used"] = True
        result["coverage"]["external_evidence_note"] = "External fact verification is scored separately; text error labels/statuses were not promoted."
        return result

    workflow = EvidenceWorkflow(detect, recheck, kb_dir=kb_dir, ov_config=ov_config, allow_fallback=allow_fallback)
    runtime = runner({"documentId": "document", "question": "先检测绑定文本，再按需检索、读取来源并复核；文本错误与外部事实分别报告。"},
                     workflow, model_config, settings=runtime_settings, limits=limits)
    (session_dir / "agent_runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    if workflow.result is None:
        raise RuntimeError("Pi did not complete text detection: " + str(runtime.get("stop_reason")))
    result = workflow.result
    result["agent_runtime"] = {**runtime, "workflow_events": workflow.events, "verdict_authority": "python_evidence_rules"}
    if runtime.get("status") in {"failed", "stopped"}:
        result["coverage"]["agent_complete"] = False
    (session_dir / "text_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, session_dir


ROOT_OUTPUT = Path(__file__).resolve().parents[3] / "data/agent-runs"
