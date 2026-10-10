"""Small live-service integration; synthetic fixtures, never a quality benchmark.

prepare: original PDFs -> real B parser -> anchored KB; no inference.
retrieve: actual OpenViking ingest/find/read with local ONNX embeddings.
pi: actual configured model + Pi + live retrieval + C rules, shared ledger.
Only pi calls the configured language-model API. No test gold is sent to tools.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("factcheck/src", "pdfparse/src", "frontend/tools")]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def embedding_health():
    with urllib.request.urlopen("http://127.0.0.1:1934/health", timeout=5) as response:
        return json.load(response)


def binding(root, config):
    if config.base_url.rstrip("/") != "http://127.0.0.1:1933":
        raise ValueError("this_smoke_requires_the_local_1933_service")
    files = [root / "fixtures.json", root / "kb/ingest_manifest.json", root / "kb/kb_index.jsonl",
             root / "kb/openviking_bindings.json", ROOT / "data/local-openviking/ov.conf"]
    files.extend(sorted((root / "kb/markdown").glob("*.md")))
    health = embedding_health()
    return {"endpoint": config.base_url, "target_uri": config.target_uri,
            "files_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "embedding": {key: health[key] for key in ("model", "revision", "dimension", "max_input_tokens")}}


def prepare(root):
    from make_demo_pdfs import _make_pdf, SOURCE_LINES
    from yjparse.config import load_thresholds
    from yjparse.pipeline import parse_document, PipelineConfig
    from yjparse.kb_export import export_kb

    if (root / "fixtures.json").exists():
        raise ValueError("fixtures_already_prepared_use_new_directory")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    sources = [
        ("sample-company", SOURCE_LINES, "示例公司2025年上半年归母净利润", "1.55"),
        ("food-company", ["北辰食品股份有限公司2024年年度报告", "合并利润表", "项目 2024年度", "单位：万元", "营业收入 8800"], "北辰食品2024年营业收入", "8800"),
        ("tech-company", ["星海科技股份有限公司2023年年度报告", "合并利润表", "项目 2023年度", "单位：万元", "研发费用 360"], "星海科技2023年研发费用", "360"),
    ]
    source_checks = []
    config = PipelineConfig(primary="pymupdf", reference="pdfplumber", ocr="off", vlm="off", thresholds=load_thresholds())
    for name, lines, query, value in sources:
        path = inputs / (name + ".pdf")
        _make_pdf(path, lines)
        parsed = parse_document(path, root / "parsed", config)
        source_checks.append({"doc_id": parsed.doc.doc_id, "query": query, "value": value})
    export_kb(root / "parsed", root / "kb", project="pi-local-smoke")
    cases = [
        ("numeric-error", "公司2025年上半年归母净利润为1.50亿元。", "confirmed_error"),
        ("clean", "公司2025年上半年营业收入为12.34亿元。", "no_issue"),
        ("missing-source", "公司2025年上半年存货为9.69亿元。", "needs_review"),
    ]
    documents, expected = [], []
    for index, (label, line, status) in enumerate(cases):
        path = inputs / f"report-{index}.pdf"
        _make_pdf(path, ["示例公司(600000) 2025年半年度点评报告", line])
        documents.append({"id": str(index), "report": str(path.resolve()), "company": "示例公司"})
        expected.append({"id": str(index), "expected_status": status, "case": label})
    save(root / "fixtures.json", {"role": "synthetic_integration", "documents": documents, "source_checks": source_checks})
    save(root / "expected.json", expected)
    return {"status": "prepared", "sources": len(sources), "cases": len(cases), "kb": str(root / "kb")}


def retrieve(root):
    from yjparse.openviking import OpenVikingConfig, health, ingest_kb, search_kb, read_evidence
    fixture = json.loads((root / "fixtures.json").read_text(encoding="utf-8"))
    config = replace(OpenVikingConfig.from_env(), target_uri="viking://resources/pi-local-smoke", timeout=30)
    report = {"role": "synthetic_integration", "health": health(config), "llm_calls": 0, "fallback_allowed": False}
    # Ensure every short exported Markdown actually fits the local model.
    coverage = []
    for path in sorted((root / "kb/markdown").glob("*.md")):
        payload = json.dumps({"model": "bge-small-zh-v1.5-onnx", "input": path.read_text(encoding="utf-8"), "encoding_format": "float"}).encode()
        request = urllib.request.Request("http://127.0.0.1:1934/v1/embeddings", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
        coverage.append({"document": path.name, "usage": body.get("usage"), "truncation": body.get("truncation"),
                         "dimension": len(body["data"][0]["embedding"]), "truncated": body["data"][0].get("truncated")})
    report["embedding_coverage"] = coverage
    if any(c["truncated"] is not False for c in coverage):
        report.update(passed=False, reason="short_fixture_embedding_coverage_not_complete")
        save(root / "retrieval-result.json", report)
        return {"passed": False, "reason": report["reason"]}
    report["embedding_before_ingest"] = embedding_health()
    report["ingest"] = ingest_kb(root / "kb", config=config, processing_timeout=120)
    checks = []
    for check in fixture["source_checks"]:
        found = search_kb(root / "kb", check["query"], top_k=5, config=config, allow_fallback=False)
        hits = found.get("hits", [])
        matching = next((hit for hit in hits if hit.get("doc_id") == check["doc_id"] and check["value"] in hit.get("text", "")), None)
        read = read_evidence(root / "kb", matching["doc_id"], matching["block_id"], uri=matching.get("uri"), config=config) if matching else None
        passed = bool(found.get("openviking_used") and matching and read and read.get("openviking_used")
                      and read.get("evidence", {}).get("verified"))
        checks.append({"query": check["query"], "passed": passed, "search": found, "read": read})
    report["checks"] = checks
    report["embedding_after_retrieval"] = embedding_health()
    untruncated = report["embedding_after_retrieval"]["truncated_inputs"] == report["embedding_before_ingest"]["truncated_inputs"]
    report["passed"] = bool(checks) and report["ingest"].get("status") == "ok" and all(c["passed"] for c in checks) and untruncated
    if report["passed"]:
        report["binding"] = binding(root, config)
    save(root / "retrieval-result.json", report)
    return {"passed": report["passed"], "checks": len(checks), "path": str(root / "retrieval-result.json")}


def pi(root, args):
    from yjcheck.agent_workflow import run_agent_check
    from yjcheck.model import ModelConfig
    from yjcheck.model_runtime import RuntimeSettings, BudgetLedger
    from yjcheck.pi_bridge import AgentLimits
    from yjcheck.pipeline import verify_artifacts
    from yjparse.openviking import OpenVikingConfig
    retrieval = json.loads((root / "retrieval-result.json").read_text(encoding="utf-8"))
    if retrieval.get("passed") is not True:
        raise ValueError("real_retrieval_must_pass_before_pi")
    settings = replace(RuntimeSettings.from_env(), input_cny_per_mtok=args.input_price,
                       output_cny_per_mtok=args.output_price, price_source=args.price_source,
                       price_date=datetime.now(timezone.utc).date().isoformat(), price_valid_until=args.price_valid_until)
    config = ModelConfig.from_env()
    ov = replace(OpenVikingConfig.from_env(), target_uri="viking://resources/pi-local-smoke")
    if retrieval.get("binding") != binding(root, ov):
        raise ValueError("retrieval_gate_stale_or_service_changed")
    ledger = BudgetLedger(settings.ledger, settings.budget_cny)
    before = ledger.summary()
    results = []
    fixture = json.loads((root / "fixtures.json").read_text(encoding="utf-8"))
    for doc in fixture["documents"]:
        started = time.monotonic()
        result, directory = run_agent_check(doc["report"], [], root / "pi-results", company=doc["company"],
            model_config=config, runtime_settings=settings, kb_dir=root / "kb", ov_config=ov,
            limits=AgentLimits(max_rounds=6, max_cost_cny="0.5", timeout_seconds=300, tool_timeout_seconds=60))
        results.append({"id": doc["id"], "summary": result["summary"], "statuses": [f["status"] for f in result["findings"]],
                        "agent_runtime": result["agent_runtime"], "seconds": round(time.monotonic()-started, 3),
                        "directory": str(directory), "artifacts_verified": verify_artifacts(directory)})
        save(root / "pi-progress.json", results)
    # Expected outcomes are opened only after all predictions have been written.
    expected = {r["id"]: r for r in json.loads((root / "expected.json").read_text(encoding="utf-8"))}
    for result in results:
        result["expected_status"] = expected[result["id"]]["expected_status"]
        events = result["agent_runtime"].get("workflow_events", [])
        searched = any(e.get("tool") == "search_evidence" and e.get("provider") == "openviking" and e.get("status") == "ok" for e in events)
        tools_used = [e.get("tool") for e in events]
        chain = (searched and (result["expected_status"] == "needs_review" or
                 all(name in tools_used for name in ("read_evidence", "recheck"))))
        result["live_evidence_chain"] = chain
        result["passed"] = (result["statuses"] == [result["expected_status"]] and result["artifacts_verified"]
                            and chain and result["agent_runtime"]["status"] in {"completed", "needs_review"})
    traces = {trace["call_id"]: trace for result in results for trace in result["agent_runtime"].get("model_traces", [])}
    report = {"role": "synthetic_integration_not_accuracy_benchmark", "before": before, "after": ledger.summary(),
              "run_model_calls": len(traces), "run_accounted_cny": str(sum((Decimal(t["cost_cny"]) for t in traces.values()), Decimal(0))),
              "price": {"input": args.input_price, "output": args.output_price, "source": args.price_source,
                        "valid_until": args.price_valid_until}, "results": results, "passed": all(r["passed"] for r in results)}
    save(root / "pi-result.json", report)
    return {"passed": report["passed"], "cases": len(results), "path": str(root / "pi-result.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "retrieve", "pi"))
    parser.add_argument("--root", type=Path, default=ROOT / "data/local-openviking/smoke")
    parser.add_argument("--input-price")
    parser.add_argument("--output-price")
    parser.add_argument("--price-source")
    parser.add_argument("--price-valid-until")
    args = parser.parse_args()
    if args.stage == "pi" and not all((args.input_price, args.output_price, args.price_source, args.price_valid_until)):
        parser.error("pi requires freshly verified price fields; no paid request was made")
    result = prepare(args.root.resolve()) if args.stage == "prepare" else retrieve(args.root.resolve()) if args.stage == "retrieve" else pi(args.root.resolve(), args)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
