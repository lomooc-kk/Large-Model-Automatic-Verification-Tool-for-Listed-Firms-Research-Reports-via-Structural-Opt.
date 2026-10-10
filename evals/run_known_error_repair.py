"""Incremental repair of known historical FP/FN events, never a new benchmark.

plan: choose small batches of old-error documents, preserve their full input.
run: execute the real hybrid node; rules-only unless the frozen plan AND command
     explicitly select --live. Live calls reuse the existing shared budget.
analyze: read frozen gold/baseline offline and retain ALL old FP/FN events in the
         repair denominator, including unrun, failed and unresolved events.

The 95% target concerns known-event repair, not subset F1 or new-data accuracy.
No gold, baseline prediction or example is passed to detect_text/model messages.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "factcheck/src"), str(ROOT / "pdfparse/src")]
from evals.run_v2 import SharedCandidateClient, assert_input_only, import_model_responses, runtime_calls, validate_predictions, write_json
from yjcheck.model import ModelConfig
from yjcheck.model_runtime import BudgetedChatClient, DEFAULT_LEDGER, RuntimeSettings
from yjcheck.review_hints import all_review_hints
from yjcheck.text_review import _parse_response, detect_text

PACKAGE = ROOT / "artifacts/research442-rerun-20261005"
PUBLISHED_MANIFEST_SHA256 = "9836059fa58107d88ae6cffc680c1409aa48dc0bc874ebad8a28cb7a4f2640e6"
PUBLISHED_COUNTS = (442, 1747, 45)
PUBLISHED_BASELINE_COUNTS = (1235, 625, 512)
SCHEMA = "known-error-repair/1.0"
ROLE = "diagnostic_regression"
REDUNDANCY_REVIEW_PURPOSE = "text_review.redundancy_review"
REDUNDANCY_REVIEW_POLICIES = ("actions_v1", "context_v2")
INPUT = "inputs/inputs.research442.jsonl"
GOLD = "inputs/gold.research442.jsonl"
FROZEN_SCORER = "source/evals/fined_bench_eval.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(row):
    return row.get("document_id", row.get("doc_id"))


def index(records):
    result = {}
    for row in records:
        sid = identity(row)
        if not isinstance(sid, str) or not sid or sid in result:
            raise ValueError("missing_or_duplicate_document_id")
        result[sid] = row
    return result


def load_scorer(package):
    spec = importlib.util.spec_from_file_location("known_error_frozen_scorer", package / FROZEN_SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases_for_document(scorer, document_id, prediction, gold):
    preds = prediction.get("errors", [])
    targets = [g for g in gold["errors"] if g.get("scorable", True)]
    pairs = scorer._maximum_matching(preds, targets, lambda p, g: scorer._contains_error(p, g)
             and p.get("error_type", p.get("type")) == g.get("type", g.get("error_type")))
    matched_p, matched_g = {id(p) for p, _ in pairs}, {id(g) for _, g in pairs}
    pi = {id(p): i for i, p in enumerate(preds)}
    gi = {id(g): i for i, g in enumerate(gold["errors"])}
    entries = [("TP", p, g) for p, g in pairs]
    entries += [("FP", p, None) for p in preds if id(p) not in matched_p]
    entries += [("FN", None, g) for g in targets if id(g) not in matched_g]
    return [{"document_id": document_id, "result": label,
             "prediction_index": pi[id(p)] if p is not None else None,
             "gold_index": gi[id(g)] if g is not None else None,
             "prediction": p, "gold": g} for label, p, g in entries]


def baseline_data(package):
    inputs, golds = index(rows(package / INPUT)), index(rows(package / GOLD))
    reports = index([read(path) for path in sorted((package / "run/predictions/hybrid").glob("*.json"))])
    if inputs.keys() != golds.keys() or inputs.keys() != reports.keys():
        raise ValueError("frozen_input_gold_baseline_alignment_failed")
    scorer = load_scorer(package)
    hints = {sid: all_review_hints(validate_predictions(deepcopy(report), inputs[sid]["content"]))
             for sid, report in reports.items()}
    events = []
    for sid in sorted(inputs):
        for case in cases_for_document(scorer, sid, hints[sid], golds[sid]):
            if case["result"] != "TP":
                offset = case["gold_index"] if case["result"] == "FN" else case["prediction_index"]
                events.append({**case, "event_id": f"{case['result']}:{sid}:{offset}"})
    return inputs, golds, reports, hints, events, scorer


def audit_package(package):
    """Offline admission process; original 442 gold and baseline stay unchanged."""
    package = Path(package).resolve()
    if sha(package / "manifest.json") != PUBLISHED_MANIFEST_SHA256:
        raise ValueError("published_442_manifest_changed")
    manifest = read(package / "manifest.json")
    entries = {entry["path"]: entry for entry in manifest}
    if len(entries) != len(manifest):
        raise ValueError("duplicate_package_manifest_path")
    required = {INPUT, GOLD, FROZEN_SCORER}
    required.update(path for path in entries if path.startswith("run/predictions/hybrid/"))
    for relative in sorted(required):
        path = (package / relative).resolve()
        if not path.is_relative_to(package) or relative not in entries or sha(path) != entries[relative]["sha256"]:
            raise ValueError("frozen_package_file_hash_mismatch")
    inputs, golds, reports, hints, events, scorer = baseline_data(package)
    for row in inputs.values():
        assert_input_only(row)
    scorable = sum(bool(e.get("scorable", True)) for g in golds.values() for e in g["errors"])
    excluded = sum(not e.get("scorable", True) for g in golds.values() for e in g["errors"])
    if (len(inputs), scorable, excluded) != PUBLISHED_COUNTS:
        raise ValueError("frozen_442_denominator_changed")
    strict = scorer.score_paper_detection(hints.values(), golds.values())
    if tuple(strict[key] for key in ("true_positive", "false_positive", "false_negative")) != PUBLISHED_BASELINE_COUNTS:
        raise ValueError("frozen_baseline_scoring_drift")
    return {"schema_version": SCHEMA, "role": ROLE, "package": str(package),
            "manifest_sha256": sha(package / "manifest.json"), "input_sha256": sha(package / INPUT),
            "gold_sha256": sha(package / GOLD), "scorer_sha256": sha(package / FROZEN_SCORER),
            "documents": len(inputs), "scorable_gold_errors": scorable, "excluded_gold_errors": excluded,
            "baseline_strict_all_hints": strict,
            "old_event_count": len(events), "old_event_counts": dict(Counter(e["result"] for e in events)),
            "known_events": [{"event_id": e["event_id"], "document_id": e["document_id"], "result": e["result"],
                              "error_type": (e["prediction"] or e["gold"]).get("error_type", (e["prediction"] or e["gold"]).get("type"))}
                             for e in events],
            "event_denominator": "Every frozen FP and FN is a separate event; a wrong-type detection may produce both, without deduplication",
            "old_dataset_in_new_suite": False, "weight_training": False}


def admission(package):
    process = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "audit", "--package", str(package)],
                             capture_output=True, text=True, encoding="utf-8")
    if process.returncode:
        raise ValueError("frozen_442_admission_failed")
    value = json.loads(process.stdout)
    if value.get("role") != ROLE or value.get("manifest_sha256") != PUBLISHED_MANIFEST_SHA256:
        raise ValueError("unexpected_admission_receipt")
    return value


def source_hashes():
    paths = sorted((ROOT / "factcheck/src/yjcheck").glob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "evals/run_v2.py"]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def redundancy_review_options(spec):
    """Read explicit experiments without changing historical default plans."""
    policy = spec.get("redundancy_review_policy", "actions_v1")
    include_reason = spec.get("redundancy_review_include_reason", True)
    if not isinstance(policy, str) or policy not in REDUNDANCY_REVIEW_POLICIES:
        raise ValueError("invalid_redundancy_review_policy")
    if type(include_reason) is not bool:
        raise ValueError("invalid_redundancy_review_include_reason")
    if (policy != "actions_v1" or not include_reason) and spec.get("redundancy_review") is not True:
        raise ValueError("nondefault_review_options_require_redundancy_review")
    return policy, include_reason


def candidate_version(spec):
    identity = {key: spec[key] for key in
                ("source_hashes", "runtime", "detector", "examples", "max_input_tokens")}
    # Historical plans predate this option. Their frozen identity must remain
    # readable; newly created plans bind both explicit false and true values.
    if "redundancy_review" in spec:
        if type(spec["redundancy_review"]) is not bool:
            raise ValueError("invalid_redundancy_review_flag")
        identity["redundancy_review"] = spec["redundancy_review"]
    redundancy_review_options(spec)
    for key in ("redundancy_review_policy", "redundancy_review_include_reason"):
        if key in spec:
            identity[key] = spec[key]
    return fingerprint(identity)


def shared_budget(settings):
    """Read an existing ledger without constructing or increasing any budget."""
    path = Path(settings.ledger).resolve()
    if path != DEFAULT_LEDGER.resolve() or not path.is_file():
        raise ValueError("existing_shared_budget_ledger_required")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        budget = db.execute("SELECT limit_micro FROM budget WHERE singleton=1").fetchone()
        used, count = db.execute("SELECT COALESCE(SUM(charge_micro),0),COUNT(*) FROM calls").fetchone()
    if (not budget or not 0 < budget[0] <= 100_000_000
            or Decimal(settings.budget_cny) * 1_000_000 != budget[0]):
        raise ValueError("shared_budget_cap_mismatch_or_over_100_cny")
    return {"ledger": str(path), "budget_cny": str(Decimal(budget[0]) / 1_000_000),
            "accounted_cny": str(Decimal(used) / 1_000_000), "calls": count}


def runtime_spec(live, max_output_tokens=16384):
    if not live:
        return {"mode": "rules_only", "model": None, "runtime": None}, None, None
    config = ModelConfig.from_env()
    settings = replace(RuntimeSettings.from_env(), max_retries=0, max_output_tokens=max_output_tokens)
    if not 0 < settings.max_output_tokens < settings.context_tokens:
        raise ValueError("invalid_round_output_token_limit")
    budget = shared_budget(settings)
    runtime = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()}
    public = {"mode": "live", "model": config.model, "endpoint_sha256": hashlib.sha256(config.base_url.encode()).hexdigest(),
              "thinking": config.thinking, "reasoning_effort": config.reasoning_effort, "timeout": config.timeout,
              "runtime": runtime, "ledger": budget["ledger"], "budget_cny": budget["budget_cny"]}
    return public, config, settings


def resolve_document_ids(requested, eligible_ids):
    """Accept complete IDs or unique last-component IDs, never fuzzy matches."""
    selected = []
    for item in requested:
        for token in item.split(","):
            token = token.strip()
            candidates = [token] if token in eligible_ids else sorted(sid for sid in eligible_ids if sid.rsplit(":", 1)[-1] == token)
            if len(candidates) != 1:
                raise ValueError("unknown_or_ambiguous_known_error_document_id:" + token)
            if candidates[0] in selected:
                raise ValueError("duplicate_selected_document_id")
            selected.append(candidates[0])
    return selected


def create_plan(args):
    package, out = args.package.resolve(), args.out.resolve()
    redundancy_review = getattr(args, "redundancy_review", False)
    if type(redundancy_review) is not bool:
        raise ValueError("invalid_redundancy_review_flag")
    hide_reason = getattr(args, "redundancy_review_hide_reason", False)
    if type(hide_reason) is not bool:
        raise ValueError("invalid_redundancy_review_hide_reason")
    review_policy, include_reason = redundancy_review_options({
        "redundancy_review": redundancy_review,
        "redundancy_review_policy": getattr(args, "redundancy_review_policy", "actions_v1"),
        "redundancy_review_include_reason": not hide_reason})
    if out.exists() or out.is_relative_to(package) or out.is_relative_to(ROOT / "data/active_suite_v1"):
        raise ValueError("fresh_output_outside_frozen_and_new_datasets_required")
    audit = admission(package)
    eligible = audit["known_events"]
    if args.error_type:
        eligible = [e for e in eligible if e["error_type"] in set(args.error_type)]
        if set(args.error_type) - {e["error_type"] for e in audit["known_events"]}:
            raise ValueError("unknown_old_error_type")
    known_ids = {e["document_id"] for e in eligible}
    requested = args.document_id + getattr(args, "doc_ids", [])
    if requested:
        selected_ids = resolve_document_ids(requested, known_ids)
    else:
        selected_ids = sorted(known_ids)
        if not args.all_known_errors:
            selected_ids = selected_ids[:args.limit]
    if not selected_ids:
        raise ValueError("no_known_error_documents_selected")
    inputs = index(rows(package / INPUT))
    selected = [{"document_id": sid, "content": inputs[sid]["content"], "scene": inputs[sid].get("scene", "")}
                for sid in selected_ids]
    for row in selected:
        assert_input_only(row)
    runtime, _, settings = runtime_spec(args.live, getattr(args, "max_output_tokens", 16384))
    reuse_sources = getattr(args, "reuse_model_responses_from", []) or []
    replay_only = bool(getattr(args, "replay_only", False))
    cached_detection_only = getattr(args, "cached_detection_only", False)
    if type(cached_detection_only) is not bool:
        raise ValueError("invalid_cached_detection_only_flag")
    if cached_detection_only and not (redundancy_review and args.live and reuse_sources):
        raise ValueError("cached_detection_only_requires_live_redundancy_review_and_response_sources")
    if replay_only and not reuse_sources:
        raise ValueError("replay_only_requires_explicit_response_source")
    manifests, reusable = prepare_known_response_reuse(reuse_sources, runtime, out,
        settings.context_tokens - settings.max_output_tokens if settings else 16000)
    codes = source_hashes()
    out.mkdir(parents=True)
    queue_path = out / "inputs.selected.jsonl"
    queue_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    for relative, expected in codes.items():
        raw = (ROOT / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("source_changed_while_freezing_plan")
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    spec = {"schema_version": SCHEMA, "role": ROLE, "package": str(package),
            "source_manifest_sha256": audit["manifest_sha256"], "source_input_sha256": audit["input_sha256"],
            "source_gold_sha256": audit["gold_sha256"], "frozen_scorer_sha256": audit["scorer_sha256"],
            "document_ids": selected_ids, "selected_input_sha256": sha(queue_path), "source_hashes": codes,
            "runtime": runtime, "configuration_sha256": fingerprint(runtime), "workers": 1,
            "detector": "hybrid", "examples": [], "gold_in_model_prompt": False,
            "redundancy_review": redundancy_review,
            "cached_detection_only": cached_detection_only,
            "max_input_tokens": settings.context_tokens - settings.max_output_tokens if settings else 16000,
            "selection_error_types": args.error_type or [], "total_old_events": audit["old_event_count"],
            "old_event_counts": audit["old_event_counts"], "old_event_denominator": audit["event_denominator"],
            "total_source_documents": audit["documents"], "scorable_gold_errors": audit["scorable_gold_errors"],
            "excluded_gold_errors": audit["excluded_gold_errors"], "old_dataset_in_new_suite": False,
            "weight_training": False, "goal_metric": "benchmark_event_recovery",
            "business_repair_status": "pending_adjudication; disappearance of an unmatched hint is not proof of business correctness",
            "resume_policy": "same fingerprint only; completed attempts including failures are not silently retried; exact-request model response cache is retained"}
    # Omitting defaults preserves the established plan/version identity shape.
    if review_policy != "actions_v1":
        spec["redundancy_review_policy"] = review_policy
    if not include_reason:
        spec["redundancy_review_include_reason"] = False
    if manifests:
        spec["reused_model_response_sources"] = manifests
        spec["imported_response_hashes"] = {key: fingerprint(value) for key, value in reusable.items()}
        spec["replay_only"] = replay_only
        spec["response_reuse_policy"] = "Only successful raw responses, identical runtime and exact request/purpose; no predictions or gold. Provenance retained; replay is not a fresh model call."
    if cached_detection_only:
        spec["cache_miss_policy"] = "Only text_review.redundancy_review may make a new request; detect/global must replay an exact imported response. replay_only, when set, forbids every cache miss."
    spec["candidate_version"] = candidate_version(spec)
    plan = {"spec": spec, "fingerprint": fingerprint(spec), "created_at": datetime.now(timezone.utc).isoformat(),
            "budget_at_plan": shared_budget(settings) if settings else None}
    import_model_responses(out / "model_responses", reusable)
    write_json(out / "plan.json", plan)
    print(json.dumps({"planned_documents": len(selected), "total_old_events": audit["old_event_count"],
                      "mode": runtime["mode"], "role": ROLE, "paid_calls": 0, "fingerprint": plan["fingerprint"]}))
    return plan


def load_plan(out):
    plan = read(out / "plan.json")
    spec = plan["spec"]
    if spec.get("schema_version") != SCHEMA or spec.get("role") != ROLE or fingerprint(spec) != plan["fingerprint"]:
        raise ValueError("plan_fingerprint_or_role_changed")
    if spec.get("candidate_version") != candidate_version(spec):
        raise ValueError("candidate_version_changed")
    if "cached_detection_only" in spec and type(spec["cached_detection_only"]) is not bool:
        raise ValueError("invalid_cached_detection_only_flag")
    if spec.get("cached_detection_only") and not (spec.get("redundancy_review") is True
            and spec["runtime"]["mode"] == "live" and spec.get("reused_model_response_sources")):
        raise ValueError("cached_detection_only_requires_live_redundancy_review_and_response_sources")
    if sha(out / "inputs.selected.jsonl") != spec["selected_input_sha256"]:
        raise ValueError("selected_input_changed")
    for relative, expected in spec["source_hashes"].items():
        path = (out / "source" / relative).resolve()
        if not path.is_relative_to((out / "source").resolve()) or sha(path) != expected:
            raise ValueError("frozen_source_snapshot_changed")
    return plan


def response_runtime_identity(runtime):
    """Price verification timestamps/URL never enter the inference request.

    Keep model, endpoint, reasoning, timeout, token limits, rates, budget and
    every unknown field strict. Historical trace prices/costs stay untouched.
    New plans and actual calls still validate their current full runtime.
    """
    identity = deepcopy(runtime)
    if not isinstance(identity, dict) or not isinstance(identity.get("runtime"), dict):
        raise ValueError("response_source_model_runtime_mismatch")
    for key in ("price_source", "price_date", "price_valid_until"):
        identity["runtime"].pop(key, None)
    return identity


def failed_context_review_raw_receipt(saved, spec, state):
    """Authenticate only completed API bytes, never credit a failed pipeline.

    This narrow exception is for two known v2 protocol rejections. It does not
    change complete_live_record, final findings, gold, scoring or old statuses.
    """
    report = saved.get("report", {})
    stage = report.get("redundancy_review")
    if not isinstance(stage, dict):
        return False
    policy, include_reason = redundancy_review_options(spec)
    coverage, base = report.get("coverage"), stage.get("base_coverage")
    failure = stage.get("failure")
    if (saved.get("mode") != "live" or saved.get("code_unchanged") is not True
            or state.get("all_selected_attempted") is not True or state.get("stop_reason")
            or set(state.get("reports", {})) != set(spec["document_ids"])
            or policy != "context_v2" or spec.get("redundancy_review") is not True
            or stage.get("schema_version") not in {"redundancy-review/2.0", "redundancy-review/2.1"}
            or stage.get("policy") != policy or type(stage.get("include_reason")) is not bool
            or stage["include_reason"] != include_reason
            or stage.get("purpose") != REDUNDANCY_REVIEW_PURPOSE
            or stage.get("source_content_sha256") != saved.get("source_content_sha256")
            or stage.get("status") != "incomplete" or stage.get("complete") is not False
            or stage.get("model_ran") is not True or stage.get("base_execution_complete") is not True
            or not isinstance(base, dict) or base.get("execution_complete") is not True
            or base.get("model_complete") is not True
            or not isinstance(coverage, dict)
            or any(coverage.get(k) is not False for k in ("execution_complete", "complete", "model_complete"))
            or stage.get("decisions") != [] or stage.get("withdrawn") != []
            or not isinstance(failure, dict) or failure.get("error_type") != "_Invalid"
            or failure.get("code") not in {"invalid_source_evidence", "invalid_normal_context"}):
        return False
    jobs = report.get("traces")
    if not isinstance(jobs, list) or not all(isinstance(job, dict) for job in jobs):
        return False
    review = [job for job in jobs if job.get("purpose") == REDUNDANCY_REVIEW_PURPOSE]
    base_jobs = [job for job in jobs if job.get("purpose") != REDUNDANCY_REVIEW_PURPOSE]
    if len(review) != 1 or not base_jobs:
        return False
    seen = set()
    for job in base_jobs:
        trace = job.get("runtime_trace")
        index = job.get("job_index")
        if (job.get("status") != "ok" or type(index) is not int or index < 0 or index in seen
                or job.get("purpose") not in {"text_review.detect", "text_review.global"}
                or not isinstance(trace, dict) or trace.get("status") != "ok"
                or trace.get("purpose") != job["purpose"]
                or trace.get("finish_reason") == "length" or trace.get("response_content_incomplete")):
            return False
        seen.add(index)
    job, trace = review[0], review[0].get("runtime_trace")
    raw = stage.get("raw_response")
    if (job.get("status") != "failed" or job.get("job_index") != max(seen) + 1
            or job.get("failure_code") != failure["code"]
            or not isinstance(trace, dict) or trace.get("status") != "ok"
            or trace.get("finish_reason") != "stop" or trace.get("response_content_incomplete")
            or not isinstance(trace.get("call_id"), str) or not trace["call_id"].strip()
            or trace.get("purpose") != REDUNDANCY_REVIEW_PURPOSE
            or not re.fullmatch(r"[0-9a-f]{64}", str(trace.get("request_sha256", "")))
            or stage.get("request_sha256") != trace["request_sha256"]
            or stage.get("runtime_trace") != trace
            or not isinstance(raw, str) or not raw.strip()
            or stage.get("response_sha256") != hashlib.sha256(raw.encode()).hexdigest()):
        return False
    try:
        cost = Decimal(str(trace.get("cost_cny")))
    except (ArithmeticError, TypeError, ValueError):
        return False
    return cost.is_finite() and cost >= 0


def completed_response_receipts(source, prior):
    """Independent historical anchor: state-frozen raw candidates and call trace.

    Older runs did not store the full response bytes in their receipts. This
    therefore binds parsed errors and the complete runtime trace, not arbitrary
    JSON formatting or unused top-level response fields. The independent review
    has a different schema and saves exact response text in its stage audit.
    Final findings and gold never select or validate a response.
    """
    state_path = source / "state.json"
    if state_path.resolve().parent != source or not state_path.is_file():
        raise ValueError("response_source_completed_receipt_required")
    state = read(state_path)
    if state.get("fingerprint") != prior["fingerprint"] or not isinstance(state.get("reports"), dict):
        raise ValueError("response_source_state_mismatch")
    selected = index(rows(source / "inputs.selected.jsonl"))
    if list(selected) != prior["spec"]["document_ids"]:
        raise ValueError("response_source_selected_queue_mismatch")
    receipts = {}
    for sid, entry in state["reports"].items():
        relative = "predictions/" + hashlib.sha256(sid.encode()).hexdigest()[:24] + ".json"
        path = (source / relative).resolve()
        if (sid not in selected or not isinstance(entry, dict) or entry.get("path") != relative
                or not path.is_relative_to(source) or not path.is_file() or sha(path) != entry.get("sha256")):
            raise ValueError("response_source_completed_receipt_changed")
        saved = read(path)
        report = saved.get("report", {})
        if (saved.get("schema_version") != SCHEMA or saved.get("document_id") != sid
                or saved.get("fingerprint") != prior["fingerprint"]
                or saved.get("candidate_version") != prior["spec"]["candidate_version"]
                or saved.get("source_content_sha256") != hashlib.sha256(selected[sid]["content"].encode()).hexdigest()
                or report.get("document_id") != sid):
            raise ValueError("response_source_completed_receipt_mismatch")
        pipeline_complete = complete_live_record(saved)
        protocol_failed_raw_only = (not pipeline_complete
                                    and failed_context_review_raw_receipt(saved, prior["spec"], state))
        if not pipeline_complete and not protocol_failed_raw_only:
            continue
        raw = report.get("raw_candidates")
        jobs = report.get("traces")
        if not isinstance(raw, list) or not isinstance(jobs, list):
            raise ValueError("response_source_raw_receipt_required")
        seen_jobs = set()
        review_job_seen = False
        for job in jobs:
            failed_review_job = (protocol_failed_raw_only and job.get("purpose") == REDUNDANCY_REVIEW_PURPOSE)
            # The failed pipeline exception authenticates its review bytes only.
            # Its base calls require their own prior complete/locked receipts.
            if protocol_failed_raw_only and not failed_review_job:
                continue
            if job.get("status") != "ok" and not failed_review_job:
                continue
            job_index, trace = job.get("job_index"), job.get("runtime_trace")
            if (type(job_index) is not int or job_index < 0 or job_index in seen_jobs
                    or not isinstance(trace, dict) or trace.get("status") != "ok"
                    or job.get("purpose") != trace.get("purpose")):
                raise ValueError("response_source_raw_receipt_mismatch")
            seen_jobs.add(job_index)
            key = (trace.get("call_id"), trace.get("request_sha256"), trace.get("purpose"))
            proof = {"basis": "completed_receipt_raw_candidates_and_trace",
                     "response_bytes_authenticated": False, "state_sha256": sha(state_path),
                     "receipt_path": relative, "receipt_sha256": entry["sha256"],
                     "document_id": sid, "job_index": job_index}
            if trace.get("purpose") == REDUNDANCY_REVIEW_PURPOSE:
                stage = report.get("redundancy_review")
                policy, include_reason = redundancy_review_options(prior["spec"])
                expected_schemas = ({"redundancy-review/2.0", "redundancy-review/2.1"}
                                    if policy == "context_v2" else {"redundancy-review/1.0"})
                if (review_job_seen or prior["spec"].get("redundancy_review") is not True
                        or not isinstance(stage, dict)
                        or stage.get("schema_version") not in expected_schemas
                        or stage.get("policy", "actions_v1") != policy
                        or type(stage.get("include_reason", True)) is not bool
                        or stage.get("include_reason", True) != include_reason
                        or (policy == "context_v2" and not {"policy", "include_reason"} <= stage.keys())
                        or stage.get("purpose") != REDUNDANCY_REVIEW_PURPOSE
                        or (not failed_review_job and (stage.get("status") != "complete" or stage.get("complete") is not True))
                        or not isinstance(stage.get("raw_response"), str) or not stage["raw_response"]
                        or stage.get("response_sha256") != hashlib.sha256(stage["raw_response"].encode()).hexdigest()
                        or stage.get("runtime_trace") != trace):
                    raise ValueError("response_source_redundancy_receipt_mismatch")
                review_job_seen = True
                if failed_review_job:
                    proof.update(source_pipeline_complete=False, authentication_scope="raw-only-authenticated",
                                 source_review_status="incomplete", source_failure=deepcopy(stage["failure"]),
                                 source_base_execution_complete=True, source_api_complete=True)
                receipts.setdefault(key, []).append({"content": stage["raw_response"], "trace": trace,
                    "proof": {**proof, "basis": ("failed_context_protocol_api_response_and_trace" if failed_review_job
                                                  else "completed_receipt_redundancy_response_and_trace"),
                              "response_bytes_authenticated": True,
                              "response_sha256": stage["response_sha256"]}})
                continue
            candidates = [item for item in raw if item.get("job_index") == job_index]
            if any(not isinstance(item.get("candidate"), dict)
                   or item.get("candidate_ref") != f"model:{job_index}:{i}"
                   for i, item in enumerate(candidates)):
                raise ValueError("response_source_raw_receipt_mismatch")
            receipts.setdefault(key, []).append({"errors": [item["candidate"] for item in candidates],
                                                "trace": trace, "proof": proof})
    return receipts


def response_receipt_proof(record, receipts):
    # An imported record needs its source plan's locked hash. Do not let an
    # untracked provenance chain fall back to its own replay receipt.
    trace = record["trace"]
    if record.get("cache_provenance") or trace.get("reused_from_prior_run") or trace.get("cache_provenance"):
        raise ValueError("response_source_imported_manifest_required")
    is_review = trace.get("purpose") == REDUNDANCY_REVIEW_PURPOSE
    if is_review:
        # A decisions response must never be interpreted as an errors response.
        # Authentication is exact text plus the independently frozen full trace.
        parsed_trace = trace
    else:
        try:
            errors, parsed_trace = _parse_response(record)
        except (ValueError, TypeError) as exc:
            raise ValueError("response_source_raw_receipt_mismatch") from exc
    key = (trace["call_id"], trace["request_sha256"], trace["purpose"])
    for receipt in receipts.get(key, []):
        expected_trace = deepcopy(receipt["trace"])
        # A same-run cache hit adds this local bookkeeping flag only; original
        # rates, token counts, cost and all unknown runtime fields remain exact.
        if expected_trace.get("reused_response") is True and "reused_response" not in parsed_trace:
            expected_trace.pop("reused_response")
        content_matches = ("content" in receipt and record.get("content") == receipt["content"]) if is_review else (
            "errors" in receipt and errors == receipt["errors"])
        if content_matches and parsed_trace == expected_trace:
            return receipt["proof"]
    raise ValueError("response_source_raw_receipt_mismatch")


def prepare_known_response_reuse(directories, runtime, destination, max_input_tokens):
    """Import raw responses bound to a frozen plan hash or completed raw receipt."""
    if directories and runtime["mode"] != "live":
        raise ValueError("response_reuse_requires_live_model_configuration")
    records, manifests, seen = {}, [], set()
    for directory in directories:
        source = Path(directory).resolve()
        if source == Path(destination).resolve():
            raise ValueError("response_source_must_be_another_run")
        if source in seen:
            continue
        seen.add(source)
        prior = load_plan(source)
        if (response_runtime_identity(prior["spec"]["runtime"]) != response_runtime_identity(runtime)
                or prior["spec"]["max_input_tokens"] != max_input_tokens):
            raise ValueError("response_source_model_runtime_mismatch")
        cache = source / "model_responses"
        if cache.resolve().parent != source or not cache.is_dir():
            raise ValueError("invalid_response_cache_directory")
        locked = prior["spec"].get("imported_response_hashes", {})
        if not isinstance(locked, dict):
            raise ValueError("response_source_imported_manifest_invalid")
        for key, expected in locked.items():
            if (not re.fullmatch(r"[0-9a-f]{64}", str(key))
                    or not re.fullmatch(r"[0-9a-f]{64}", str(expected))):
                raise ValueError("response_source_imported_manifest_invalid")
            path = cache / (key + ".json")
            if (path.resolve().parent != cache.resolve() or not path.is_file()
                    or fingerprint(read(path)) != expected):
                raise ValueError("response_source_imported_record_changed")
        provenance = {"source_run": str(source), "source_plan_sha256": sha(source / "plan.json"),
                      "source_fingerprint": prior["fingerprint"], "source_candidate_version": prior["spec"]["candidate_version"],
                      "source_runtime_sha256": fingerprint(prior["spec"]["runtime"]),
                      "destination_runtime_sha256": fingerprint(runtime),
                      "price_verification_metadata_changed": prior["spec"]["runtime"] != runtime,
                      "inference_runtime_identity_sha256": fingerprint(response_runtime_identity(runtime))}
        hashes, skipped, receipts = {}, 0, None
        for path in sorted(cache.glob("*.json")):
            if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name) or path.resolve().parent != cache.resolve():
                raise ValueError("invalid_response_cache_path")
            record = read(path)
            if path.stem in locked and fingerprint(record) != locked[path.stem]:
                raise ValueError("response_source_imported_record_changed")
            trace = record.get("trace", {})
            if record.get("failed") or trace.get("status") in {"error", "reserved"}:
                skipped += 1
                continue
            if (trace.get("status") != "ok" or not isinstance(record.get("content"), str)
                    or not isinstance(trace.get("call_id"), str) or not trace["call_id"]
                    or not re.fullmatch(r"[0-9a-f]{64}", str(trace.get("request_sha256", "")))
                    or not isinstance(trace.get("purpose"), str) or not trace["purpose"]
                    or trace.get("model") != runtime["model"]
                    or trace.get("thinking", "") != runtime.get("thinking", "")
                    or trace.get("reasoning_effort", "") != runtime.get("reasoning_effort", "")
                    or trace.get("max_output_tokens") != runtime["runtime"]["max_output_tokens"]):
                raise ValueError("response_trace_does_not_match_source_plan")
            request = {"key": path.stem, "purpose": trace["purpose"], "messages_sha256": trace["request_sha256"]}
            if record.get("cache_request") != request:
                raise ValueError("response_request_metadata_mismatch")
            if path.stem in locked:
                proof = {"basis": "source_plan_imported_response_hash", "record_sha256": locked[path.stem]}
            else:
                if receipts is None:
                    receipts = completed_response_receipts(source, prior)
                proof = response_receipt_proof(record, receipts)
            imported = {"content": record["content"], "trace": trace, "cache_request": request,
                        "cache_provenance": {**provenance, "source_response_sha256": sha(path),
                                             "source_response_integrity": proof,
                                             "prior_provenance": record.get("cache_provenance")}}
            if path.stem in records and any(records[path.stem][key] != imported[key] for key in ("content", "trace", "cache_request")):
                raise ValueError("conflicting_responses_for_identical_request")
            records.setdefault(path.stem, imported)
            hashes[path.name] = sha(path)
        manifests.append({**provenance, "successful_records": len(hashes), "failed_records_skipped": skipped,
                          "successful_response_hashes": hashes})
    return manifests, records


def refuse_uncached_request(messages, *, purpose="review"):
    raise RuntimeError("replay_only_cache_miss_no_network_request_permitted")


def review_only_uncached_client(client):
    """Place below the shared cache, so only a new review can spend money."""
    def request(messages, *, purpose="review"):
        if purpose != REDUNDANCY_REVIEW_PURPOSE:
            raise RuntimeError("cached_detection_only_cache_miss_no_network_request_permitted")
        return client(messages, purpose=purpose)
    return request


def run_batch(args):
    out = args.out.resolve()
    plan = load_plan(out)
    spec = plan["spec"]
    review_policy, include_reason = redundancy_review_options(spec)
    live = spec["runtime"]["mode"] == "live"
    if live != bool(args.live):
        raise ValueError("live_requires_both_frozen_plan_and_explicit_run_flag")
    runtime, config, settings = runtime_spec(live, spec["runtime"]["runtime"]["max_output_tokens"] if live else 16384)
    if runtime != spec["runtime"] or source_hashes() != spec["source_hashes"]:
        raise ValueError("runtime_or_source_changed_create_new_plan")
    audit = admission(Path(spec["package"]))
    if any(audit[key] != spec[other] for key, other in (("manifest_sha256", "source_manifest_sha256"),
            ("input_sha256", "source_input_sha256"), ("gold_sha256", "source_gold_sha256"), ("scorer_sha256", "frozen_scorer_sha256"))):
        raise ValueError("frozen_package_drift")
    incoming = rows(out / "inputs.selected.jsonl")
    if [identity(row) for row in incoming] != spec["document_ids"]:
        raise ValueError("selected_queue_changed")
    for row in incoming:
        assert_input_only(row)
    for key, expected in spec.get("imported_response_hashes", {}).items():
        if fingerprint(read(out / "model_responses" / (key + ".json"))) != expected:
            raise ValueError("imported_response_changed_after_plan")
    raw_client = (refuse_uncached_request if spec.get("replay_only") else BudgetedChatClient(config, settings)) if live else None
    if live and spec.get("cached_detection_only") and not spec.get("replay_only"):
        raw_client = review_only_uncached_client(raw_client)
    client = SharedCandidateClient(raw_client, out / "model_responses") if live else None
    state_path = out / "state.json"
    state = read(state_path) if state_path.exists() else {"fingerprint": plan["fingerprint"], "reports": {}, "stop_reason": None}
    if state.get("fingerprint") != plan["fingerprint"]:
        raise ValueError("state_belongs_to_different_run")
    for row in incoming:
        sid = identity(row)
        relative = "predictions/" + hashlib.sha256(sid.encode()).hexdigest()[:24] + ".json"
        path = out / relative
        if sid in state["reports"]:
            if sha(path) != state["reports"][sid]["sha256"]:
                raise ValueError("completed_prediction_changed")
            continue
        if source_hashes() != spec["source_hashes"]:
            state["stop_reason"] = "source_changed_during_run"
            break
        started = time.monotonic()
        try:
            report = detect_text(row["content"], document_id=sid, scene=row.get("scene", ""),
                                 detector="hybrid", chat=client, examples=(), max_input_tokens=spec["max_input_tokens"],
                                 redundancy_review=spec.get("redundancy_review", False),
                                 redundancy_review_policy=review_policy,
                                 redundancy_review_include_reason=include_reason)
            report = validate_predictions(report, row["content"])
        except Exception as exc:
            report = {"document_id": sid, "detector": "hybrid", "errors": [], "traces": [],
                      "coverage": {"complete": False, "execution_complete": False, "rules_complete": False,
                                   "failure_type": type(exc).__name__}}
        stable = source_hashes() == spec["source_hashes"]
        record = {"schema_version": SCHEMA, "document_id": sid, "fingerprint": plan["fingerprint"],
                  "candidate_version": spec["candidate_version"],
                  "source_content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
                  "mode": spec["runtime"]["mode"], "code_unchanged": stable,
                  "wall_seconds": round(time.monotonic() - started, 6), "report": report}
        write_json(path, record)
        state["reports"][sid] = {"path": relative, "sha256": sha(path)}
        write_json(state_path, state)
        print(json.dumps({"attempted": len(state["reports"]), "planned": len(incoming), "document_id": sid,
                          "execution_complete": report.get("coverage", {}).get("execution_complete", False)}), flush=True)
        if not stable or "BudgetExceeded" in json.dumps(report, ensure_ascii=False):
            state["stop_reason"] = "source_changed_during_run" if not stable else "budget_limit"
            break
    state["last_completed_at"] = datetime.now(timezone.utc).isoformat()
    state["all_selected_attempted"] = len(state["reports"]) == len(incoming)
    state["budget_after"] = shared_budget(settings) if settings else None
    write_json(state_path, state)
    return state


def anchored_spans(error, content):
    result = []
    for raw in error.get("spans") or error.get("original_spans") or []:
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str) or not raw["text"]:
            continue
        text, start, end = raw["text"], raw.get("start"), raw.get("end")
        if type(start) is int and type(end) is int and 0 <= start < end <= len(content) and content[start:end] == text:
            result.append({"start": start, "end": end, "text": text})
        elif content.count(text) == 1:
            start = content.index(text)
            result.append({"start": start, "end": start + len(text), "text": text})
    return result


def same_problem(old, current, content):
    old_spans, new_spans = anchored_spans(old, content), anchored_spans(current, content)
    same_type = old.get("error_type", old.get("type")) == current.get("error_type", current.get("type"))
    for before in old_spans:
        for after in new_spans:
            if before["text"] == after["text"] or (same_type and max(before["start"], after["start"]) < min(before["end"], after["end"])):
                return {"same_type": same_type, "old_span": before, "current_span": after,
                        "basis": "exact_source_quote" if before["text"] == after["text"] else "same_type_overlapping_source"}
    return None


def ambiguous_relation(old, current, content):
    for before in anchored_spans(old, content):
        for after in anchored_spans(current, content):
            if max(before["start"], after["start"]) < min(before["end"], after["end"]):
                return {"old_span": before, "current_span": after, "basis": "overlapping_source_changed_type_or_quote"}
    return None


def complete_live_record(record):
    return bool(record and record.get("mode") == "live" and record.get("code_unchanged")
                and record["report"].get("coverage", {}).get("execution_complete") is True)


def repair_events(old_events, inputs, current_cases, records):
    """Count known events, not documents; unrun/failed/ambiguous remain unpaid debt."""
    by_doc = {}
    for case in current_cases:
        by_doc.setdefault(case["document_id"], []).append(case)
    repaired, new_fps = [], []
    for event in old_events:
        sid = event["document_id"]
        record = records.get(sid)
        eligible = complete_live_record(record)
        current = by_doc.get(sid, [])
        result = {**event, "status": "not_run" if not record else "failed_or_rules_only",
                  "business_repair_status": "pending_adjudication", "business_repair_pass": False,
                  "current_run_fingerprint": record.get("fingerprint") if record else None, "current_matching": []}
        if eligible:
            if event["result"] == "FN":
                result["current_matching"] = [case for case in current if case["result"] == "TP" and case["gold_index"] == event["gold_index"]]
                result["status"] = "benchmark_recovered" if result["current_matching"] else "persists"
            else:
                content = inputs[sid]["content"]
                anchor = anchored_spans(event["prediction"], content)
                if not anchor:
                    result["status"] = "unresolved_old_anchor"
                else:
                    unanchored_same_type = []
                    ambiguous = []
                    for case in current:
                        if case["result"] != "FP":
                            continue
                        relation = same_problem(event["prediction"], case["prediction"], content)
                        if relation:
                            result["current_matching"].append({"case": case, "relation": relation})
                        elif ambiguous_relation(event["prediction"], case["prediction"], content):
                            ambiguous.append({"case": case, "relation": ambiguous_relation(event["prediction"], case["prediction"], content)})
                        elif not anchored_spans(case["prediction"], content) and case["prediction"].get("error_type") == event["prediction"].get("error_type"):
                            unanchored_same_type.append(case)
                    result["status"] = ("persists" if result["current_matching"] else "unresolved_ambiguous_relation" if ambiguous
                                        else "unresolved_current_anchor" if unanchored_same_type else "benchmark_recovered")
                    result["ambiguous_current_matching"] = ambiguous
                    result["unresolved_current_candidates"] = unanchored_same_type
                    result["old_source_anchors"] = anchor
        repaired.append(result)
    for case in current_cases:
        if case["result"] != "FP":
            continue
        related = [event["event_id"] for event in old_events if event["result"] == "FP" and event["document_id"] == case["document_id"]
                   and same_problem(event["prediction"], case["prediction"], inputs[case["document_id"]]["content"])]
        new_fps.append({**case, "related_old_fp_events": related,
                        "side_effect": "persistent_known_FP" if related else "new_or_unresolved_FP"})
    # A broad old quote can contain distinct issues. Multiple current FP
    # allegations must not all be declared the same old FP merely because
    # their type and source passage coincide. Preserve the conservative old
    # event status, but expose this one-to-many linkage as an unresolved side
    # effect instead of hiding the added review workload.
    linked_counts = Counter(eid for row in new_fps for eid in row["related_old_fp_events"])
    for row in new_fps:
        competing = [eid for eid in row["related_old_fp_events"] if linked_counts[eid] > 1]
        if competing:
            row["side_effect"] = "new_or_unresolved_FP"
            row["linkage_status"] = "ambiguous_one_old_fp_to_multiple_current_candidates"
            row["ambiguous_old_fp_events"] = competing
    counts = {kind: {"denominator": sum(e["result"] == kind for e in repaired),
                     "recovered": sum(e["result"] == kind and e["status"] == "benchmark_recovered" for e in repaired)} for kind in ("FP", "FN")}
    for value in counts.values():
        value["recovery_rate"] = value["recovered"] / value["denominator"] if value["denominator"] else None
    fixed = sum(e["status"] == "benchmark_recovered" for e in repaired)
    return repaired, new_fps, {"old_FP_elimination": counts["FP"], "old_FN_recovery": counts["FN"],
        "metric": "benchmark_event_recovery", "recovered_events": fixed, "total_old_events": len(repaired),
        "recovery_rate": fixed / len(repaired) if repaired else None,
        "status_counts": dict(Counter(e["status"] for e in repaired)),
        "event_counting": "FP and FN are separate frozen events, including double events from one wrong-type issue; never deduplicated",
        "new_FP_not_subtracted_from_benchmark_recovery_rate": True,
        "numerical_rate_reaches95": bool(repaired) and fixed / len(repaired) >= .95,
        "benchmark_target95_reached": False,
        "business_repair_pending_adjudication": len(repaired), "business_repair_pass": False,
        "not_business_correctness": True, "not_new_dataset_accuracy": True}


def analyze(args):
    package, out = args.package.resolve(), args.out.resolve()
    audit = audit_package(package)
    inputs, golds, baseline, baseline_hints, old_events, scorer = baseline_data(package)
    records, run_meta = {}, []
    for directory in args.runs:
        directory = directory.resolve()
        plan = load_plan(directory)
        spec = plan["spec"]
        if (spec["source_manifest_sha256"] != audit["manifest_sha256"] or spec["source_gold_sha256"] != audit["gold_sha256"]
                or spec["frozen_scorer_sha256"] != audit["scorer_sha256"]):
            raise ValueError("run_does_not_match_frozen_baseline")
        state = read(directory / "state.json")
        if state["fingerprint"] != plan["fingerprint"]:
            raise ValueError("run_state_fingerprint_changed")
        for sid, saved in state["reports"].items():
            path = (directory / saved["path"]).resolve()
            if sid not in spec["document_ids"] or not path.is_relative_to(directory) or sha(path) != saved["sha256"]:
                raise ValueError("prediction_receipt_changed")
            record = read(path)
            if (record["fingerprint"] != plan["fingerprint"] or record["document_id"] != sid
                    or record["report"].get("document_id") != sid or record["mode"] != spec["runtime"]["mode"]
                    or record.get("candidate_version") != spec["candidate_version"]
                    or record["source_content_sha256"] != hashlib.sha256(inputs[sid]["content"].encode()).hexdigest()):
                raise ValueError("prediction_source_or_fingerprint_changed")
            records[sid] = record
        run_meta.append({"path": str(directory), "fingerprint": plan["fingerprint"], "mode": spec["runtime"]["mode"], "candidate_version": spec["candidate_version"]})
    predictions = {sid: validate_predictions(deepcopy(record["report"]), inputs[sid]["content"]) for sid, record in records.items()}
    hints = {sid: all_review_hints(prediction) for sid, prediction in predictions.items()}
    current_cases = [case for sid in sorted(hints) for case in cases_for_document(scorer, sid, hints[sid], golds[sid])]
    repairs, side_effects, repair_summary = repair_events(old_events, inputs, current_cases, records)
    versions = {record["candidate_version"] for record in records.values()}
    known_ids = {event["document_id"] for event in old_events}
    complete_ids = {sid for sid, record in records.items() if complete_live_record(record)}
    repair_summary.update(single_candidate_version=len(versions) == 1,
                          total_known_error_documents=len(known_ids),
                          completed_known_error_documents=len(known_ids & complete_ids),
                          all_known_error_documents_live_complete=known_ids <= complete_ids)
    completed_events = [event for event in repairs if event["document_id"] in complete_ids]
    attempted_events = [event for event in repairs if event["document_id"] in records]
    repair_summary["completed_batch_events"] = len(completed_events)
    repair_summary["completed_batch_recovered_events"] = sum(event["status"] == "benchmark_recovered" for event in completed_events)
    repair_summary["completed_batch_recovery_rate"] = (repair_summary["completed_batch_recovered_events"] / len(completed_events)) if completed_events else None
    repair_summary["attempted_batch_events"] = len(attempted_events)
    repair_summary["attempted_batch_recovered_events"] = sum(event["status"] == "benchmark_recovered" for event in attempted_events)
    repair_summary["attempted_batch_recovery_rate"] = (repair_summary["attempted_batch_recovered_events"] / len(attempted_events)) if attempted_events else None
    repair_summary["attempted_batch_incomplete_events"] = len(attempted_events) - len(completed_events)
    repair_summary["batch_rate_interpretation"] = "Attempted batch includes failed/rules-only events. Completed batch is diagnostic only; neither replaces the fixed all-old-event denominator."
    repair_summary["benchmark_target95_reached"] = bool(repair_summary["numerical_rate_reaches95"] and len(versions) == 1 and known_ids <= complete_ids)
    strict = scorer.score_paper_detection(hints.values(), golds.values())
    selected_gold = [golds[sid] for sid in sorted(records)]
    before_selected = scorer.score_paper_detection([baseline_hints[sid] for sid in sorted(records)], selected_gold)
    after_selected = scorer.score_paper_detection(hints.values(), selected_gold)
    current_tp = {(case["document_id"], case["gold_index"]) for case in current_cases if case["result"] == "TP"}
    tp_regressions = []
    for sid in sorted(complete_ids):
        for case in cases_for_document(scorer, sid, baseline_hints[sid], golds[sid]):
            if case["result"] == "TP" and (sid, case["gold_index"]) not in current_tp:
                tp_regressions.append({**case, "gold_id": f"{sid}:gold:{case['gold_index']}",
                                       "current_matching": [c for c in current_cases if c["document_id"] == sid and c["gold_index"] == case["gold_index"]],
                                       "candidate_version": records[sid]["candidate_version"]})
    typed_counts = {}
    for kind in sorted({(event["prediction"] or event["gold"]).get("error_type", (event["prediction"] or event["gold"]).get("type")) for event in repairs}):
        own = [event for event in repairs if (event["prediction"] or event["gold"]).get("error_type", (event["prediction"] or event["gold"]).get("type")) == kind]
        typed_counts[kind] = {"events": len(own), "benchmark_recovered": sum(event["status"] == "benchmark_recovered" for event in own)}
    calls = {}
    for record in records.values():
        calls.update(runtime_calls(record["report"]))
    report = {"schema_version": SCHEMA, "role": ROLE, "old_dataset_in_new_suite": False, "runs": run_meta,
              "selection_policy": "Last explicitly supplied run per document; never choose its best score",
              "mixed_iteration_fingerprints": len({record["fingerprint"] for record in records.values()}) > 1,
              "candidate_versions": sorted(versions), "mixed_candidate_versions": len(versions) > 1,
              "benchmark_event_recovery": repair_summary, "by_error_type": typed_counts,
              "business_repair": {"status": "pending_adjudication", "pass": False,
                                  "reason": "Frozen FP can be an unlabelled real issue or have incorrect reasoning. An absent unmatched hint does not establish business correctness."},
              "scoring_denominator": {"documents": len(golds), "scorable_gold_errors": audit["scorable_gold_errors"], "excluded_gold_errors": audit["excluded_gold_errors"]},
              "strict_all_hints_all_442": strict,
              "strict_attempted_documents_before": before_selected,
              "strict_attempted_documents_after": after_selected,
              "all_442_strict_interpretation": "Coverage-inclusive audit only; missing predictions remain FN. Use attempted-document before/after for batch comparison; neither is the 95% goal.",
              "strict_candidate_all_442": scorer.score_paper_detection(predictions.values(), golds.values()),
              "strict_confirmed_all_442": scorer.score_paper_detection([{**p, "errors": [e for e in p["errors"] if e.get("status") == "confirmed_error"]} for p in predictions.values()], golds.values()),
              "frozen_baseline_strict_all_hints": audit["baseline_strict_all_hints"],
              "attempted_documents": len(records), "live_complete_documents": sum(record["mode"] == "live" and record["code_unchanged"] and record["report"].get("coverage", {}).get("execution_complete") is True for record in records.values()),
              "new_or_unresolved_FP": sum(row["side_effect"] == "new_or_unresolved_FP" for row in side_effects),
              "one_to_many_FP_linkage_candidates": sum(row.get("linkage_status") == "ambiguous_one_old_fp_to_multiple_current_candidates" for row in side_effects),
              "old_TP_regression_count": len(tp_regressions),
              "old_TP_regression_gold_ids": [row["gold_id"] for row in tp_regressions],
              "old_TP_regression_scope": "Only live-complete documents; incomplete/unrun are not evidence of semantic regression",
              "model_calls_in_selected_reports": len(calls), "accounted_cny_in_selected_reports": str(sum((Decimal(str(call.get("cost_cny", "0"))) for call in calls.values()), Decimal(0))),
              "prior_run_model_calls_reused": sum(bool(call.get("reused_from_prior_run")) for call in calls.values()),
              "new_model_calls": sum(not call.get("reused_from_prior_run") for call in calls.values()),
              "newly_incurred_cny": str(sum((Decimal(str(call.get("cost_cny", "0"))) for call in calls.values() if not call.get("reused_from_prior_run")), Decimal(0))),
              "previously_incurred_cny_reused": str(sum((Decimal(str(call.get("cost_cny", "0"))) for call in calls.values() if call.get("reused_from_prior_run")), Decimal(0))),
              "response_reuse_interpretation": "Exact-request raw-response replay tests current postprocessing; it is not fresh model inference or independent end-to-end latency.",
              "interpretation": "This is mechanical benchmark_event_recovery across supplied iteration records, not business correctness or current-version independent accuracy. Business repairs remain pending adjudication. Unrun, failed, rules-only and anchor-ambiguous events cannot count recovered. New FP side effects are separate. All 442 gold records and excluded flags remain unchanged."}
    if out.exists() or out.is_relative_to(package) or out.is_relative_to(ROOT / "data/active_suite_v1"):
        raise ValueError("fresh_analysis_output_outside_datasets_required")
    out.mkdir(parents=True)
    for name, values in (("old-event-repairs.jsonl", repairs), ("current-strict-cases.jsonl", current_cases), ("current-FP-side-effects.jsonl", side_effects), ("old-TP-regressions.jsonl", tp_regressions)):
        (out / name).write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
    write_json(out / "report.json", report)
    print(json.dumps({"role": ROLE, "attempted_documents": len(records), "benchmark_event_recovery": repair_summary,
                      "new_or_unresolved_FP": report["new_or_unresolved_FP"]}, ensure_ascii=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--package", type=Path, default=PACKAGE)
    plan = sub.add_parser("plan")
    plan.add_argument("--package", type=Path, default=PACKAGE)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--document-id", action="append", default=[])
    plan.add_argument("--doc-ids", nargs="+", default=[], help="Full IDs or unique short IDs, space/comma separated")
    plan.add_argument("--max-output-tokens", type=int, default=16384, help="Frozen per-call round cap, default 16384; retries always 0")
    plan.add_argument("--error-type", action="append", default=[])
    plan.add_argument("--limit", type=int, default=10, help="Known-error documents, preserving full source; default 10")
    plan.add_argument("--all-known-errors", action="store_true", help="Only documents containing old FP/FN, not all 442")
    plan.add_argument("--live", action="store_true", help="Freeze existing model/budget settings; plan itself makes no calls")
    plan.add_argument("--reuse-model-responses-from", action="append", type=Path, default=[], help="Import successful exact-request raw responses from another verified plan")
    plan.add_argument("--replay-only", action="store_true", help="With explicit response sources: forbid all cache-miss network requests")
    plan.add_argument("--redundancy-review", action="store_true", help="Opt into one independent redundancy review per document; default off and bound to the plan version")
    plan.add_argument("--redundancy-review-policy", choices=REDUNDANCY_REVIEW_POLICIES, default="actions_v1",
                      help="Independent review protocol; a nondefault policy requires --redundancy-review")
    plan.add_argument("--redundancy-review-hide-reason", action="store_true",
                      help="Omit the base candidate reason from review input; requires --redundancy-review")
    plan.add_argument("--cached-detection-only", action="store_true", help="Require live redundancy review and response sources; only the review purpose may make new requests")
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--live", action="store_true", help="Explicitly allow calls under an already-live plan and existing shared cap")
    score = sub.add_parser("analyze")
    score.add_argument("--package", type=Path, default=PACKAGE)
    score.add_argument("--runs", type=Path, nargs="+", required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan" and args.limit <= 0:
        parser.error("--limit must be positive; use --all-known-errors explicitly")
    if args.command == "audit":
        print(json.dumps(audit_package(args.package), ensure_ascii=False))
    elif args.command == "plan":
        create_plan(args)
    elif args.command == "run":
        run_batch(args)
    else:
        analyze(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, ValueError) else "io_or_contract_failure"}), file=sys.stderr)
        raise SystemExit(2)
