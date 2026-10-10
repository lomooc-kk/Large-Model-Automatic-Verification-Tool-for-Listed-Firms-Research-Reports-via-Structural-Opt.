"""Offline boundary tests; synthetic content only, never paid calls."""
from argparse import Namespace
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from evals import run_known_error_repair as runner
from evals.run_v2 import request_cache_key, request_messages_hash


def error(kind="A", text="bc", start=1, **extra):
    return {"error_type": kind, "spans": [{"text": text, "start": start, "end": start + len(text)}], **extra}


def live_record(document="source:000001", errors=(), *, complete=True, version="v1"):
    return {"mode": "live", "code_unchanged": True, "candidate_version": version, "fingerprint": "plan",
            "report": {"document_id": document, "errors": list(errors), "coverage": {"execution_complete": complete}}}


@pytest.fixture
def suite(tmp_path, monkeypatch):
    package = tmp_path / "frozen"
    (package / "inputs").mkdir(parents=True)
    ids = ["source:000001", "source:000002"]
    inputs = {ids[0]: {"document_id": ids[0], "content": "abcdefgh"},
              ids[1]: {"document_id": ids[1], "content": "ijklmnop"}}
    golds = {ids[0]: {"document_id": ids[0], "errors": [error("A"), error("A", "ef", 4), error("X", "gh", 6, scorable=False)]},
             ids[1]: {"document_id": ids[1], "errors": [error("A", "jk", 1)]}}
    for row in golds.values():
        for item in row["errors"]:
            item["type"] = item.pop("error_type")
    baseline = {ids[0]: {"document_id": ids[0], "errors": [error("B"), error("A", "ef", 4)]},
                ids[1]: {"document_id": ids[1], "errors": []}}
    scorer = runner.load_scorer(runner.PACKAGE)
    events = []
    for sid in ids:
        for case in runner.cases_for_document(scorer, sid, baseline[sid], golds[sid]):
            if case["result"] != "TP":
                offset = case["gold_index"] if case["result"] == "FN" else case["prediction_index"]
                events.append({**case, "event_id": f"{case['result']}:{sid}:{offset}"})
    (package / runner.INPUT).write_text("".join(json.dumps(row) + "\n" for row in inputs.values()), encoding="utf-8")
    audit = {"manifest_sha256": "manifest", "input_sha256": runner.sha(package / runner.INPUT), "gold_sha256": "gold", "scorer_sha256": "scorer",
             "documents": 2, "scorable_gold_errors": 3, "excluded_gold_errors": 1,
             "old_event_count": 3, "old_event_counts": {"FP": 1, "FN": 2}, "event_denominator": "separate FP and FN",
             "known_events": [{"event_id": e["event_id"], "document_id": e["document_id"], "result": e["result"],
                               "error_type": (e["prediction"] or e["gold"]).get("error_type", (e["prediction"] or e["gold"]).get("type"))} for e in events],
             "baseline_strict_all_hints": scorer.score_paper_detection(baseline.values(), golds.values())}
    monkeypatch.setattr(runner, "admission", lambda package: audit)
    monkeypatch.setattr(runner, "audit_package", lambda package: audit)
    monkeypatch.setattr(runner, "baseline_data", lambda package: (inputs, golds, baseline, baseline, events, scorer))
    relative = "evals/run_known_error_repair.py"
    codes = {relative: runner.sha(runner.ROOT / relative)}
    monkeypatch.setattr(runner, "source_hashes", lambda: codes)
    settings = runner.RuntimeSettings(context_tokens=1000000, max_output_tokens=65536, max_retries=5, budget_cny="100", ledger=tmp_path / "shared.sqlite3")
    monkeypatch.setattr(runner.RuntimeSettings, "from_env", lambda **kw: settings)
    config = SimpleNamespace(model="fake-model", base_url="https://invalid.example", thinking="disabled", reasoning_effort="", timeout=60, api_key="never-log-this")
    monkeypatch.setattr(runner.ModelConfig, "from_env", lambda: config)
    monkeypatch.setattr(runner, "shared_budget", lambda s: {"ledger": str(s.ledger), "budget_cny": "100", "accounted_cny": "29", "calls": 10})
    monkeypatch.setattr(runner, "BudgetedChatClient", lambda *a: pytest.fail("No model client should be constructed in a free run"))
    return SimpleNamespace(package=package, ids=ids, inputs=inputs, golds=golds, baseline=baseline, events=events, scorer=scorer,
                           audit=audit, codes=codes, settings=settings, root=tmp_path)


def plan(suite, name="batch", live=False, documents=None, **options):
    return runner.create_plan(Namespace(package=suite.package, out=suite.root / name, document_id=[], doc_ids=documents or [],
                                        error_type=[], all_known_errors=not documents, limit=10, live=live, max_output_tokens=16384,
                                        **options))


def install_fake_live(monkeypatch, reports):
    monkeypatch.setattr(runner, "BudgetedChatClient", lambda *args: object())
    def detect(content, **kwargs):
        assert kwargs["detector"] == "hybrid" and kwargs["examples"] == ()
        assert isinstance(kwargs["chat"], runner.SharedCandidateClient)
        return reports[kwargs["document_id"]]
    monkeypatch.setattr(runner, "detect_text", detect)


def cached_response(suite, source="source-run", *, content='{"errors":[]}', failed=False, purpose="text_review.detect",
                    review_policy="actions_v1", hide_reason=False, stage_metadata=False, documents=None):
    is_review = purpose == runner.REDUNDANCY_REVIEW_PURPOSE
    prior = plan(suite, source, live=True, documents=documents, redundancy_review=is_review,
                 redundancy_review_policy=review_policy, redundancy_review_hide_reason=hide_reason)
    messages = [{"role": "user", "content": "source only"}]
    key = request_cache_key(messages, purpose)
    rt = prior["spec"]["runtime"]
    trace = {"call_id": "paid-original-call", "status": "error" if failed else "ok", "model": rt["model"],
             "thinking": rt["thinking"], "reasoning_effort": rt["reasoning_effort"], "purpose": purpose,
             "request_sha256": request_messages_hash(messages), "max_output_tokens": rt["runtime"]["max_output_tokens"], "cost_cny": "0.1"}
    record = {"content": content, "trace": trace, "cache_request": {"key": key, "purpose": purpose, "messages_sha256": trace["request_sha256"]}}
    if failed:
        record["failed"] = True
    runner.write_json(suite.root / source / "model_responses" / (key + ".json"), record)
    if not failed:
        errors, parsed_trace = ([], trace) if is_review else runner._parse_response(record)
        sid = suite.ids[0]
        relative = "predictions/" + hashlib.sha256(sid.encode()).hexdigest()[:24] + ".json"
        receipt = {"schema_version": runner.SCHEMA, "document_id": sid, "fingerprint": prior["fingerprint"],
                   "candidate_version": prior["spec"]["candidate_version"], "mode": "live", "code_unchanged": True,
                   "source_content_sha256": hashlib.sha256(suite.inputs[sid]["content"].encode()).hexdigest(),
                   "report": {"document_id": sid, "coverage": {"execution_complete": True},
                              "traces": [{"job_index": 0, "status": "ok", "purpose": purpose, "runtime_trace": parsed_trace}],
                              "raw_candidates": [{"candidate": item, "job_index": 0, "candidate_ref": f"model:0:{i}"}
                                                 for i, item in enumerate(errors)]}}
        if is_review:
            receipt["report"]["redundancy_review"] = {
                "schema_version": "redundancy-review/2.0" if review_policy == "context_v2" else "redundancy-review/1.0", "purpose": purpose,
                "status": "complete", "complete": True, "raw_response": content,
                "response_sha256": hashlib.sha256(content.encode()).hexdigest(), "runtime_trace": trace}
            if stage_metadata or review_policy != "actions_v1" or hide_reason:
                receipt["report"]["redundancy_review"].update(policy=review_policy, include_reason=not hide_reason)
        runner.write_json(suite.root / source / relative, receipt)
        runner.write_json(suite.root / source / "state.json", {"fingerprint": prior["fingerprint"],
            "reports": {sid: {"path": relative, "sha256": runner.sha(suite.root / source / relative)}}})
    return prior, messages, key, record


def test_reuse_only_imports_raw_response_with_verified_provenance(suite):
    prior, messages, key, record = cached_response(suite)
    manifests, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])
    assert manifests[0]["successful_records"] == 1
    assert records[key]["content"] == record["content"]
    assert records[key]["cache_provenance"]["source_fingerprint"] == prior["fingerprint"]
    proof = records[key]["cache_provenance"]["source_response_integrity"]
    assert proof["basis"] == "completed_receipt_raw_candidates_and_trace"
    assert proof["response_bytes_authenticated"] is False
    assert set(records[key]) == {"content", "trace", "cache_request", "cache_provenance"}
    runner.import_model_responses(suite.root / "next/model_responses", records)
    client = runner.SharedCandidateClient(runner.refuse_uncached_request, suite.root / "next/model_responses")
    reply = client(messages, purpose="text_review.detect")
    assert reply["trace"]["reused_from_prior_run"] is True
    with pytest.raises(RuntimeError, match="no_network_request"):
        client([{"role": "user", "content": "changed prompt"}], purpose="text_review.detect")


def test_review_response_import_uses_exact_raw_text_not_errors_parser(suite, monkeypatch):
    content = '{"decisions":[{"candidate_ref":"candidate:0","verdict":"keep"}]}'
    prior, messages, key, record = cached_response(suite, content=content, purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    monkeypatch.setattr(runner, "_parse_response", lambda *args: pytest.fail("Review decisions cannot use the errors parser"))
    _, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
                                                      suite.root / "next", prior["spec"]["max_input_tokens"])
    proof = records[key]["cache_provenance"]["source_response_integrity"]
    assert proof["basis"] == "completed_receipt_redundancy_response_and_trace"
    assert proof["response_bytes_authenticated"] is True
    assert proof["response_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert records[key]["content"] == content and records[key]["trace"] == record["trace"]
    runner.import_model_responses(suite.root / "next/model_responses", records)
    client = runner.SharedCandidateClient(runner.refuse_uncached_request, suite.root / "next/model_responses")
    assert client(messages, purpose=runner.REDUNDANCY_REVIEW_PURPOSE)["trace"]["reused_from_prior_run"]
    with pytest.raises(RuntimeError, match="no_network_request"):
        client(messages, purpose="text_review.detect")


@pytest.mark.parametrize("policy,hide_reason", [("actions_v1", False), ("actions_v1", True),
                                              ("context_v2", False), ("context_v2", True)])
def test_review_policy_receipts_preserve_exact_raw_response_and_cache_messages(suite, monkeypatch, policy, hide_reason):
    content = '```json\n{"decisions":[{"candidate_ref":"candidate:0","verdict":"not_redundant","context_units":[],"new_information":"different context"}]}\n```'
    prior, messages, key, record = cached_response(suite, content=content, purpose=runner.REDUNDANCY_REVIEW_PURPOSE,
        review_policy=policy, hide_reason=hide_reason, stage_metadata=True)
    monkeypatch.setattr(runner, "_parse_response", lambda *args: pytest.fail("Review cannot be coerced to errors"))
    _, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
        suite.root / "next", prior["spec"]["max_input_tokens"])
    assert records[key]["content"] == content and records[key]["trace"] == record["trace"]
    assert records[key]["cache_provenance"]["source_response_integrity"]["response_bytes_authenticated"] is True
    runner.import_model_responses(suite.root / "next/model_responses", records)
    client = runner.SharedCandidateClient(runner.refuse_uncached_request, suite.root / "next/model_responses")
    assert client(messages, purpose=runner.REDUNDANCY_REVIEW_PURPOSE)["trace"]["reused_from_prior_run"]
    # Same purpose does not permit a cache hit when an experiment changes its actual prompt.
    with pytest.raises(RuntimeError, match="no_network_request"):
        client([{"role": "user", "content": "different policy or omitted reason"}], purpose=runner.REDUNDANCY_REVIEW_PURPOSE)


@pytest.mark.parametrize("mutation", ["schema", "policy", "include_reason", "include_reason_int",
                                      "missing_policy", "missing_include_reason"])
def test_context_review_receipt_rejects_policy_schema_or_visibility_mismatch(suite, mutation):
    prior, _, _, _ = cached_response(suite, content='{"decisions":[]}', purpose=runner.REDUNDANCY_REVIEW_PURPOSE,
                                     review_policy="context_v2", hide_reason=True)
    state_path = suite.root / "source-run/state.json"
    state = runner.read(state_path)
    receipt_path = suite.root / "source-run" / state["reports"][suite.ids[0]]["path"]
    receipt = runner.read(receipt_path)
    stage = receipt["report"]["redundancy_review"]
    if mutation.startswith("missing_"):
        del stage[mutation[len("missing_"):]]
    else:
        field, value = {"schema": ("schema_version", "redundancy-review/1.0"),
                        "policy": ("policy", "actions_v1"), "include_reason": ("include_reason", True),
                        "include_reason_int": ("include_reason", 0)}[mutation]
        stage[field] = value
    runner.write_json(receipt_path, receipt)
    state["reports"][suite.ids[0]]["sha256"] = runner.sha(receipt_path)
    runner.write_json(state_path, state)
    with pytest.raises(ValueError, match="response_source_redundancy_receipt_mismatch"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
                                           suite.root / "next", prior["spec"]["max_input_tokens"])


def test_actions_hidden_reason_receipt_cannot_use_legacy_default_metadata(suite):
    prior, _, _, _ = cached_response(suite, content='{"decisions":[]}', purpose=runner.REDUNDANCY_REVIEW_PURPOSE,
                                     hide_reason=True)
    state_path = suite.root / "source-run/state.json"
    state = runner.read(state_path)
    receipt_path = suite.root / "source-run" / state["reports"][suite.ids[0]]["path"]
    receipt = runner.read(receipt_path)
    del receipt["report"]["redundancy_review"]["include_reason"]
    runner.write_json(receipt_path, receipt)
    state["reports"][suite.ids[0]]["sha256"] = runner.sha(receipt_path)
    runner.write_json(state_path, state)
    with pytest.raises(ValueError, match="response_source_redundancy_receipt_mismatch"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
                                           suite.root / "next", prior["spec"]["max_input_tokens"])


@pytest.mark.parametrize("mutation", ["decisions", "formatting", "cost", "future_trace", "request", "provenance"])
def test_review_cache_cannot_forge_response_or_trace(suite, mutation):
    prior, _, key, record = cached_response(suite, content='{"decisions":[]}', purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    if mutation == "decisions":
        record["content"] = '{"decisions":[{"verdict":"drop"}]}'
    elif mutation == "formatting":
        record["content"] = '{ "decisions": [] }'
    elif mutation == "request":
        record["trace"]["request_sha256"] = "0" * 64
        record["cache_request"]["messages_sha256"] = "0" * 64
    elif mutation == "provenance":
        record["cache_provenance"] = {"source_run": "untracked"}
    else:
        record["trace"]["cost_cny" if mutation == "cost" else "unknown_future_trace"] = "changed"
    runner.write_json(suite.root / "source-run/model_responses" / (key + ".json"), record)
    with pytest.raises(ValueError, match="response_source_(raw_receipt_mismatch|imported_manifest_required)"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
                                           suite.root / "next", prior["spec"]["max_input_tokens"])


@pytest.mark.parametrize("mutation", ["state_hash", "missing_stage", "raw_hash", "trace", "purpose", "incomplete", "duplicate_job"])
def test_review_response_requires_complete_consistent_frozen_receipt(suite, mutation):
    prior, _, _, _ = cached_response(suite, content='{"decisions":[]}', purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    state_path = suite.root / "source-run/state.json"
    state = runner.read(state_path)
    receipt_path = suite.root / "source-run" / state["reports"][suite.ids[0]]["path"]
    receipt = runner.read(receipt_path)
    stage = receipt["report"]["redundancy_review"]
    if mutation in {"state_hash", "raw_hash"}:
        stage["raw_response"] = '{"decisions":[{}]}'
    elif mutation == "missing_stage":
        del receipt["report"]["redundancy_review"]
    elif mutation == "trace":
        stage["runtime_trace"]["cost_cny"] = "changed"
    elif mutation == "purpose":
        stage["purpose"] = "text_review.detect"
    elif mutation == "incomplete":
        stage.update(status="incomplete", complete=False)
    else:
        receipt["report"]["traces"].append({**receipt["report"]["traces"][0], "job_index": 1})
    runner.write_json(receipt_path, receipt)
    if mutation != "state_hash":
        state["reports"][suite.ids[0]]["sha256"] = runner.sha(receipt_path)
        runner.write_json(state_path, state)
    with pytest.raises(ValueError, match="response_source_(completed_receipt_changed|redundancy_receipt_mismatch)"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
                                           suite.root / "next", prior["spec"]["max_input_tokens"])


def test_same_completed_report_authenticates_detection_and_review_separately(suite):
    prior, messages, review_key, review_record = cached_response(
        suite, content='{"decisions":[]}', purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    detect_key = request_cache_key(messages, "text_review.detect")
    detect_trace = {**review_record["trace"], "purpose": "text_review.detect", "call_id": "separate-detection-call"}
    detect_record = {"content": '{"errors":[{"reason":"original detection"}]}', "trace": detect_trace,
                     "cache_request": {"key": detect_key, "purpose": "text_review.detect", "messages_sha256": request_messages_hash(messages)}}
    runner.write_json(suite.root / "source-run/model_responses" / (detect_key + ".json"), detect_record)
    state_path = suite.root / "source-run/state.json"
    state = runner.read(state_path)
    receipt_path = suite.root / "source-run" / state["reports"][suite.ids[0]]["path"]
    receipt = runner.read(receipt_path)
    receipt["report"]["traces"][0]["job_index"] = 1
    receipt["report"]["traces"].insert(0, {"job_index": 0, "purpose": "text_review.detect", "status": "ok", "runtime_trace": detect_trace})
    receipt["report"]["raw_candidates"] = [{"candidate": {"reason": "original detection"}, "job_index": 0, "candidate_ref": "model:0:0"}]
    runner.write_json(receipt_path, receipt)
    state["reports"][suite.ids[0]]["sha256"] = runner.sha(receipt_path)
    runner.write_json(state_path, state)
    _, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
                                                      suite.root / "next", prior["spec"]["max_input_tokens"])
    assert set(records) == {detect_key, review_key}
    assert records[detect_key]["cache_provenance"]["source_response_integrity"]["response_bytes_authenticated"] is False
    assert records[review_key]["cache_provenance"]["source_response_integrity"]["response_bytes_authenticated"] is True


def test_reuse_rejects_changed_model_settings_or_input_budget(suite):
    prior, _, _, _ = cached_response(suite)
    runtime = prior["spec"]["runtime"]
    for changed, cap in [({**runtime, "reasoning_effort": "high"}, prior["spec"]["max_input_tokens"]),
                         (runtime, 123), ({**runtime, "mode": "rules_only"}, 123)]:
        with pytest.raises(ValueError, match="response_.*(mismatch|configuration)"):
            runner.prepare_known_response_reuse([suite.root / "source-run"], changed, suite.root / "next", cap)


def test_failed_source_response_is_not_imported(suite):
    prior, _, _, _ = cached_response(suite, failed=True)
    manifests, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])
    assert records == {} and manifests[0]["failed_records_skipped"] == 1


def test_reuse_allows_only_price_verification_metadata_and_keeps_original_cost(suite):
    prior, _, key, record = cached_response(suite)
    runtime = json.loads(json.dumps(prior["spec"]["runtime"]))
    runtime["runtime"].update(price_date="2030-01-02", price_valid_until="2030-01-03T00:00:00Z", price_source="https://official.example/pricing")
    manifests, records = runner.prepare_known_response_reuse([suite.root / "source-run"], runtime, suite.root / "next", prior["spec"]["max_input_tokens"])
    assert records[key]["trace"] == record["trace"]
    assert records[key]["trace"]["cost_cny"] == "0.1"
    assert manifests[0]["price_verification_metadata_changed"] is True
    assert manifests[0]["source_runtime_sha256"] != manifests[0]["destination_runtime_sha256"]
    assert prior["spec"]["runtime"]["runtime"]["price_date"] != runtime["runtime"]["price_date"]


@pytest.mark.parametrize("key,value,nested", [
    ("model", "another-model", False), ("endpoint_sha256", "different", False),
    ("thinking", "enabled", False), ("timeout", 481, False),
    ("max_output_tokens", 32768, True), ("context_tokens", 800000, True),
    ("max_retries", 2, True), ("input_cny_per_mtok", "10", True),
    ("budget_cny", "200", True), ("unknown_future_setting", "new", True),
])
def test_price_refresh_does_not_relax_other_runtime_checks(suite, key, value, nested):
    prior, _, _, _ = cached_response(suite)
    runtime = json.loads(json.dumps(prior["spec"]["runtime"]))
    (runtime["runtime"] if nested else runtime)[key] = value
    runtime["runtime"]["price_date"] = "2030-01-02"
    with pytest.raises(ValueError, match="runtime_mismatch"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], runtime, suite.root / "next", prior["spec"]["max_input_tokens"])


def test_reuse_rejects_inconsistent_request_receipt(suite):
    prior, _, key, record = cached_response(suite)
    record["cache_request"]["messages_sha256"] = "0" * 64
    runner.write_json(suite.root / "source-run/model_responses" / (key + ".json"), record)
    with pytest.raises(ValueError, match="request_metadata_mismatch"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])


def test_conflicting_source_responses_are_not_cherry_picked(suite):
    prior, _, _, _ = cached_response(suite, "source-a", content='{"errors":[{"reason":"first response"}]}')
    cached_response(suite, "source-b", content='{"errors":[{"reason":"different response"}]}')
    with pytest.raises(ValueError, match="conflicting_responses"):
        runner.prepare_known_response_reuse([suite.root / "source-a", suite.root / "source-b"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])


@pytest.mark.parametrize("mutation", ["content", "cost_cny", "input_cny_per_mtok", "future_trace_field", "request", "untracked_origin"])
def test_source_cache_cannot_self_authenticate_before_import(suite, mutation):
    prior, _, key, record = cached_response(suite)
    if mutation == "content":
        record["content"] = '{"errors":[{"reason":"fabricated finding"}]}'
    elif mutation == "request":
        record["trace"]["request_sha256"] = "0" * 64
        record["cache_request"]["messages_sha256"] = "0" * 64
    elif mutation == "untracked_origin":
        record["cache_provenance"] = {"source_run": "untracked"}
    else:
        record["trace"][mutation] = "changed"
    runner.write_json(suite.root / "source-run/model_responses" / (key + ".json"), record)
    with pytest.raises(ValueError, match="response_source_(raw_receipt_mismatch|imported_manifest_required)"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])


@pytest.mark.parametrize("mutation", ["missing_state", "state_fingerprint", "receipt_content", "incomplete", "wrong_source"])
def test_fresh_response_requires_independent_completed_receipt(suite, mutation):
    prior, _, _, _ = cached_response(suite)
    state_path = suite.root / "source-run/state.json"
    state = runner.read(state_path)
    receipt_path = suite.root / "source-run" / state["reports"][suite.ids[0]]["path"]
    if mutation == "missing_state":
        state_path.unlink()
    elif mutation == "state_fingerprint":
        state["fingerprint"] = "different"
        runner.write_json(state_path, state)
    else:
        receipt = runner.read(receipt_path)
        if mutation == "incomplete":
            receipt["report"]["coverage"]["execution_complete"] = False
        elif mutation == "wrong_source":
            receipt["source_content_sha256"] = "0" * 64
        else:
            receipt["report"]["traces"][0]["runtime_trace"]["cost_cny"] = "altered"
        runner.write_json(receipt_path, receipt)
        if mutation != "receipt_content":
            state["reports"][suite.ids[0]]["sha256"] = runner.sha(receipt_path)
            runner.write_json(state_path, state)
    with pytest.raises(ValueError, match="response_source_"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])


def test_source_plan_locked_import_needs_no_prediction_or_gold_read(suite, monkeypatch):
    prior, _, key, record = cached_response(suite)
    prior["spec"]["imported_response_hashes"] = {key: runner.fingerprint(record)}
    prior["fingerprint"] = runner.fingerprint(prior["spec"])
    runner.write_json(suite.root / "source-run/plan.json", prior)
    monkeypatch.setattr(runner, "completed_response_receipts", lambda *args: pytest.fail("Locked import must use original plan, not predictions"))
    _, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])
    assert records[key]["cache_provenance"]["source_response_integrity"] == {
        "basis": "source_plan_imported_response_hash", "record_sha256": runner.fingerprint(record)}


@pytest.mark.parametrize("mutation", ["content", "trace_cost", "failed", "missing"])
def test_source_plan_locked_import_rejects_tampering_before_skip(suite, mutation):
    prior, _, key, record = cached_response(suite)
    prior["spec"]["imported_response_hashes"] = {key: runner.fingerprint(record)}
    prior["fingerprint"] = runner.fingerprint(prior["spec"])
    runner.write_json(suite.root / "source-run/plan.json", prior)
    cache_path = suite.root / "source-run/model_responses" / (key + ".json")
    if mutation == "missing":
        cache_path.unlink()
    else:
        if mutation == "content":
            record["content"] = '{"errors":[{}]}'
        elif mutation == "trace_cost":
            record["trace"]["cost_cny"] = "0"
        else:
            record["failed"] = True
        runner.write_json(cache_path, record)
    with pytest.raises(ValueError, match="source_imported_record_changed"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])


def test_receipt_preserves_raw_candidate_order_and_same_run_cache_hit(suite):
    content = '{"errors":[{"reason":"one"},{"reason":"two"}]}'
    prior, _, key, record = cached_response(suite, content=content)
    state_path = suite.root / "source-run/state.json"
    state = runner.read(state_path)
    path = suite.root / "source-run" / state["reports"][suite.ids[0]]["path"]
    receipt = runner.read(path)
    receipt["report"]["traces"][0]["runtime_trace"]["reused_response"] = True
    runner.write_json(path, receipt)
    state["reports"][suite.ids[0]]["sha256"] = runner.sha(path)
    runner.write_json(state_path, state)
    _, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])
    assert records[key]["trace"] == record["trace"]
    record["content"] = '{"errors":[{"reason":"two"},{"reason":"one"}]}'
    runner.write_json(suite.root / "source-run/model_responses" / (key + ".json"), record)
    with pytest.raises(ValueError, match="source_raw_receipt_mismatch"):
        runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"], suite.root / "next", prior["spec"]["max_input_tokens"])


def test_replay_plan_checks_import_integrity_before_any_network_client(suite):
    _, _, key, record = cached_response(suite)
    args = Namespace(package=suite.package, out=suite.root / "replay", document_id=[], doc_ids=[],
                     error_type=[], all_known_errors=True, limit=10, live=True, max_output_tokens=16384,
                     reuse_model_responses_from=[suite.root / "source-run"], replay_only=True)
    made = runner.create_plan(args)
    assert made["spec"]["replay_only"] is True and key in made["spec"]["imported_response_hashes"]
    imported_path = suite.root / "replay/model_responses" / (key + ".json")
    imported = runner.read(imported_path)
    imported["content"] = "tampered"
    runner.write_json(imported_path, imported)
    with pytest.raises(ValueError, match="imported_response_changed"):
        runner.run_batch(Namespace(out=suite.root / "replay", live=True))


def test_replay_plan_requires_a_source(suite):
    with pytest.raises(ValueError, match="requires_explicit_response_source"):
        runner.create_plan(Namespace(package=suite.package, out=suite.root / "no-source", document_id=[], doc_ids=[],
                           error_type=[], all_known_errors=True, limit=10, live=True, max_output_tokens=16384,
                           reuse_model_responses_from=[], replay_only=True))


def test_short_ids_are_exact_and_unique():
    assert runner.resolve_document_ids(["000001,000002"], {"x:000001", "x:000002"}) == ["x:000001", "x:000002"]
    assert runner.resolve_document_ids(["x:000001"], {"x:000001", "y:000001"}) == ["x:000001"]
    for requested, eligible in [(["000001"], {"x:000001", "y:000001"}), (["001"], {"x:000001"}), (["missing"], {"x:000001"})]:
        with pytest.raises(ValueError, match="unknown_or_ambiguous"):
            runner.resolve_document_ids(requested, eligible)
    with pytest.raises(ValueError, match="duplicate"):
        runner.resolve_document_ids(["000001", "x:000001"], {"x:000001"})


def test_plan_is_free_fulltext_and_no_gold(suite, monkeypatch):
    monkeypatch.setattr(runner.RuntimeSettings, "from_env", lambda: pytest.fail("Free plan must not inspect live runtime"))
    value = plan(suite, documents=["000001"])
    spec = value["spec"]
    assert spec["runtime"]["mode"] == "rules_only"
    assert spec["examples"] == [] and spec["gold_in_model_prompt"] is False
    assert spec["total_old_events"] == 3 and spec["excluded_gold_errors"] == 1
    assert runner.rows(suite.root / "batch/inputs.selected.jsonl") == [{"document_id": suite.ids[0], "content": "abcdefgh", "scene": ""}]
    assert not (suite.package / runner.GOLD).exists()


def test_live_freeze_overrides_large_local_output_and_retries(suite):
    value = plan(suite, live=True)
    spec = value["spec"]
    assert spec["runtime"]["runtime"]["max_output_tokens"] == 16384
    assert spec["runtime"]["runtime"]["max_retries"] == 0
    assert spec["max_input_tokens"] == 983616
    assert spec["configuration_sha256"] == runner.fingerprint(spec["runtime"])
    assert "never-log-this" not in json.dumps(value)


def test_candidate_version_excludes_batch_selection(suite):
    one = plan(suite, "one", documents=["000001"])
    two = plan(suite, "two", documents=["000002"])
    assert one["fingerprint"] != two["fingerprint"]
    assert one["spec"]["candidate_version"] == two["spec"]["candidate_version"]


def test_redundancy_review_changes_candidate_version_and_defaults_off(suite):
    disabled = plan(suite, "disabled")
    enabled = plan(suite, "enabled", redundancy_review=True)
    assert disabled["spec"]["redundancy_review"] is False
    assert enabled["spec"]["redundancy_review"] is True
    assert disabled["spec"]["candidate_version"] != enabled["spec"]["candidate_version"]
    assert enabled["spec"]["runtime"]["mode"] == "rules_only"


def test_historical_plan_without_review_flag_keeps_original_identity(suite):
    prior = plan(suite)
    spec = prior["spec"]
    spec.pop("redundancy_review")
    spec.pop("cached_detection_only")
    identity = {key: spec[key] for key in ("source_hashes", "runtime", "detector", "examples", "max_input_tokens")}
    spec["candidate_version"] = runner.fingerprint(identity)
    prior["fingerprint"] = runner.fingerprint(spec)
    runner.write_json(suite.root / "batch/plan.json", prior)
    assert runner.load_plan(suite.root / "batch") == prior


def test_review_default_options_preserve_r11_identity_and_omit_new_spec_fields(suite):
    implicit = plan(suite, "implicit", redundancy_review=True)
    explicit = plan(suite, "explicit", redundancy_review=True,
                    redundancy_review_policy="actions_v1", redundancy_review_hide_reason=False)
    for made in (implicit, explicit):
        spec = made["spec"]
        assert "redundancy_review_policy" not in spec and "redundancy_review_include_reason" not in spec
        previous_identity = {key: spec[key] for key in
            ("source_hashes", "runtime", "detector", "examples", "max_input_tokens", "redundancy_review")}
        assert spec["candidate_version"] == runner.fingerprint(previous_identity)
    assert implicit["spec"]["candidate_version"] == explicit["spec"]["candidate_version"]
    assert runner.load_plan(suite.root / "implicit") == implicit


@pytest.mark.parametrize("policy,hide_reason", [("context_v2", False), ("actions_v1", True), ("context_v2", True)])
def test_nondefault_review_options_are_frozen_and_change_candidate_version(suite, policy, hide_reason):
    base = plan(suite, "base", redundancy_review=True)
    changed = plan(suite, "changed", redundancy_review=True,
                   redundancy_review_policy=policy, redundancy_review_hide_reason=hide_reason)
    spec = changed["spec"]
    assert spec["candidate_version"] != base["spec"]["candidate_version"]
    assert ("redundancy_review_policy" in spec) == (policy != "actions_v1")
    assert ("redundancy_review_include_reason" in spec) == hide_reason
    assert runner.redundancy_review_options(spec) == (policy, not hide_reason)
    assert runner.load_plan(suite.root / "changed") == changed


@pytest.mark.parametrize("options,reason", [
    ({"redundancy_review_policy": "context_v2"}, "nondefault_review_options_require"),
    ({"redundancy_review_hide_reason": True}, "nondefault_review_options_require"),
    ({"redundancy_review_policy": "unknown"}, "invalid_redundancy_review_policy"),
    ({"redundancy_review_hide_reason": 1}, "invalid_redundancy_review_hide_reason"),
])
def test_review_plan_options_reject_invalid_or_disabled_experiments(suite, options, reason):
    with pytest.raises(ValueError, match=reason):
        plan(suite, **options)
    assert not (suite.root / "batch").exists()


@pytest.mark.parametrize("field,value", [("redundancy_review_policy", "unknown"),
                                        ("redundancy_review_include_reason", 1),
                                        ("redundancy_review_include_reason", None)])
def test_candidate_version_validates_review_options(suite, field, value):
    spec = plan(suite)["spec"]
    spec[field] = value
    with pytest.raises(ValueError, match="invalid_redundancy_review"):
        runner.candidate_version(spec)


@pytest.mark.parametrize("policy,hide_reason", [("actions_v1", False), ("context_v2", False),
                                              ("actions_v1", True), ("context_v2", True)])
def test_run_forwards_frozen_review_policy_and_reason_visibility(suite, monkeypatch, policy, hide_reason):
    sid = suite.ids[0]
    plan(suite, live=True, documents=[sid], redundancy_review=True,
         redundancy_review_policy=policy, redundancy_review_hide_reason=hide_reason)
    monkeypatch.setattr(runner, "BudgetedChatClient", lambda *args: object())
    seen = []
    def detect(content, **kwargs):
        seen.append((kwargs["redundancy_review"], kwargs["redundancy_review_policy"],
                     kwargs["redundancy_review_include_reason"]))
        return {"document_id": sid, "errors": [], "traces": [], "coverage": {"execution_complete": True}}
    monkeypatch.setattr(runner, "detect_text", detect)
    runner.run_batch(Namespace(out=suite.root / "batch", live=True))
    assert seen == [(True, policy, not hide_reason)]


def test_cli_parses_explicit_review_experiment_options(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(runner, "create_plan", seen.append)
    assert runner.main(["plan", "--out", str(tmp_path / "unused"), "--redundancy-review",
                        "--redundancy-review-policy", "context_v2", "--redundancy-review-hide-reason"]) == 0
    assert seen[0].redundancy_review is True
    assert seen[0].redundancy_review_policy == "context_v2" and seen[0].redundancy_review_hide_reason is True


@pytest.mark.parametrize("options", [
    {"redundancy_review": True, "live": True},
    {"redundancy_review": False, "live": True, "reuse_model_responses_from": [Path("unused")]},
    {"redundancy_review": True, "live": False, "reuse_model_responses_from": [Path("unused")]},
])
def test_cached_detection_only_requires_review_live_and_explicit_source(suite, options):
    with pytest.raises(ValueError, match="requires_live_redundancy_review_and_response_sources"):
        plan(suite, cached_detection_only=True, **options)


def test_cached_detection_guard_reuses_detect_but_only_review_miss_can_call(suite):
    prior, messages, key, _ = cached_response(suite)
    _, records = runner.prepare_known_response_reuse([suite.root / "source-run"], prior["spec"]["runtime"],
                                                      suite.root / "next", prior["spec"]["max_input_tokens"])
    runner.import_model_responses(suite.root / "next/model_responses", records)
    called = []
    def raw(messages, *, purpose):
        called.append(purpose)
        return {"content": '{"decisions":[]}', "trace": {"purpose": purpose,
                "request_sha256": request_messages_hash(messages), "call_id": "one-new-review", "status": "ok", "cost_cny": "0.2"}}
    client = runner.SharedCandidateClient(runner.review_only_uncached_client(raw), suite.root / "next/model_responses")
    detected = client(messages, purpose="text_review.detect")
    assert detected["trace"]["reused_from_prior_run"]
    for purpose in ("text_review.detect", "text_review.global", "unknown-purpose"):
        with pytest.raises(RuntimeError, match="cached_detection_only_cache_miss"):
            client([{"role": "user", "content": "cache miss"}], purpose=purpose)
    assert called == []
    # Even identical messages cannot collide across distinct purposes.
    assert key != request_cache_key(messages, runner.REDUNDANCY_REVIEW_PURPOSE)
    first = client(messages, purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    second = client(messages, purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    assert called == [runner.REDUNDANCY_REVIEW_PURPOSE]
    assert first["content"] == second["content"] and second["trace"]["reused_response"]
    calls = runner.runtime_calls({"traces": [{"runtime_trace": reply["trace"]} for reply in (detected, first, second)]})
    assert len(calls) == 2
    assert sum(runner.Decimal(c["cost_cny"]) for c in calls.values()) == runner.Decimal("0.3")
    assert sum(runner.Decimal(c["cost_cny"]) for c in calls.values() if not c.get("reused_from_prior_run")) == runner.Decimal("0.2")


def test_cached_policy_freezes_imports_and_replay_only_wins(suite, monkeypatch):
    _, messages, key, _ = cached_response(suite)
    made = plan(suite, "review-replay", live=True, documents=[suite.ids[0]], redundancy_review=True,
                cached_detection_only=True, reuse_model_responses_from=[suite.root / "source-run"], replay_only=True)
    assert made["spec"]["cached_detection_only"] is True
    assert key in made["spec"]["imported_response_hashes"]
    assert runner.load_plan(suite.root / "review-replay") == made
    seen = []
    def detect(content, **kwargs):
        assert kwargs["redundancy_review"] is True
        seen.append(kwargs["chat"](messages, purpose="text_review.detect"))
        kwargs["chat"](messages, purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
        pytest.fail("replay-only must reject an uncached review")
    monkeypatch.setattr(runner, "detect_text", detect)
    # The fixture makes construction of a paid client fail immediately.
    state = runner.run_batch(Namespace(out=suite.root / "review-replay", live=True))
    assert len(seen) == 1 and seen[0]["trace"]["reused_from_prior_run"]
    receipt = runner.read(suite.root / "review-replay" / state["reports"][suite.ids[0]]["path"])
    assert receipt["report"]["coverage"]["execution_complete"] is False
    assert receipt["report"]["coverage"]["failure_type"] == "RuntimeError"


def test_run_passes_frozen_review_flag_and_failed_review_cannot_earn_recovery(suite, monkeypatch):
    sid = suite.ids[0]
    plan(suite, live=True, documents=[sid], redundancy_review=True)
    monkeypatch.setattr(runner, "BudgetedChatClient", lambda *args: object())
    seen = []
    def detect(content, **kwargs):
        seen.append(kwargs["redundancy_review"])
        return {"document_id": sid, "errors": [error()], "coverage": {"execution_complete": False},
                "redundancy_review": {"status": "failed"}, "traces": []}
    monkeypatch.setattr(runner, "detect_text", detect)
    args = Namespace(out=suite.root / "batch", live=True)
    state = runner.run_batch(args)
    runner.run_batch(args)
    assert seen == [True]
    record = runner.read(args.out / state["reports"][sid]["path"])
    assert record["report"]["errors"] == [error()]
    assert not runner.complete_live_record(record)
    _, _, score = runner.repair_events(suite.events, suite.inputs,
        [{"document_id": sid, "result": "TP", "gold_index": 0, "prediction": error()}], {sid: record})
    assert score["recovered_events"] == 0 and score["total_old_events"] == 3


def test_rules_only_calls_real_entry_contract_and_resumes_without_retry(suite, monkeypatch):
    plan(suite)
    seen = []
    def detect(content, **kwargs):
        assert kwargs["chat"] is None and kwargs["examples"] == () and kwargs["detector"] == "hybrid"
        seen.append((content, kwargs["document_id"]))
        if len(seen) == 2:
            raise RuntimeError("synthetic failure")
        return {"document_id": kwargs["document_id"], "errors": [], "coverage": {"rules_complete": True, "execution_complete": False}}
    monkeypatch.setattr(runner, "detect_text", detect)
    args = Namespace(out=suite.root / "batch", live=False)
    first = runner.run_batch(args)
    second = runner.run_batch(args)
    assert first["all_selected_attempted"] and second["all_selected_attempted"]
    assert seen == [(suite.inputs[sid]["content"], sid) for sid in suite.ids]
    failed = runner.read(args.out / first["reports"][suite.ids[1]]["path"])
    assert failed["report"]["coverage"]["failure_type"] == "RuntimeError"
    assert failed["report"]["coverage"]["execution_complete"] is False


@pytest.mark.parametrize("planned_live,flag", [(True, False), (False, True)])
def test_live_requires_both_plan_and_run_flag(suite, planned_live, flag):
    plan(suite, live=planned_live)
    with pytest.raises(ValueError, match="both_frozen_plan"):
        runner.run_batch(Namespace(out=suite.root / "batch", live=flag))


def test_changed_queue_or_code_cannot_resume(suite, monkeypatch):
    plan(suite)
    monkeypatch.setattr(runner, "source_hashes", lambda: {"changed": "code"})
    with pytest.raises(ValueError, match="runtime_or_source_changed"):
        runner.run_batch(Namespace(out=suite.root / "batch", live=False))
    (suite.root / "batch/inputs.selected.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selected_input_changed"):
        runner.load_plan(suite.root / "batch")


def test_only_original_strict_containment_and_exclusions_apply(suite):
    gold = {"document_id": "d", "errors": [{"type": "A", "spans": [error(text="bc")["spans"][0], error(text="ef", start=4)["spans"][0]]},
                                             {"type": "A", "spans": [], "scorable": False}]}
    partial = {"document_id": "d", "errors": [error()]}
    cases = runner.cases_for_document(suite.scorer, "d", partial, gold)
    assert [c["result"] for c in cases] == ["FP", "FN"]
    complete = {"document_id": "d", "errors": [error(text="abcdef", start=0), error(text="abcdef", start=0)]}
    cases = runner.cases_for_document(suite.scorer, "d", complete, gold)
    assert [c["result"] for c in cases] == ["TP", "FP"]
    assert cases[0]["gold_index"] == 0


def test_unrun_failed_and_rules_only_keep_all_event_denominators(suite):
    for records in ({}, {suite.ids[0]: live_record(complete=False)}, {suite.ids[0]: {**live_record(), "mode": "rules_only"}}):
        _, _, score = runner.repair_events(suite.events, suite.inputs, [], records)
        assert score["total_old_events"] == 3 and score["recovered_events"] == 0
        assert score["old_FP_elimination"]["denominator"] == 1
        assert score["old_FN_recovery"]["denominator"] == 2
        assert not score["benchmark_target95_reached"]


@pytest.mark.parametrize("candidate,status", [(error("B", "abc", 0, reason="renamed"), "persists"),
                                               (error("C"), "persists"),
                                               (error("C", "abc", 0), "unresolved_ambiguous_relation"),
                                               ({"error_type": "B", "spans": []}, "unresolved_current_anchor")])
def test_fp_identity_or_type_edits_cannot_game_elimination(suite, candidate, status):
    event = next(e for e in suite.events if e["result"] == "FP")
    cases = [{"document_id": suite.ids[0], "result": "FP", "prediction": candidate}]
    events, _, score = runner.repair_events([event], suite.inputs, cases, {suite.ids[0]: live_record()})
    assert events[0]["status"] == status
    assert score["recovered_events"] == 0


def test_unanchorable_old_fp_is_not_silently_fixed(suite):
    event = {**next(e for e in suite.events if e["result"] == "FP"), "prediction": {"error_type": "B", "spans": []}}
    events, _, _ = runner.repair_events([event], suite.inputs, [], {suite.ids[0]: live_record()})
    assert events[0]["status"] == "unresolved_old_anchor"


def test_double_event_counting_new_fp_side_effect_and_business_pending(suite):
    sid = suite.ids[0]
    cases = [{"document_id": sid, "result": "TP", "gold_index": 0, "prediction": error()},
             {"document_id": sid, "result": "FP", "prediction": error("C", "gh", 6)}]
    events, sides, score = runner.repair_events(suite.events, suite.inputs, cases, {sid: live_record()})
    assert score["recovered_events"] == 2 and score["total_old_events"] == 3
    assert score["recovery_rate"] == 2 / 3
    assert sides[0]["side_effect"] == "new_or_unresolved_FP"
    assert all(e["business_repair_pass"] is False for e in events)
    assert score["business_repair_pending_adjudication"] == 3


def test_one_old_broad_quote_cannot_hide_two_current_allegations(suite):
    sid = suite.ids[0]
    cases = [{"document_id": sid, "result": "FP", "prediction": error("B", "abc", 0, reason=reason)}
             for reason in ("First terminology issue", "Second independent terminology issue")]
    events, sides, score = runner.repair_events(suite.events, suite.inputs, cases, {sid: live_record()})
    assert next(e for e in events if e["result"] == "FP")["status"] == "persists"
    assert len(sides) == 2 and all(s["side_effect"] == "new_or_unresolved_FP" for s in sides)
    assert all(s["linkage_status"] == "ambiguous_one_old_fp_to_multiple_current_candidates" for s in sides)


def test_completed_predictions_with_tampered_bytes_refuse_resume(suite, monkeypatch):
    plan(suite, documents=["000001"])
    monkeypatch.setattr(runner, "detect_text", lambda text, **kw: {"document_id": kw["document_id"], "errors": [], "coverage": {}})
    args = Namespace(out=suite.root / "batch", live=False)
    state = runner.run_batch(args)
    (args.out / state["reports"][suite.ids[0]]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed_prediction_changed"):
        runner.run_batch(args)


def test_analyze_subset_before_after_tp_regression_and_no_false95(suite, monkeypatch):
    plan(suite, live=True, documents=["000001"])
    sid = suite.ids[0]
    install_fake_live(monkeypatch, {sid: live_record(sid, [error()])["report"]})
    runner.run_batch(Namespace(out=suite.root / "batch", live=True))
    report = runner.analyze(Namespace(package=suite.package, runs=[suite.root / "batch"], out=suite.root / "analysis"))
    assert report["strict_attempted_documents_before"]["true_positive"] == 1
    assert report["strict_attempted_documents_after"]["true_positive"] == 1
    assert report["strict_attempted_documents_before"]["false_positive"] == 1
    assert report["strict_attempted_documents_after"]["false_positive"] == 0
    assert report["old_TP_regression_gold_ids"] == [f"{sid}:gold:1"]
    assert report["strict_all_hints_all_442"]["excluded_gold_errors"] == 1
    assert report["benchmark_event_recovery"]["recovered_events"] == 2
    assert not report["benchmark_event_recovery"]["benchmark_target95_reached"]
    assert report["business_repair"]["pass"] is False


def test_same_version_batches_may_cover95_but_mixed_versions_cannot(suite, monkeypatch):
    for number, sid in enumerate(suite.ids):
        directory = f"b{number}"
        plan(suite, directory, live=True, documents=[sid])
        correct = [error() if number == 0 else error("A", "jk", 1)]
        if number == 0:
            correct.append(error("A", "ef", 4))
        install_fake_live(monkeypatch, {sid: live_record(sid, correct)["report"]})
        runner.run_batch(Namespace(out=suite.root / directory, live=True))
    output = runner.analyze(Namespace(package=suite.package, runs=[suite.root / "b0", suite.root / "b1"], out=suite.root / "complete"))
    assert output["mixed_iteration_fingerprints"] is True and output["mixed_candidate_versions"] is False
    assert output["benchmark_event_recovery"]["benchmark_target95_reached"] is True
    monkeypatch.setattr(runner.RuntimeSettings, "from_env", lambda: replace(suite.settings, context_tokens=900000))
    plan(suite, "b2", live=True, documents=[suite.ids[1]])
    runner.run_batch(Namespace(out=suite.root / "b2", live=True))
    output = runner.analyze(Namespace(package=suite.package, runs=[suite.root / "b0", suite.root / "b2"], out=suite.root / "mixed"))
    assert output["benchmark_event_recovery"]["recovery_rate"] == 1
    assert output["mixed_candidate_versions"] is True
    assert output["benchmark_event_recovery"]["benchmark_target95_reached"] is False


def test_later_failed_attempt_cannot_retain_earlier_best_score(suite, monkeypatch):
    sid = suite.ids[0]
    for name, complete in (("good", True), ("failed", False)):
        plan(suite, name, live=True, documents=[sid])
        install_fake_live(monkeypatch, {sid: live_record(sid, [error()], complete=complete)["report"]})
        runner.run_batch(Namespace(out=suite.root / name, live=True))
    report = runner.analyze(Namespace(package=suite.package, runs=[suite.root / "good", suite.root / "failed"], out=suite.root / "latest"))
    assert report["benchmark_event_recovery"]["recovered_events"] == 0
    assert report["live_complete_documents"] == 0
    assert report["benchmark_event_recovery"]["attempted_batch_events"] == 2
    assert report["benchmark_event_recovery"]["completed_batch_events"] == 0
    assert report["benchmark_event_recovery"]["attempted_batch_incomplete_events"] == 2
    assert report["benchmark_event_recovery"]["attempted_batch_recovery_rate"] == 0
    assert report["benchmark_event_recovery"]["completed_batch_recovery_rate"] is None


def test_existing_ledger_readonly_and_never_reset_or_create(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.sqlite3"
    monkeypatch.setattr(runner, "DEFAULT_LEDGER", ledger)
    settings = runner.RuntimeSettings(ledger=ledger, budget_cny="100")
    with pytest.raises(ValueError, match="existing_shared"):
        runner.shared_budget(settings)
    assert not ledger.exists()
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE budget(singleton INTEGER, limit_micro INTEGER)")
        db.execute("INSERT INTO budget VALUES(1,100000000)")
        db.execute("CREATE TABLE calls(charge_micro INTEGER)")
        db.execute("INSERT INTO calls VALUES(29000000)")
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    result = runner.shared_budget(settings)
    assert result["accounted_cny"] == "29" and result["calls"] == 1
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
    for invalid in (replace(settings, budget_cny="20"), replace(settings, budget_cny="101"), replace(settings, ledger=tmp_path / "new.sqlite3")):
        with pytest.raises(ValueError):
            runner.shared_budget(invalid)
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before


def test_admission_subprocess_pins_utf8_for_windows_chinese_metadata(monkeypatch):
    receipt = {"role": runner.ROLE, "manifest_sha256": runner.PUBLISHED_MANIFEST_SHA256, "label": "数值不一致错误"}
    def invoke(command, **kwargs):
        assert command[1:3] == ["-X", "utf8"] and kwargs["encoding"] == "utf-8"
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt, ensure_ascii=False))
    monkeypatch.setattr(runner.subprocess, "run", invoke)
    assert runner.admission(runner.PACKAGE) == receipt


# Raw-only authentication tests: original pipeline failure never becomes complete.
def failed_protocol_response(suite, *, failure_code="invalid_source_evidence", schema="redundancy-review/2.0", hide_reason=False):
    sid = suite.ids[0]
    prior, messages, key, record = cached_response(suite, content='{"decisions":[]}',
        purpose=runner.REDUNDANCY_REVIEW_PURPOSE, review_policy="context_v2", hide_reason=hide_reason, documents=[sid])
    record["trace"]["finish_reason"] = "stop"
    runner.write_json(suite.root / "source-run/model_responses" / (key + ".json"), record)
    state_path = suite.root / "source-run/state.json"
    state = runner.read(state_path)
    state["all_selected_attempted"] = True
    receipt_path = suite.root / "source-run" / state["reports"][sid]["path"]
    receipt = runner.read(receipt_path)
    report = receipt["report"]
    report["coverage"].update(execution_complete=False, complete=False, model_complete=False)
    report["errors"] = [error()]
    report["model_candidate_representation"] = [{"candidate_ref":"base:0","error_id":"e"}]
    stage = report["redundancy_review"]
    stage.update(schema_version=schema, status="incomplete", complete=False, model_ran=True,
        source_content_sha256=receipt["source_content_sha256"], base_execution_complete=True,
        base_coverage={"execution_complete":True,"model_complete":True}, decisions=[],withdrawn=[],
        request_sha256=record["trace"]["request_sha256"], runtime_trace=record["trace"],
        failure={"code":failure_code,"error_type":"_Invalid"})
    base_trace = {**record["trace"],"purpose":"text_review.detect","call_id":"separate-base-call"}
    report["traces"] = [
        {"job_index":0,"status":"ok","purpose":"text_review.detect","runtime_trace":base_trace},
        {"job_index":1,"status":"failed","purpose":runner.REDUNDANCY_REVIEW_PURPOSE,
         "failure_code":failure_code,"runtime_trace":record["trace"]}]
    runner.write_json(receipt_path,receipt)
    state["reports"][sid]["sha256"] = runner.sha(receipt_path)
    runner.write_json(state_path,state)
    return prior,messages,key,record,receipt_path,state_path


@pytest.mark.parametrize("failure",["invalid_source_evidence","invalid_normal_context"])
@pytest.mark.parametrize("schema",["redundancy-review/2.0","redundancy-review/2.1"])
@pytest.mark.parametrize("hide_reason",[False,True])
def test_protocol_failure_authenticates_only_raw_bytes_not_pipeline_recovery(suite,failure,schema,hide_reason):
    prior,messages,key,record,path,state_path = failed_protocol_response(suite,failure_code=failure,schema=schema,hide_reason=hide_reason)
    before=path.read_bytes()
    _,records=runner.prepare_known_response_reuse([suite.root/"source-run"],prior["spec"]["runtime"],suite.root/"next",prior["spec"]["max_input_tokens"])
    proof=records[key]["cache_provenance"]["source_response_integrity"]
    assert proof["basis"] == "failed_context_protocol_api_response_and_trace"
    assert proof["source_pipeline_complete"] is False and proof["authentication_scope"] == "raw-only-authenticated"
    assert proof["source_failure"] == {"code":failure,"error_type":"_Invalid"}
    assert proof["response_bytes_authenticated"] is True
    assert records[key]["content"] == record["content"] and records[key]["trace"] == record["trace"]
    assert not runner.complete_live_record(runner.read(path)) and path.read_bytes() == before
    runner.import_model_responses(suite.root/"next/model_responses",records)
    client=runner.SharedCandidateClient(runner.refuse_uncached_request,suite.root/"next/model_responses")
    reply=client(messages,purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    assert reply["trace"]["reused_from_prior_run"] and reply["trace"]["cost_cny"] == "0.1"
    with pytest.raises(RuntimeError,match="no_network_request"):
        client([{"role":"user","content":"different v2 request"}],purpose=runner.REDUNDANCY_REVIEW_PURPOSE)
    calls=runner.runtime_calls({"traces":[{"runtime_trace":reply["trace"]}]})
    assert sum(runner.Decimal(c["cost_cny"]) for c in calls.values()) == runner.Decimal("0.1")
    assert sum(runner.Decimal(c["cost_cny"]) for c in calls.values() if not c.get("reused_from_prior_run")) == 0


@pytest.mark.parametrize("mutation",["state_hash","not_terminal","base_incomplete","base_model_incomplete","base_job_failed",
    "unknown_failure","wrong_exception","wrong_schema","wrong_policy","wrong_visibility","missing_visibility",
    "coverage_complete","decisions","withdrawn","duplicate_review","job_failure_code","job_index",
    "raw_hash","request_sha","source_sha","stage_trace","finish_length","finish_missing","content_incomplete","nan_cost"])
def test_failed_protocol_receipt_rejects_nonprotocol_or_unbound_sources(suite,mutation):
    prior,_,_,_,path,state_path=failed_protocol_response(suite)
    receipt=runner.read(path);state=runner.read(state_path);report=receipt["report"];stage=report["redundancy_review"]
    if mutation == "state_hash": stage["raw_response"] += "tampered"
    elif mutation == "not_terminal": state["all_selected_attempted"] = False
    elif mutation == "base_incomplete": stage["base_coverage"]["execution_complete"] = False
    elif mutation == "base_model_incomplete": stage["base_coverage"]["model_complete"] = False
    elif mutation == "base_job_failed": report["traces"][0]["status"] = "failed"
    elif mutation == "unknown_failure": stage["failure"]["code"] = "model_call_or_review_failed"
    elif mutation == "wrong_exception": stage["failure"]["error_type"] = "TimeoutError"
    elif mutation == "wrong_schema": stage["schema_version"] = "redundancy-review/9.0"
    elif mutation == "wrong_policy": stage["policy"] = "actions_v1"
    elif mutation == "wrong_visibility": stage["include_reason"] = False
    elif mutation == "missing_visibility": del stage["include_reason"]
    elif mutation == "coverage_complete": report["coverage"]["model_complete"] = True
    elif mutation in {"decisions","withdrawn"}: stage[mutation] = [{}]
    elif mutation == "duplicate_review": report["traces"].append(report["traces"][1])
    elif mutation == "job_failure_code": report["traces"][1]["failure_code"] = "wrong"
    elif mutation == "job_index": report["traces"][1]["job_index"] = 0
    elif mutation == "raw_hash": stage["response_sha256"] = "0"*64
    elif mutation == "request_sha": stage["request_sha256"] = "0"*64
    elif mutation == "source_sha": stage["source_content_sha256"] = "0"*64
    elif mutation == "stage_trace": stage["runtime_trace"] = {**stage["runtime_trace"],"cost_cny":"0"}
    else:
        trace=stage["runtime_trace"]
        if mutation == "finish_length": trace["finish_reason"] = "length"
        elif mutation == "finish_missing": trace.pop("finish_reason")
        elif mutation == "content_incomplete": trace["response_content_incomplete"] = True
        elif mutation == "nan_cost": trace["cost_cny"] = "NaN"
        report["traces"][1]["runtime_trace"] = trace
    runner.write_json(path,receipt)
    if mutation != "state_hash": state["reports"][suite.ids[0]]["sha256"] = runner.sha(path)
    runner.write_json(state_path,state)
    with pytest.raises(ValueError,match="response_source_(completed_receipt_changed|raw_receipt_mismatch)"):
        runner.prepare_known_response_reuse([suite.root/"source-run"],prior["spec"]["runtime"],suite.root/"next",prior["spec"]["max_input_tokens"])


@pytest.mark.parametrize("mutation",["content","trace_cost","trace_status","content_length"])
def test_failed_protocol_raw_cache_cannot_forge_frozen_success_trace(suite,mutation):
    prior,_,key,record,_,_=failed_protocol_response(suite)
    if mutation == "content": record["content"] = '{"decisions":[{}]}'
    elif mutation == "trace_cost": record["trace"]["cost_cny"] = "0"
    elif mutation == "trace_status": record["trace"]["status"] = "pending"
    else: record["trace"]["finish_reason"] = "length"
    runner.write_json(suite.root/"source-run/model_responses"/(key+".json"),record)
    with pytest.raises(ValueError,match="response_source_raw_receipt_mismatch|response_trace_does_not_match_source_plan"):
        runner.prepare_known_response_reuse([suite.root/"source-run"],prior["spec"]["runtime"],suite.root/"next",prior["spec"]["max_input_tokens"])


def test_completed_context_schema21_receipt_still_authenticates_exact_response(suite):
    prior,_,key,record=cached_response(suite,content='{"decisions":[]}',purpose=runner.REDUNDANCY_REVIEW_PURPOSE,review_policy="context_v2")
    state_path=suite.root/"source-run/state.json";state=runner.read(state_path)
    path=suite.root/"source-run"/state["reports"][suite.ids[0]]["path"]
    receipt=runner.read(path);receipt["report"]["redundancy_review"]["schema_version"]="redundancy-review/2.1"
    runner.write_json(path,receipt);state["reports"][suite.ids[0]]["sha256"]=runner.sha(path);runner.write_json(state_path,state)
    _,records=runner.prepare_known_response_reuse([suite.root/"source-run"],prior["spec"]["runtime"],suite.root/"next",prior["spec"]["max_input_tokens"])
    assert records[key]["content"] == record["content"]
    assert records[key]["cache_provenance"]["source_response_integrity"]["basis"] == "completed_receipt_redundancy_response_and_trace"
