"""Boundary tests only; no benchmark samples, paid requests or real services."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evals import run_active_suite as runner
from evals.benchmark_retrieval import ExcerptStore, digest


PAYLOAD = {"instruction": "What was Acme revenue?", "claim": "12 USD", "evidence_text": "Acme revenue was 12 USD."}


class NoCalls:
    def __call__(self, *args, **kwargs):
        raise AssertionError("This boundary must not call a model")


class Store:
    def search(self, query, limit):
        return [{"evidence_id": "found-id", "source_id": "excerpt-id", "text": PAYLOAD["evidence_text"]}]

    def read(self, hit):
        return {"source_id": "excerpt-id", "text": PAYLOAD["evidence_text"], "verified": True}


def test_empty_evidence_is_abstention_not_a_clean_decision():
    session = runner.ClaimSession(PAYLOAD, NoCalls(), Store())
    view = session("detect_document", {})
    assert session.result["has_error"] is None
    assert view["status"] == "needs_review"
    assert view["business_status"] == "needs_review"


def test_pi_business_tool_has_runtime_required_result_reference():
    session = runner.ClaimSession(PAYLOAD, NoCalls(), Store())
    view = session("detect_document", {})
    assert isinstance(view.get("result_ref"), str) and view["result_ref"]


def test_pi_read_requires_search_and_recheck_requires_read():
    session = runner.ClaimSession(PAYLOAD, NoCalls(), Store())
    with pytest.raises(ValueError, match="detect"):
        session("search_evidence", {"query": "Acme"})
    session("detect_document", {})
    with pytest.raises(ValueError, match="prior_search"):
        session("read_evidence", {"evidence_id": "invented"})
    session("search_evidence", {"query": "Acme"})
    with pytest.raises(ValueError, match="verified_reads"):
        session("recheck", {"evidence_ids": ["found-id"]})
    assert session("read_evidence", {"evidence_id": "found-id"})["verified"] is True


def test_pi_call_uses_official_task_object_and_failed_runtime_abstains():
    row = {"document_id": "opaque", **PAYLOAD}
    candidate = {"has_error": False, "label": "supported", "status": "needs_review", "reason": "Agrees"}

    def fake_pi(task, handler, *args, **kwargs):
        assert isinstance(task, dict)
        assert task.get("documentId") == "bound-claim"
        assert isinstance(task.get("question"), str)
        handler.result = dict(candidate)
        return {"status": "stopped", "stop_reason": "timeout", "model_traces": []}

    with patch("yjcheck.model_runtime.BudgetedChatClient", return_value=NoCalls()), patch("yjcheck.pi_bridge.run_pi", side_effect=fake_pi):
        result = runner.run_one(runner.CLAIM, "pi_provided", row, None, None)["result"]
    assert result["has_error"] is None
    assert result.get("status") == "failed" or result.get("agent", {}).get("status") == "stopped"


def test_direct_run_function_rejects_reserved_before_reading_its_inputs(tmp_path):
    args = Namespace(suite=tmp_path / "suite", out=tmp_path / "output", track=runner.CLAIM,
                     role="final_candidate_reserved", limit=0, arms="direct", store=None, workers=1)
    with patch.object(runner, "gate", return_value={"status": "passed"}), patch.object(runner, "rows", side_effect=AssertionError("Reserved inputs were opened")):
        with pytest.raises(ValueError):
            runner.run(args)


def test_legacy_schema_is_rejected_by_admission(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"schema_version": "fined/1.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="admission_failed"):
        runner.gate(tmp_path)


def test_pi_result_retains_all_node_calls_and_orchestration_costs():
    class Chat:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return {"content": "{}", "trace": {"call_id": f"node-{self.calls}", "cost_cny": "0.01"}}

    def fake_pi(task, handler, *args, **kwargs):
        handler.client([], purpose="first")
        handler.client([], purpose="second")
        handler.result = {"status": "needs_review", "has_error": None, "label": "insufficient"}
        return {"status": "completed", "model_traces": [{"call_id": "pi-1", "cost_cny": "0.02"}]}

    with patch("yjcheck.model_runtime.BudgetedChatClient", return_value=Chat()), patch("yjcheck.pi_bridge.run_pi", side_effect=fake_pi):
        result = runner.run_one(runner.CLAIM, "pi_provided", {"document_id": "opaque", **PAYLOAD}, None, None)["result"]
    assert {trace["call_id"] for trace in result["model_traces"]} == {"node-1", "node-2", "pi-1"}


def test_run_never_reads_gold_and_costs_are_own_unique_calls(tmp_path):
    from yjcheck.model import ModelConfig
    from yjcheck.model_runtime import RuntimeSettings, BudgetLedger
    suite, out = tmp_path / "suite", tmp_path / "out"
    (suite / runner.CLAIM).mkdir(parents=True)
    (suite / "MANIFEST.json").write_text("{}", encoding="utf-8")
    input_path = suite / runner.CLAIM / "inputs.development.jsonl"
    input_path.write_text(json.dumps({"document_id": "opaque", **PAYLOAD}) + "\n", encoding="utf-8")
    settings = RuntimeSettings(ledger=tmp_path / "isolated-test-ledger.sqlite3")
    ledger = BudgetLedger(settings.ledger, settings.budget_cny)

    def other_activity(*args):
        cid = ledger.reserve("0.1", {"purpose": "independent-activity"})
        ledger.settle(cid, "0.1", {"status": "ok"})
        return {"document_id": "opaque", "arm": "direct", "seconds": 0,
                "result": {"has_error": None, "model_traces": [
                    {"call_id": "own-a", "cost_cny": "0.02"}, {"call_id": "own-a", "cost_cny": "0.02"}]}}

    original_rows = runner.rows

    def inputs_only(path):
        assert Path(path) == input_path
        return original_rows(path)

    args = Namespace(suite=suite, out=out, track=runner.CLAIM, role="development", limit=0,
                     arms="direct", store=None, workers=1)
    with patch.object(runner, "gate", return_value={"status": "passed"}), patch.object(runner, "rows", side_effect=inputs_only), \
            patch.object(runner, "run_one", side_effect=other_activity), \
            patch.object(ModelConfig, "from_env", return_value=ModelConfig("http://localhost:8000/v1", "offline")), \
            patch.object(RuntimeSettings, "from_env", return_value=settings), redirect_stdout(io.StringIO()):
        runner.run(args)
    done = json.loads((out / "completion.json").read_text(encoding="utf-8"))
    assert done["new_model_calls"] == 1
    assert done["new_accounted_cny"] == "0.02"
    assert done["budget_after"]["accounted_cny"] == 0.1


def experiment(tmp_path, truth, predictions):
    suite, out = tmp_path / "suite", tmp_path / "experiment"
    (suite / runner.CLAIM).mkdir(parents=True)
    out.mkdir()
    (suite / "MANIFEST.json").write_text("{}", encoding="utf-8")
    inputs = [{"document_id": key, **PAYLOAD} for key in truth]
    gold = [{"document_id": key, "has_error": value} for key, value in truth.items()]
    for name, values in (("inputs", inputs), ("gold", gold)):
        (suite / runner.CLAIM / f"{name}.development.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
    (out / "predictions.jsonl").write_text("".join(json.dumps({"document_id": key, "arm": "direct", "seconds": 0,
                                                              "result": value}) + "\n" for key, value in predictions.items()), encoding="utf-8")
    runner.save(out / "plan.json", {"suite": str(suite), "track": runner.CLAIM, "role": "development",
                                   "old_dataset_used": False, "suite_sha256": runner.sha(suite / "MANIFEST.json"),
                                   "input_sha256": runner.sha(suite / runner.CLAIM / "inputs.development.jsonl"),
                                   "document_ids": list(truth), "arms": ["direct"]})
    runner.save(out / "completion.json", {"predictions_sha256": runner.sha(out / "predictions.jsonl"), "code_unchanged": True,
                                          "new_model_calls": 0, "new_accounted_cny": "0"})
    return out


def scored(out):
    with patch.object(runner, "gate", return_value={"status": "passed"}), redirect_stdout(io.StringIO()):
        runner.score(Namespace(out=out))
    return json.loads((out / "score.json").read_text(encoding="utf-8"))["arms"]["direct"]


def test_score_keeps_abstentions_and_missing_predictions_in_denominator(tmp_path):
    out = experiment(tmp_path, {"error-a": True, "error-b": True, "clean-a": False, "clean-b": False},
                     {"error-a": {"has_error": True}, "clean-a": {"has_error": None}})
    result = scored(out)
    assert result["tp"] == 1 and result["fn"] == 1 and result["tn"] == 0
    assert result["abstentions"] == 3 and result["clean_abstentions"] == 2
    assert result["accuracy_full_denominator"] == 0.25
    assert result["decision_coverage"] == 0.25


def test_score_failed_label_is_not_credited(tmp_path):
    out = experiment(tmp_path, {"error-a": True, "clean-a": False},
                     {"error-a": {"status": "failed", "has_error": True},
                      "clean-a": {"status": "failed", "has_error": False}})
    result = scored(out)
    assert result["accuracy_full_denominator"] == 0
    assert result["abstentions"] == 2


def test_score_single_error_pilot_has_no_clean_denominator(tmp_path):
    out = experiment(tmp_path, {"error-a": True}, {"error-a": {"has_error": True}})
    result = scored(out)
    assert result["accuracy_full_denominator"] == 1
    assert result["clean_denominator"] == 0
    assert result["false_positive_rate"] is None


def test_excerpt_source_keeps_original_crlf_bytes(tmp_path):
    text = "First line.\r\nSecond line."
    sid = digest(text)
    (tmp_path / (sid + ".txt")).write_bytes(text.encode("utf-8"))
    store = object.__new__(ExcerptStore)
    store.directory, store.sources = tmp_path, {sid: {"id": sid}}
    assert store.source(sid) == text


def test_excerpt_read_rejects_a_source_outside_the_bound_pool():
    store = object.__new__(ExcerptStore)
    store.bindings = [{"source_id": "allowed", "root_uri": "viking://resources/allowed"}]
    with pytest.raises(ValueError, match="unbound"):
        store.read({"source_id": "other", "uri": "viking://resources/other"})


def test_pi_search_deduplicates_sources_and_bounds_queries():
    class Duplicates(Store):
        calls = 0

        def search(self, query, limit):
            self.calls += 1
            return [{"evidence_id": str(i), "source_id": "same-source", "text": "source"} for i in range(3)]

    store = Duplicates()
    session = runner.ClaimSession(PAYLOAD, NoCalls(), store)
    session("detect_document", {})
    assert len(session("search_evidence", {"query": "Acme"})["hits"]) == 1
    session("search_evidence", {"query": "Acme revenue"})
    blocked = session("search_evidence", {"query": "repeat"})
    assert store.calls == 2
    assert blocked["status"] == "search_limit_reached"
    assert blocked["available_evidence_ids"] == ["0"]


def test_pi_recheck_same_sources_does_not_pay_for_repeated_judgments():
    session = runner.ClaimSession(PAYLOAD, NoCalls(), Store())
    session("detect_document", {})
    session("search_evidence", {"query": "Acme"})
    session("read_evidence", {"evidence_id": "found-id"})
    candidate = {"status": "needs_review", "has_error": None, "label": "insufficient"}
    with patch("yjcheck.claim_verification.verify_claim", return_value=candidate) as verify:
        session("recheck", {"evidence_ids": ["found-id"]})
        session("recheck", {"evidence_ids": ["found-id", "found-id"]})
    assert verify.call_count == 1


def test_score_rejects_changed_frozen_scorer(tmp_path):
    out = experiment(tmp_path, {"error-a": True}, {"error-a": {"has_error": True}})
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan["code_sha256"] = {"evals/dataset_readiness.py": "wrong",
                           "evals/run_active_suite.py": runner.sha(runner.ROOT / "evals/run_active_suite.py")}
    runner.save(out / "plan.json", plan)
    with pytest.raises(ValueError, match="scorer_changed"):
        scored(out)


def test_score_rejects_changed_frozen_inputs_even_if_gate_is_mocked(tmp_path):
    out = experiment(tmp_path, {"error-a": True}, {"error-a": {"has_error": True}})
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan["input_sha256"] = "wrong"
    runner.save(out / "plan.json", plan)
    with pytest.raises(ValueError, match="inputs_changed"):
        scored(out)


def test_score_rejects_plan_change_that_could_shrink_denominator(tmp_path):
    out = experiment(tmp_path, {"error-a": True, "error-b": True}, {"error-a": {"has_error": True}})
    done = json.loads((out / "completion.json").read_text(encoding="utf-8"))
    done["plan_sha256"] = runner.sha(out / "plan.json")
    runner.save(out / "completion.json", done)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan["document_ids"] = ["error-a"]
    runner.save(out / "plan.json", plan)
    with pytest.raises(ValueError, match="plan_changed"):
        scored(out)


def test_score_rejects_unknown_planned_ids(tmp_path):
    out = experiment(tmp_path, {"error-a": True}, {"error-a": {"has_error": True}})
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan["document_ids"].append("not-in-suite")
    runner.save(out / "plan.json", plan)
    with pytest.raises(ValueError, match="ids_do_not_align"):
        scored(out)


def development_bank(tmp_path):
    bank = {"schema_version": "development-examples/1.0", "role": "development", "track": "clfec",
            "weight_training": False, "selected_document_ids": [f"example-{i}" for i in range(4)],
            "examples": [{"input_text": f"Development source {i}", "corrected_text": f"Development correction {i}"} for i in range(4)]}
    path = tmp_path / "development-bank.json"
    path.write_text(json.dumps(bank), encoding="utf-8")
    receipt = {"valid": True, "sha256": runner.sha(path), "example_count": 4, "role": "development", "track": "clfec"}
    return path, bank, receipt


def test_examples_bank_is_verified_in_subprocess_before_reading_exact_bytes(tmp_path):
    path, bank, receipt = development_bank(tmp_path)
    suite = tmp_path / "suite"
    def verifier(command, **kwargs):
        assert command[1] == str(runner.ROOT / "evals/prepare_development_examples.py")
        assert command[-4:] == ["--suite", str(suite.resolve()), "--verify-bank", str(path.resolve())]
        return Namespace(returncode=0, stdout=json.dumps(receipt))
    with patch.object(runner.subprocess, "run", side_effect=verifier), \
            patch.object(runner, "rows", side_effect=AssertionError("gold or suite rows must not be opened")):
        examples, metadata = runner.load_examples_bank(path, suite)
    assert examples == bank["examples"]
    assert metadata["sha256"] == receipt["sha256"]
    assert metadata["selected_document_ids"] == bank["selected_document_ids"]


def test_examples_bank_replacement_after_subprocess_verification_is_rejected(tmp_path):
    path, bank, receipt = development_bank(tmp_path)
    def verifier(*args, **kwargs):
        bank["examples"][0]["corrected_text"] = "Tampered answer after verification"
        path.write_text(json.dumps(bank), encoding="utf-8")
        return Namespace(returncode=0, stdout=json.dumps(receipt))
    with patch.object(runner.subprocess, "run", side_effect=verifier):
        with pytest.raises(ValueError, match="invalid_or_changed"):
            runner.load_examples_bank(path, tmp_path / "suite")


def test_rejected_examples_bank_is_not_opened_by_inference_process(tmp_path):
    missing = tmp_path / "must-not-open.json"
    with patch.object(runner.subprocess, "run", return_value=Namespace(returncode=2, stdout="")):
        with pytest.raises(ValueError, match="admission_failed"):
            runner.load_examples_bank(missing, tmp_path / "suite")


def test_fewshot_arm_routes_only_verified_pairs_to_structured_node(tmp_path):
    _, bank, _ = development_bank(tmp_path)
    captured = {}
    def node(payload, client, **kwargs):
        captured.update(payload=payload, **kwargs)
        return {"corrected_text": payload["input_text"], "status": "needs_review"}
    row = {"document_id": "target", "input_text": "Current report.", "hidden_gold": "not allowed"}
    with patch("yjcheck.model_runtime.BudgetedChatClient", return_value=NoCalls()), \
            patch("yjcheck.correction_review.correct_text", side_effect=node):
        result = runner.run_one("clfec", "structured_fewshot", row, None, None, examples=bank["examples"])
    assert captured == {"payload": {"input_text": "Current report."}, "variant": "structured", "examples": bank["examples"]}
    assert result["arm"] == "structured_fewshot"
    assert result["result"]["status"] == "needs_review"


@pytest.mark.parametrize("use_examples", [False, True])
def test_clfec_run_defaults_or_fewshot_keep_gold_out_and_freeze_bank_metadata(tmp_path, use_examples):
    from yjcheck.model import ModelConfig
    from yjcheck.model_runtime import RuntimeSettings
    suite, out = tmp_path / "suite", tmp_path / "out"
    (suite / "clfec").mkdir(parents=True)
    (suite / "MANIFEST.json").write_text("{}", encoding="utf-8")
    input_path = suite / "clfec/inputs.development.jsonl"
    input_path.write_text(json.dumps({"document_id": "target", "input_text": "Source text"}) + "\n", encoding="utf-8")
    examples = [{"input_text": "Development", "corrected_text": "Corrected development"}]
    metadata = {"sha256": "f" * 64, "example_count": 1, "selected_document_ids": ["development-example"], "role": "development"}
    calls = []
    def node(*args, **kwargs):
        calls.append((args[1], kwargs))
        return {"document_id": "target", "arm": args[1], "seconds": 0,
                "result": {"corrected_text": "Source text", "status": "needs_review", "model_traces": []}}
    original_rows = runner.rows
    def source_only(path):
        assert Path(path) == input_path
        return original_rows(path)
    args = Namespace(suite=suite, out=out, track="clfec", role="development", limit=0,
                     arms="structured_fewshot" if use_examples else None, store=None, workers=1,
                     examples_bank=tmp_path / "bank-only-read-for-fewshot.json")
    loader = (lambda *args: (examples, metadata)) if use_examples else (lambda *args: pytest.fail("baseline must not read examples bank"))
    with patch.object(runner, "gate", return_value={"status": "passed"}), \
            patch.object(runner, "rows", side_effect=source_only), patch.object(runner, "run_one", side_effect=node), \
            patch.object(runner, "load_examples_bank", side_effect=loader), \
            patch.object(ModelConfig, "from_env", return_value=ModelConfig("http://localhost:8000/v1", "offline")), \
            patch.object(RuntimeSettings, "from_env", return_value=RuntimeSettings(ledger=tmp_path / "ledger.sqlite3")), \
            redirect_stdout(io.StringIO()):
        runner.run(args)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["gold_read_by_run"] is False
    assert "evals/prepare_development_examples.py" in plan["code_sha256"]
    if use_examples:
        assert calls == [("structured_fewshot", {"examples": examples})]
        assert plan["examples_bank"] == metadata
        assert plan["optimization"].endswith("development_fewshot")
    else:
        assert sorted(calls) == [("direct", {}), ("structured", {})]
        assert plan["examples_bank"] is None


def test_fewshot_run_rejects_its_own_training_example_ids(tmp_path):
    suite = tmp_path / "suite"
    (suite / "clfec").mkdir(parents=True)
    (suite / "clfec/inputs.development.jsonl").write_text(json.dumps({"document_id": "same-id", "input_text": "Source"}) + "\n", encoding="utf-8")
    args = Namespace(suite=suite, out=tmp_path / "out", track="clfec", role="development", limit=0,
                     arms="structured_fewshot", store=None, workers=1, examples_bank=tmp_path / "bank.json")
    with patch.object(runner, "gate", return_value={"status": "passed"}), \
            patch.object(runner, "load_examples_bank", return_value=([], {"selected_document_ids": ["same-id"]})):
        with pytest.raises(ValueError, match="overlap_run_documents"):
            runner.run(args)
