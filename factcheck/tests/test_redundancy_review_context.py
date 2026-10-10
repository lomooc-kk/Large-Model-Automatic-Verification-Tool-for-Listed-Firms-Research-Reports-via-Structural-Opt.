"""Synthetic protocol tests; no labels and no external model calls."""
from copy import deepcopy
import hashlib
import json

import pytest

from yjcheck.redundancy_review import review_redundancy
from test_redundancy_review import Chat, FIRST, SECOND, SOURCE, WIDE, base_report, receipt, span


def normal(error_id="r1"):
    return {
        "error_id": error_id,
        "context_units": [{"role": "topic", "quote": {"text": FIRST["text"]}},
                          {"role": "explanation", "quote": {"text": SECOND["text"]}}],
        "new_information": [{"summary": "后句新增需求下降这一原因。", "unit_indices": [1]}],
        "verdict": "normal", "reason": "前句提出主题，后句解释原因，原错误指控不成立。",
        "normal_context": "topic_expansion",
    }


def uncertain(error_id="r1"):
    return {"error_id": error_id, "context_units": [{"role": "other", "quote": {"text": WIDE["text"]}}],
            "new_information": [], "verdict": "uncertain", "reason": "片段关系仍待判断，暂保留复核。"}


def invoke(decisions, *, source=SOURCE, before=None, **kwargs):
    before = base_report() if before is None else before
    chat = Chat(decisions)
    return review_redundancy(source, before, chat, policy="context_v2", **kwargs), chat


def assert_atomic_failure(out, before):
    assert out["errors"] == before["errors"]
    assert out["raw_candidates"] == before["raw_candidates"]
    assert out["rejected_candidates"] == before["rejected_candidates"]
    assert out["model_candidate_representation"] == before["model_candidate_representation"]
    assert out["redundancy_review"]["status"] == "incomplete"
    assert not out["redundancy_review"]["complete"]
    assert out["redundancy_review"]["decisions"] == []
    assert out["coverage"]["processed_ranges"] == before["coverage"]["processed_ranges"]
    assert all(out["coverage"][k] is False for k in ("complete", "execution_complete", "model_complete"))
    assert out["coverage"]["planned_model_calls"] == before["coverage"]["planned_model_calls"] + 1
    assert out["coverage"]["finished_model_calls"] == before["coverage"]["finished_model_calls"]


def mechanical_fixture():
    source = "计算机行业行业月报。"
    before = base_report()
    before["errors"][0]["spans"] = [{"start": 0, "end": len(source), "text": source}]
    first = span("行业", source)
    second = span("行业", source, first["end"])
    decision = {
        "error_id": "r1", "context_units": [{"role": "heading", "quote": {"text": source}}],
        "new_information": [], "verdict": "redundant", "reason": "完整标题内的行业一词紧邻复写。",
        "members": [first, second],
    }
    return source, before, decision


def test_default_actions_request_is_exact_frozen_r11_protocol():
    chat = Chat()
    result = review_redundancy(SOURCE, base_report(), chat)
    messages = chat.calls[0][0]
    # Fixed from the independently loaded frozen R11 module on this fixture.
    assert hashlib.sha256(messages[0]["content"].encode()).hexdigest() == "85b45f4fadc6a543e40073236a178296ca598f2bf60b1a11bdbc09a5ab41633a"
    assert hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest() == "b273b5536cd93487b7b7a2109757a171d4207383eb68805ca54a558be0b89665"
    stage = result["redundancy_review"]
    assert (stage["schema_version"], stage["policy"], stage["include_reason"]) == ("redundancy-review/1.0", "actions_v1", True)
    assert "verdict" not in stage["decisions"][0]
    assert result["errors"] == base_report()["errors"]


def test_normal_units_need_not_repeat_and_program_withdraws_only_target():
    before = base_report()
    original = deepcopy(before)
    raw = normal()
    out, chat = invoke([raw], before=before)
    assert before == original
    assert out["errors"] == original["errors"][1:]
    assert out["raw_candidates"] == original["raw_candidates"]
    assert out["rejected_candidates"] == original["rejected_candidates"]
    stage = out["redundancy_review"]
    assert (stage["schema_version"], stage["policy"], stage["include_reason"]) == ("redundancy-review/2.1", "context_v2", True)
    record, = stage["decisions"]
    assert record["action"] == "withdraw" and record["verdict"] == "normal"
    assert record["action_source"] == "programmed_verdict_mapping"
    assert record["context_unit_completeness"] == "model_declared_not_deterministically_verified"
    assert record["decision"]["context_units"][0]["quote"]["text"] != record["decision"]["context_units"][1]["quote"]["text"]
    assert record["decision"]["new_information"] == raw["new_information"]
    assert record["before"] == original["errors"][0] and record["after"] is None
    assert json.loads(stage["raw_response"])["decisions"][0] == raw
    assert "action" not in raw
    assert stage["runtime_trace"] == chat.last_response["trace"]
    assert out["model_candidate_representation"][0]["disposition"] == "withdrawn_by_redundancy_review"


def test_uncertain_maps_to_retain_without_rewriting_original_warning():
    before = base_report()
    out, _ = invoke([uncertain()], before=before)
    assert out["errors"] == before["errors"]
    record, = out["redundancy_review"]["decisions"]
    assert record["action"] == "retain" and record["verdict"] == "uncertain"
    assert record["after"] == record["before"]
    assert record["decision"]["reason"] != record["after"]["reason"]


def test_one_full_context_can_support_two_local_mechanical_repeat_members():
    source, before, raw = mechanical_fixture()
    out, _ = invoke([raw], source=source, before=before)
    assert out["redundancy_review"]["complete"]
    after = out["errors"][0]
    assert after["id"] == "r1" and after["status"] == "needs_review"
    assert after["spans"] == [span("行业行业", source)]
    assert after["represented_model_candidates"] == before["errors"][0]["represented_model_candidates"]
    record, = out["redundancy_review"]["decisions"]
    assert record["action"] == "revise" and record["verdict"] == "redundant"
    assert record["after"] == after and len(record["decision"]["spans"]) == 2
    assert len(record["evidence"]) == 1
    assert record["before"] == before["errors"][0]
    unmerged, _ = invoke([raw], source=source, before=before, normalize_spans=False)
    assert unmerged["errors"][0]["spans"] == raw["members"]


@pytest.mark.parametrize("policy", ["actions_v1", "context_v2"])
def test_hiding_reason_is_a_single_payload_field_change(policy):
    report = base_report()
    report["errors"][0]["reason"] = "ORIGINAL_REASON_SECRET"
    decisions = [normal()] if policy == "context_v2" else None
    with_chat, without_chat = Chat(decisions), Chat(decisions)
    included = review_redundancy(SOURCE, report, with_chat, policy=policy, include_reason=True)
    omitted = review_redundancy(SOURCE, report, without_chat, policy=policy, include_reason=False)
    left, right = with_chat.calls[0][0], without_chat.calls[0][0]
    assert left[0] == right[0]
    a, b = json.loads(left[1]["content"]), json.loads(right[1]["content"])
    assert a["input_text"] == b["input_text"] == SOURCE
    assert a["candidates"][0].pop("reason") == "ORIGINAL_REASON_SECRET"
    assert a == b
    assert "ORIGINAL_REASON_SECRET" not in json.dumps(right)
    assert "ORIGINAL_REASON_SECRET" in json.dumps(omitted["redundancy_review"]["decisions"][0]["before"])
    assert included["redundancy_review"]["include_reason"] is True
    assert omitted["redundancy_review"]["include_reason"] is False
    assert included["redundancy_review"]["request_sha256"] != omitted["redundancy_review"]["request_sha256"]


@pytest.mark.parametrize("change", [
    {"action": "retain"}, {"verdict": "confirmed_error"}, {"members": [FIRST, SECOND]},
    {"normal_context": "any_reason"}, {"context_units": []},
    {"context_units": [{"role": "topic", "quote": {"text": FIRST["text"]}}]},
    {"context_units": [{"role": "unrecognized", "quote": {"text": FIRST["text"]}},
                       {"role": "explanation", "quote": {"text": SECOND["text"]}}]},
    {"context_units": [{"role": "topic", "quote": {"text": FIRST["text"]}, "proof": "confirmed"},
                       {"role": "explanation", "quote": {"text": SECOND["text"]}}]},
    {"reason": " "}, {"new_information": None},
])
def test_invalid_v2_schema_cannot_override_programmed_actions(change):
    before = base_report()
    raw = {**normal(), **deepcopy(change)}
    out, _ = invoke([raw], before=before)
    assert_atomic_failure(out, before)


@pytest.mark.parametrize("indices", [[], [True], [-1], [2], [0, 0], ["0"], [0, {}], None])
def test_new_information_citations_require_valid_nonempty_unique_integer_indices(indices):
    before = base_report()
    raw = normal()
    raw["new_information"] = [{"summary": "新增原因。", "unit_indices": indices}]
    out, _ = invoke([raw], before=before)
    assert_atomic_failure(out, before)
    assert out["redundancy_review"]["failure"]["code"] == "invalid_context_unit_indices"


@pytest.mark.parametrize("addition", [
    {"summary": "", "unit_indices": [1]}, {"summary": None, "unit_indices": [1]},
    {"summary": "新增原因。", "unit_indices": [1], "invented_fact": True},
])
def test_new_information_summary_schema_is_validated(addition):
    before = base_report()
    raw = normal()
    raw["new_information"] = [addition]
    out, _ = invoke([raw], before=before)
    assert_atomic_failure(out, before)


def test_all_original_spans_must_be_related_to_context_units():
    before = base_report()
    before["errors"][0]["spans"] = [FIRST, SECOND]
    raw = normal()
    raw["context_units"][1]["quote"] = {"text": "另一段正常信息。"}
    out, _ = invoke([raw], before=before)
    assert_atomic_failure(out, before)
    assert out["redundancy_review"]["failure"]["code"] == "context_units_unrelated_to_candidate"


def test_exact_source_evidence_does_not_certify_semantic_completeness_or_business_truth():
    out, _ = invoke([normal()])
    assert out["redundancy_review"]["evidence_scope"] == "source_only_model_judgment_not_proof"
    record, = out["redundancy_review"]["decisions"]
    assert record["judgment"] == "model_proposal_not_deterministic_proof"
    assert record["context_unit_completeness"] == "model_declared_not_deterministically_verified"


@pytest.mark.parametrize("kind", ["missing", "one", "three", "overlap", "outside_unit", "unrelated"])
def test_redundant_members_must_be_exact_paired_related_and_inside_full_units(kind):
    source, before, raw = mechanical_fixture()
    if kind == "missing":
        del raw["members"]
    elif kind == "one":
        raw["members"] = raw["members"][:1]
    elif kind == "three":
        raw["members"].append(span("月报", source))
    elif kind == "overlap":
        raw["members"][1] = deepcopy(raw["members"][0])
    elif kind == "outside_unit":
        raw["context_units"][0]["quote"] = raw["members"][0]
    else:
        before["errors"][0]["spans"] = [span("月报", source)]
    out, _ = invoke([raw], source=source, before=before)
    assert_atomic_failure(out, before)


def test_ambiguous_context_quote_or_wrong_coordinates_never_guess():
    source, before, raw = mechanical_fixture()
    raw["context_units"][0]["quote"] = {"text": "行业"}
    out, _ = invoke([raw], source=source, before=before)
    assert_atomic_failure(out, before)
    assert out["redundancy_review"]["failure"]["code"] == "ambiguous_source_quote"
    raw["context_units"][0]["quote"] = {"start": 1, "end": len(source), "text": source}
    out, _ = invoke([raw], source=source, before=before)
    assert_atomic_failure(out, before)


@pytest.mark.parametrize("invalid", ["unknown", "duplicate", "missing", "partial"])
def test_v2_batch_rolls_back_every_valid_decision_if_one_id_or_item_is_invalid(invalid):
    before = base_report()
    other = deepcopy(before["errors"][0]); other["id"] = "r2"
    before["errors"].append(other)
    decisions = [normal(), normal("r2")]
    if invalid == "unknown":
        decisions[1]["error_id"] = "unknown"
    elif invalid == "duplicate":
        decisions[1]["error_id"] = "r1"
    elif invalid == "missing":
        decisions.pop()
    else:
        decisions[1]["new_information"] = [{"summary": "", "unit_indices": [1]}]
    out, _ = invoke(decisions, before=before)
    assert_atomic_failure(out, before)


def test_v2_still_requires_exact_budget_receipt_and_preserves_failure_trace():
    before = base_report()
    def response(messages):
        return {"content": json.dumps({"decisions": [normal()]}),
                "trace": receipt(messages, status="reserved", call_id="synthetic-unfinished")}
    chat = Chat(response=response)
    out = review_redundancy(SOURCE, before, chat, policy="context_v2")
    assert_atomic_failure(out, before)
    assert out["redundancy_review"]["runtime_trace"] == chat.last_response["trace"]
    assert out["redundancy_review"]["raw_response"] == chat.last_response["content"]


def test_v2_budget_refusal_and_not_applicable_have_no_calls():
    before = base_report()
    out, chat = invoke([normal()], before=before, max_input_tokens=1)
    assert_atomic_failure(out, before)
    assert not chat.calls
    before["errors"] = before["errors"][1:]
    out, chat = invoke([], before=before)
    assert not chat.calls
    assert out["redundancy_review"]["status"] == "not_applicable"
    assert out["redundancy_review"]["schema_version"] == "redundancy-review/2.1"


@pytest.mark.parametrize("policy", [None, "context_v3", [], True])
def test_invalid_policy_fails_before_chat(policy):
    chat = Chat()
    with pytest.raises(ValueError, match="policy"):
        review_redundancy(SOURCE, base_report(), chat, policy=policy)
    assert not chat.calls


@pytest.mark.parametrize("option", [None, "false", 0, 1, []])
def test_include_reason_requires_explicit_boolean(option):
    chat = Chat()
    with pytest.raises(ValueError, match="include_reason"):
        review_redundancy(SOURCE, base_report(), chat, include_reason=option)
    assert not chat.calls
