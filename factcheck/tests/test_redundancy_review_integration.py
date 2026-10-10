"""The optional reviewer must not replay the global-prompt regression."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from yjcheck.review_hints import all_review_hints
from yjcheck.text_review import detect_text


FIRST = "公司本期经营承压。"
SECOND = "公司本期经营承压，原因是新增关税导致出口下降。"
TERM = "应收账款变成坏账准备。"
SOURCE = FIRST + "\n" + SECOND + "\n" + TERM + "\n落款日期2025年2月30日。"
RAW = {"errors": [
    {"error_type": "冗余语句", "spans": [{"text": FIRST}, {"text": SECOND}], "reason": "两处均称经营承压。"},
    {"error_type": "术语误用", "spans": [{"text": TERM}], "reason": "应收账款的损失与坏账准备概念混用。"},
]}


def base_chat(messages, *, purpose):
    assert purpose == "text_review.detect"
    return {"content": json.dumps(RAW, ensure_ascii=False), "trace": {}}


def run_base():
    return detect_text(SOURCE, document_id="synthetic-scoped-review", chat=base_chat)


def decision_chat(action):
    calls = []
    def chat(messages, *, purpose):
        calls.append(purpose)
        if purpose == "text_review.detect":
            return base_chat(messages, purpose=purpose)
        assert purpose == "text_review.redundancy_review"
        payload = json.loads(messages[-1]["content"])
        assert payload["input_text"] == SOURCE
        assert len(payload["candidates"]) == 1
        item = {"error_id": payload["candidates"][0]["error_id"], "action": action,
                "reason": "前句提出主题，后句提供新增关税和出口下降原因。",
                "evidence": [{"text": FIRST}, {"text": SECOND}]}
        if action == "withdraw":
            item["normal_context"] = "topic_expansion"
        if action == "revise":
            item["spans"] = [{"text": FIRST}, {"text": SECOND}]
        return {"content": json.dumps({"decisions": [item]}, ensure_ascii=False), "trace": {
            "status": "ok", "call_id": "synthetic-review-call", "purpose": purpose, "cost_cny": "0",
            "request_sha256": sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
        }}
    return chat, calls


def test_withdraw_preserves_non_redundancy_and_confirmed_findings_byte_for_byte():
    base = run_base()
    chat, calls = decision_chat("withdraw")
    result = detect_text(SOURCE, document_id="synthetic-scoped-review", chat=chat, redundancy_review=True)
    assert calls == ["text_review.detect", "text_review.redundancy_review"]
    assert result["coverage"]["execution_complete"]
    assert result["coverage"]["planned_model_calls"] == 2
    assert result["coverage"]["finished_model_calls"] == 2
    assert result["errors"] == [e for e in base["errors"] if e["error_type"] != "冗余语句"]
    assert any(e["error_type"] == "术语误用" for e in result["errors"])
    assert any(e["status"] == "confirmed_error" for e in result["errors"])
    assert result["raw_candidates"] == base["raw_candidates"]
    assert result["rejected_candidates"] == base["rejected_candidates"]
    assert len(all_review_hints(deepcopy(result))["errors"]) == len(all_review_hints(deepcopy(base))["errors"]) - 1
    withdrawal = result["redundancy_review"]["withdrawn"][0]
    assert withdrawal["error"] == next(e for e in base["errors"] if e["error_type"] == "冗余语句")
    assert any(r.get("disposition") == "withdrawn_by_redundancy_review" for r in result["model_candidate_representation"])


def test_invalid_second_stage_cannot_be_reported_as_complete_or_erase_old_hints():
    base = run_base()
    def chat(messages, *, purpose):
        if purpose == "text_review.detect":
            return base_chat(messages, purpose=purpose)
        return {"content": '{"decisions":[]}', "trace": {}}
    result = detect_text(SOURCE, document_id="synthetic-scoped-review", chat=chat, redundancy_review=True)
    assert result["errors"] == base["errors"]
    assert result["raw_candidates"] == base["raw_candidates"]
    assert result["rejected_candidates"] == base["rejected_candidates"]
    assert not result["coverage"]["execution_complete"]
    assert not result["coverage"]["complete"]
    assert not result["coverage"]["model_complete"]
    assert result["coverage"]["processed_ranges"] == base["coverage"]["processed_ranges"]
    assert result["redundancy_review"]["base_execution_complete"] is True


def test_default_keeps_original_messages_and_never_adds_a_second_call():
    base = run_base()
    assert "redundancy_review" not in base
    assert base["coverage"]["planned_model_calls"] == 1


@pytest.mark.parametrize("value", [None, "true", 1, 0, {}, []])
def test_explicit_boolean_required(value):
    with pytest.raises(ValueError, match="redundancy_review must"):
        detect_text(SOURCE, document_id="synthetic-invalid", redundancy_review=value)


def test_legacy_rules_cannot_silently_claim_a_model_review_completed():
    with pytest.raises(ValueError, match="requires a model detector"):
        detect_text(SOURCE, document_id="synthetic-legacy", detector="legacy_rules", redundancy_review=True)


@pytest.mark.parametrize("config", [
    {"redundancy_review_policy": "context_v2"},
    {"redundancy_review_include_reason": False},
])
def test_nondefault_review_configuration_cannot_be_silently_ignored(config):
    with pytest.raises(ValueError, match="requires redundancy_review"):
        detect_text(SOURCE, document_id="synthetic-config", **config)


@pytest.mark.parametrize("config", [
    {"redundancy_review_policy": "unknown"},
    {"redundancy_review_policy": []},
    {"redundancy_review_include_reason": "false"},
    {"redundancy_review_include_reason": 0},
])
def test_invalid_review_configuration_fails_before_any_base_call(config):
    with pytest.raises(ValueError):
        detect_text(SOURCE, document_id="synthetic-config", chat=lambda *_: pytest.fail("unexpected call"),
                    redundancy_review=True, **config)


@pytest.mark.parametrize("include_reason", [True, False])
def test_context_verdict_removes_normal_false_alarm_with_identical_base_detection(include_reason):
    base_messages = []
    def chat(messages, *, purpose):
        if purpose == "text_review.detect":
            base_messages.append(deepcopy(messages))
            return base_chat(messages, purpose=purpose)
        payload = json.loads(messages[-1]["content"])
        candidate = payload["candidates"][0]
        assert ("reason" in candidate) is include_reason
        assert set(candidate) == ({"error_id", "spans", "reason"} if include_reason else {"error_id", "spans"})
        decision = {"error_id": candidate["error_id"], "verdict": "normal",
                    "reason": "主题之后说明原因，保留原文且撤回冗余指控。",
                    "context_units": [{"role": "topic", "quote": {"text": FIRST}},
                                      {"role": "explanation", "quote": {"text": SECOND}}],
                    "new_information": [{"summary": "新增关税使出口下降。", "unit_indices": [1]}],
                    "normal_context": "topic_expansion"}
        return {"content": json.dumps({"decisions": [decision]}, ensure_ascii=False), "trace": {
            "status": "ok", "call_id": "synthetic-context-review", "purpose": purpose, "cost_cny": "0",
            "request_sha256": sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
        }}
    baseline = detect_text(SOURCE, document_id="synthetic-context", chat=chat)
    result = detect_text(SOURCE, document_id="synthetic-context", chat=chat, redundancy_review=True,
                         redundancy_review_policy="context_v2", redundancy_review_include_reason=include_reason)
    assert base_messages[0] == base_messages[1]
    assert result["coverage"]["execution_complete"]
    assert result["redundancy_review"]["schema_version"] == "redundancy-review/2.1"
    assert result["redundancy_review"]["decisions"][0]["action"] == "withdraw"
    assert result["errors"] == [e for e in baseline["errors"] if e["error_type"] != "冗余语句"]
    assert result["raw_candidates"] == baseline["raw_candidates"]
    assert result["rejected_candidates"] == baseline["rejected_candidates"]
