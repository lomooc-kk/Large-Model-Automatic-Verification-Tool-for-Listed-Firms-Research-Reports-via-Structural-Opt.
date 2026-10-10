import copy
import hashlib
import json

import pytest

from yjcheck.text_context import TextSlice, estimated_input_tokens
from yjcheck.text_review import (
    _messages, _parse_response, detect_text, system_for_decision_contract,
)


SOURCE = "甲公司2025年收入100万元。另一表载甲公司2025年收入120万元。"


def decision(verdict="error_supported", *, text=SOURCE, kind="数值不一致错误", reason="同主体同年收入值不一致。"):
    return {"verdict": verdict, "error_type": kind, "spans": [{"text": text}], "reason": reason}


class Chat:
    def __init__(self, decisions=None, *, content=None, trace=None):
        self.content = content if content is not None else json.dumps({"decisions": decisions or []}, ensure_ascii=False)
        self.trace = trace or {"finish_reason": "stop"}
        self.calls = []

    def __call__(self, messages, *, purpose):
        self.calls.append((copy.deepcopy(messages), purpose))
        return {"content": self.content, "trace": self.trace}


def run(chat, *, text=SOURCE, **kwargs):
    return detect_text(text, document_id="synthetic-decision", detector="model_direct", chat=chat,
                       decision_contract=True, **kwargs)


def test_disabled_contract_leaves_default_messages_and_report_identical():
    seen = []
    def chat(messages, *, purpose):
        seen.append(messages)
        return {"content": '{"errors":[]}', "trace": {"finish_reason": "stop"}}
    first = detect_text(SOURCE, document_id="same", detector="model_direct", chat=chat)
    second = detect_text(SOURCE, document_id="same", detector="model_direct", chat=chat, decision_contract=False)
    assert first == second and seen[0] == seen[1]
    assert "raw_decisions" not in first and "decision_contract" not in first


def test_experimental_request_only_changes_explicit_output_contract():
    contexts = [TextSlice(0, len(SOURCE), SOURCE)]
    baseline = _messages(contexts, "same", "scene", [])
    changed = _messages(contexts, "same", "scene", [], decision_contract=True)
    assert changed[1:] == baseline[1:]
    assert changed[0]["content"] == system_for_decision_contract(baseline[0]["content"])
    with pytest.raises(ValueError, match="exactly one"):
        system_for_decision_contract(changed[0]["content"])


def test_all_verdicts_remain_auditable_with_original_indexes_and_hashes():
    items = [decision("no_error", reason="当前核查点不成立。"),
             decision("insufficient_evidence", reason="该另一核查点缺少必要口径定义。"), decision()]
    original = copy.deepcopy(items)
    report = run(Chat(items))
    assert [x["decision"] for x in report["raw_decisions"]] == original
    assert [x["decision_ref"] for x in report["raw_decisions"]] == [f"decision:0:{i}" for i in range(3)]
    routes = report["model_decision_dispositions"]
    assert [x["disposition"] for x in routes] == ["no_error", "insufficient_evidence", "error_candidate"]
    assert routes[2]["candidate_ref"] == report["raw_candidates"][0]["candidate_ref"] == "model:0:2"
    assert [x["candidate_ref"] for x in report["model_candidate_representation"]] == ["model:0:2"]
    represented = report["errors"][0]["represented_model_candidates"][0]
    assert represented["candidate"] == items[2]
    assert represented["raw_candidate_sha256"] == hashlib.sha256(json.dumps(items[2], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert all(not x["verdict_is_business_proof"] for x in routes)
    assert report["decision_contract"]["business_correctness"] is None
    assert report["coverage"]["execution_complete"] and report["coverage"]["candidate_quality_complete"]
    assert items == original


def test_normal_verdict_cannot_cancel_rule_proof_or_independent_model_error():
    text = "公告日期为2024年2月30日。收入为100万元。"
    quote = text[:text.index("收入")]
    baseline = detect_text(text, document_id="rule", detector="hybrid",
                           chat=lambda messages, **kw: {"content": '{"errors":[]}', "trace": {}})
    items = [decision("no_error", text=quote, kind="时间信息非法", reason="合成的错误正常判定。"),
             decision(text=quote, kind="格式错误", reason="合成的另一独立候选。")]
    result = detect_text(text, document_id="rule", detector="hybrid", chat=Chat(items), decision_contract=True)
    before = {x["id"]:x for x in baseline["errors"] if x["status"] == "confirmed_error"}
    after = {x["id"]:x for x in result["errors"] if x["status"] == "confirmed_error"}
    assert before and before == after
    assert any(x["error_type"] == "格式错误" and x["status"] == "needs_review" for x in result["errors"])
    assert result["model_decision_dispositions"][0]["disposition"] == "no_error"


def test_negative_words_never_override_positive_verdict():
    report = run(Chat([decision(reason="数值一致，无冲突，不应报告。")]))
    assert len(report["errors"]) == len(report["raw_candidates"]) == 1
    assert report["errors"][0]["status"] == "needs_review"
    assert report["errors"][0]["reason"] == "数值一致，无冲突，不应报告。"
    # The protocol exposes the contradictory model opinion; it does not certify it.
    assert report["decision_contract"]["business_correctness"] is None


@pytest.mark.parametrize("verdict", ["no_error", "insufficient_evidence"])
def test_bad_nonerror_quote_is_visible_and_marks_quality_incomplete(verdict):
    report = run(Chat([decision(verdict, text="不在源文中的句子。")]))
    assert report["coverage"]["execution_complete"] is True
    assert report["coverage"]["candidate_quality_complete"] is False
    assert report["coverage"]["complete"] is False
    assert len(report["raw_decisions"]) == len(report["model_decision_dispositions"]) == 1
    assert report["model_decision_dispositions"][0]["disposition"] == "anchor_rejected"
    assert not report["raw_candidates"] and not report["errors"]


def test_positive_anchor_failure_keeps_direct_candidate_and_raw_decision():
    report = run(Chat([decision(text="不在源文中的句子。")]))
    assert len(report["raw_decisions"]) == len(report["raw_candidates"]) == len(report["rejected_candidates"]) == 1
    assert report["errors"][0]["invalid_anchor"] is True
    assert report["model_decision_dispositions"][0]["candidate_ref"] == "model:0:0"
    assert report["coverage"]["candidate_quality_complete"] is False


@pytest.mark.parametrize("payload", [
    {"decisions": [decision(), {**decision("no_error"), "verdict": "invented"}]},
    {"decisions": [decision(), decision()]},
    {"decisions": [decision(reason="长" * 401)]},
    {"errors": []},
])
def test_invalid_whole_contract_preserves_final_response_and_no_valid_prefix(payload):
    content = json.dumps(payload, ensure_ascii=False)
    report = run(Chat(content=content))
    assert report["coverage"]["execution_complete"] is False
    assert not report["decision_contract"]["protocol_complete"]
    assert report["raw_decisions"] == report["raw_candidates"] == report["errors"] == []
    failed, = report["decision_protocol_failures"]
    assert failed["error_stage"] == "response_parse" and failed["error_code"]
    assert failed["raw_response"] == content
    assert failed["raw_response_sha256"] == hashlib.sha256(content.encode()).hexdigest()


def test_runtime_truncation_is_failure_even_if_visible_json_is_complete():
    report = run(Chat([], trace={"finish_reason": "length"}))
    assert not report["coverage"]["execution_complete"]
    assert report["decision_protocol_failures"][0]["raw_response"] == '{"decisions": []}'
    assert not report["raw_decisions"]
    with pytest.raises(ValueError, match="truncated"):
        _parse_response({"content": '{"decisions":[]}', "trace": {"response_content_incomplete": True}}, decision_contract=True)


def test_strict_contract_does_not_repair_or_strip_malformed_json():
    for content in ['{"decisions":[]', '```json\n{"decisions":[]}\n```', '{"decisions":[],"decisions":[]}']:
        report = run(Chat(content=content))
        assert not report["coverage"]["execution_complete"]
        assert report["decision_protocol_failures"][0]["raw_response"] == content


def test_explicit_opt_in_and_rule_only_misconfiguration_fail_before_call():
    for bad in (None, 0, 1, "yes"):
        with pytest.raises(ValueError, match="explicit boolean"):
            detect_text(SOURCE, document_id="bad", decision_contract=bad, chat=Chat())
    with pytest.raises(ValueError, match="model detector"):
        detect_text(SOURCE, document_id="bad", detector="legacy_rules", decision_contract=True, chat=Chat())


def test_development_examples_are_adapted_without_adding_answers():
    text = "示例收入为 元。"
    examples = [{"source_split": "dev", "source_id": "other-example", "content": text,
                 "complete_annotation": True, "errors": [{"error_type": "数值缺失", "reason": "数值空槽。",
                 "spans": [{"start": 0, "end": len(text), "text": text}]}]}]
    original = copy.deepcopy(examples)
    chat = Chat()
    report = run(chat, examples=examples)
    assistant = next(m for m in chat.calls[0][0] if m["role"] == "assistant")
    assert json.loads(assistant["content"]) == {"decisions": [{"verdict": "error_supported", **examples[0]["errors"][0]}]}
    assert examples == original and report["coverage"]["execution_complete"]


def test_windowed_and_global_requests_all_use_the_same_new_contract():
    paragraphs = [f"甲公司202{i % 6}年营业收入100万元。" + "业务描述" * 90 for i in range(10)]
    source = "\n\n".join(paragraphs)
    overhead = estimated_input_tokens(_messages([], "synthetic-decision", "", [], decision_contract=True))
    calls = []
    def chat(messages, *, purpose):
        assert '"decisions"' in messages[0]["content"]
        assert '输出errors空数组' not in messages[0]["content"]
        payload = json.loads(messages[-1]["content"])
        calls.append((messages, purpose))
        item = decision("no_error", text=payload["contexts"][0]["text"], reason="合成窗口判定。")
        item["spans"][0].update(start=payload["contexts"][0]["start"], end=payload["contexts"][0]["end"])
        return {"content": json.dumps({"decisions": [item]}, ensure_ascii=False), "trace": {"finish_reason": "stop"}}
    report = run(chat, text=source, max_input_tokens=overhead + 600)
    assert report["coverage"]["context_mode"] == "paragraph_windows_with_global_links"
    assert len(calls) > 1 and any(purpose == "text_review.global" for _, purpose in calls)
    assert all(estimated_input_tokens(messages) <= overhead + 600 for messages, _ in calls)
    assert len(report["raw_decisions"]) == len(calls)
    assert len({x["decision_ref"] for x in report["raw_decisions"]}) == len(calls)


def test_new_contract_budget_overhead_is_not_silently_removed():
    text = "原文。"
    cap = estimated_input_tokens(_messages([TextSlice(0, len(text), text)], "synthetic-decision", "", []))
    chat = Chat()
    report = run(chat, text=text, max_input_tokens=cap)
    assert not chat.calls and not report["coverage"]["execution_complete"]
    assert report["decision_contract"]["protocol_complete"] is False
