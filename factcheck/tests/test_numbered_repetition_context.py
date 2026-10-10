"""Source comparisons must preserve roles, numbers, tails and exact locations."""
import json
import pytest

from yjcheck.numbered_repetition_context import source_numbered_repeat_checks
from yjcheck.text_review import _messages, detect_text
from yjcheck.text_context import TextSlice, estimated_input_tokens


BODY = "设备业务稳步增长，海外销售保持稳定。后续产品研发投入增加。"


@pytest.mark.parametrize("gap", ["", "\n", "\r\n", " "])
def test_adjacent_multisentence_members_keep_markers_and_bodies(gap):
    text = "业务分类：1）其他业务经营正常。2）" + BODY + gap + "3）" + BODY
    checks = source_numbered_repeat_checks(text)
    assert len(checks) == 1
    check = checks[0]
    assert check["numbers"] == [2, 3]
    assert [m["text"] for m in check["body_members"]] == [BODY, BODY]
    assert [m["text"] for m in check["members"]] == ["2）" + BODY, "3）" + BODY]
    assert check["list_intro"]["text"] == "业务分类："
    assert check["business_correctness_confirmed"] is False


def test_right_new_condition_is_preserved_but_outside_copy():
    tail = "但若新增关税，预测可能调整。"
    text = "两位受访者依次回答：1）" + BODY + "2）" + BODY + tail
    check = source_numbered_repeat_checks(text)[0]
    assert check["list_intro"]["text"] == "两位受访者依次回答："
    assert check["right_member_has_additional_text"] is True
    assert check["excluded_right_tail"]["text"] == tail
    assert tail not in check["source"]["text"]
    assert check["business_correctness_confirmed"] is False


def test_preceding_heading_remains_available_when_list_starts_new_paragraph():
    text = "两位受访者分别回答：\n\n1）" + BODY + "\n2）" + BODY
    check = source_numbered_repeat_checks(text)[0]
    assert check["preceding_context"]["text"] == "两位受访者分别回答："


@pytest.mark.parametrize("ending", ["。", "。\n", "？", "。”", "："])
def test_punctuated_introduction_preserves_source_role_and_example_warning(ending):
    intro = "两位受访者分别回答" + ending
    check = source_numbered_repeat_checks("行业背景。" + intro + "1）" + BODY + "2）" + BODY)[0]
    assert check["list_intro"]["text"] == intro.strip()
    assert source_numbered_repeat_checks("下面是错误示例" + ending + "1）" + BODY + "2）" + BODY) == []


@pytest.mark.parametrize("text", [
    "1公斤产品价格：保持稳定，市场供给需求没有变化。\n2公斤产品价格：保持稳定，市场供给需求没有变化。",
    "1级资本：同比增长显著，资本充足率保持稳定。\n2级资本：同比增长显著，资本充足率保持稳定。",
    "1个月合同价格：保持稳定，市场供给需求没有变化。\n2个月合同价格：保持稳定，市场供给需求没有变化。",
    "2政策安排：继续完善各地制度，稳步提升服务能力。\n3政策安排：继续完善各地制度，稳步提升服务能力。",
])
def test_bare_numbers_are_not_assumed_to_be_ordinals(text):
    assert source_numbered_repeat_checks(text) == []


@pytest.mark.parametrize("text", [
    "1）" + BODY + "3）" + BODY,
    "1）" + BODY + "1）" + BODY,
    "1）" + BODY + "(2)" + BODY,
    "1）" + BODY + "\n\n2）" + BODY,
    "1）" + BODY + "2）" + BODY + "3）" + BODY,
    "1）项目进展正常。(1)" + BODY + "(2)" + BODY + "2）其他产业稳步发展。",
    "这是错误示例：1）" + BODY + "2）" + BODY,
])
def test_unsupported_parent_structure_or_examples_abstain(text):
    assert source_numbered_repeat_checks(text) == []


def test_changed_period_or_number_is_new_information():
    one = "公司2024年营业收入实现稳定增长。"
    two = "公司2025年营业收入实现稳定增长。"
    assert source_numbered_repeat_checks("1）" + one + "2）" + two) == []
    assert source_numbered_repeat_checks("1）" + BODY + "2）" + BODY.replace("稳定", "下降")) == []


def test_repeated_title_does_not_erase_different_item_bodies():
    text = "1）价格上涨。进口数量同比下降。2）价格上涨。出口金额同比增长。"
    assert source_numbered_repeat_checks(text) == []


def test_window_offsets_cover_every_supplied_source_fragment():
    prefix = "既有背景。\n"
    part = "产业列表：1）" + BODY + "\n2）" + BODY + "但未来税率可能变化。"
    full = prefix + part
    check = source_numbered_repeat_checks(part, offset=len(prefix))[0]
    def walk(value):
        if isinstance(value, dict):
            if {"start", "end", "text"} <= set(value):
                assert full[value["start"]:value["end"]] == value["text"]
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(check)


def test_context_payload_deduplicates_overlapping_windows_without_dropping_tail():
    text = "业务列表：1）" + BODY + "2）" + BODY + "新增订单尚未交付。"
    part = TextSlice(0, len(text), text)
    messages = _messages([part, part], "synthetic-list", "", [], redundancy_context_experiment=True)
    payload = json.loads(messages[-1]["content"])
    assert len(payload["source_numbered_repeat_checks"]) == 1
    assert payload["source_numbered_repeat_checks"][0]["excluded_right_tail"]["text"] == "新增订单尚未交付。"
    assert payload["contexts"][0]["text"] == text


def test_context_does_not_create_or_confirm_an_error_on_its_own():
    text = "两位受访者依次回答：1）" + BODY + "2）" + BODY
    def chat(messages, *, purpose):
        assert json.loads(messages[-1]["content"])["source_numbered_repeat_checks"]
        return {"content": '{"errors":[]}', "trace": {}}
    report = detect_text(text, document_id="synthetic-context-only", detector="model_direct", chat=chat,
                         redundancy_context_experiment=True)
    assert report["errors"] == []
    assert report["coverage"]["execution_complete"]


def test_large_optional_checks_never_block_a_source_window_that_fits():
    body = "行业前景保持稳定，企业生产规模没有变化。" * 60
    text = "业务列表：1）" + body + "2）" + body
    part = TextSlice(0, len(text), text)
    without_optional = _messages([part], "synthetic-long", "", [], max_input_tokens=1,
                                  redundancy_context_experiment=True)
    baseline_tokens = estimated_input_tokens(without_optional)
    assert "source_numbered_repeat_checks" not in json.loads(without_optional[-1]["content"])
    calls = []
    def chat(messages, *, purpose):
        assert estimated_input_tokens(messages) <= baseline_tokens + 20
        assert json.loads(messages[-1]["content"])["contexts"] == [part.to_dict()]
        calls.append(messages)
        return {"content": '{"errors":[]}', "trace": {}}
    report = detect_text(text, document_id="synthetic-long", detector="model_direct", chat=chat,
                         max_input_tokens=baseline_tokens + 20, redundancy_context_experiment=True)
    assert len(calls) == 1
    assert report["coverage"]["execution_complete"]
    assert report["errors"] == []


def test_default_does_not_invoke_experimental_source_comparison(monkeypatch):
    from yjcheck import text_review
    def unwanted_call(*args, **kwargs):
        raise AssertionError("Experimental comparison must be opt-in")
    monkeypatch.setattr(text_review, "source_numbered_repeat_checks", unwanted_call)
    text = "业务列表：1）" + BODY + "2）" + BODY
    def chat(messages, *, purpose):
        assert "source_numbered_repeat_checks" not in json.loads(messages[-1]["content"])
        assert messages[0]["content"] == text_review._SYSTEM
        return {"content": '{"errors":[]}', "trace": {}}
    report = detect_text(text, document_id="synthetic-default", detector="model_direct", chat=chat)
    assert report["coverage"]["execution_complete"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_experiment_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="redundancy_context_experiment"):
        detect_text("原文。", document_id="synthetic-invalid-option", redundancy_context_experiment=value)


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"offset": True}, {"limit": -1}, {"limit": True}])
def test_invalid_position_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        source_numbered_repeat_checks("", **kwargs)
