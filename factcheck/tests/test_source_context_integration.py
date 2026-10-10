import json

import pytest
from yjcheck import text_review
from yjcheck.text_review import TextSlice, _messages, detect_text


def payload(text, offset=0, *, source_integrity_hints=False):
    return json.loads(_messages([TextSlice(offset, offset + len(text), text)], "source-only", "", [],
                               source_integrity_hints=source_integrity_hints)[-1]["content"])


def test_clean_text_has_no_integrity_metadata():
    result = payload("本期营业收入1亿元，同比增长5%，归母净利润0.1亿元。")
    assert "source_integrity_checks" not in result
    assert result["source_arithmetic_checks"] == []
    assert set(result) == {"document_id", "scene", "mode", "contexts", "source_arithmetic_checks"}


def test_surface_signal_is_local_and_cannot_authorize_a_verdict():
    text = "电网累计投资1408亿元，同比增长45.33GW，同比11%。其他业务收入2亿元。"
    result = payload(text, 20, source_integrity_hints=True)
    check, = result["source_integrity_checks"]
    assert check["business_error_proven"] is False
    assert check["relationship_binding_proven"] is False
    assert check["missing_field_proven"] is False
    assert "其他业务" not in check["source"]["text"]
    assert text[check["source"]["start"] - 20:check["source"]["end"] - 20] == check["source"]["text"]


def test_proportion_rounding_aid_enters_actual_request_without_changing_source():
    text = "贡献比例由上年的31%提升至44%，同比增长44%。"
    result = payload(text)
    check, = result["source_arithmetic_checks"]
    assert check["kind"] == "conditional_displayed_proportion_growth"
    assert check["rounding_compatible"] is True
    assert result["contexts"][0]["text"] == text
    assert "gold" not in result and "expected" not in result


def test_overlapping_contexts_deduplicate_metadata_by_exact_source_position():
    text = "并网0.54万/吨。"
    part = TextSlice(10, 10 + len(text), text)
    result = json.loads(_messages([part, part], "source-only", "", [], source_integrity_hints=True)[-1]["content"])
    assert len(result["source_integrity_checks"]) == 1


def test_integrity_metadata_is_bounded_without_dropping_original_contexts():
    text = "并网0.54万/吨。" * 20
    result = payload(text, source_integrity_hints=True)
    assert len(result["source_integrity_checks"]) == 12
    assert result["contexts"][0]["text"] == text


def test_damaged_source_defaults_off_without_disabling_proportion_aid(monkeypatch):
    text = "实现营业收入亿。贡献比例由上年的31%提升至44%，同比增长44%。"
    monkeypatch.setattr(text_review, "source_integrity_checks", lambda *a, **k: pytest.fail("Default must not generate experimental hints"))
    seen = []
    def chat(messages, *, purpose):
        request = json.loads(messages[-1]["content"])
        assert "source_integrity_checks" not in request
        assert request["contexts"] == [{"start": 0, "end": len(text), "text": text}]
        check, = request["source_arithmetic_checks"]
        assert check["kind"] == "conditional_displayed_proportion_growth" and check["rounding_compatible"] is True
        seen.append(request)
        return {"content": '{"errors":[]}', "trace": {}}
    result = detect_text(text, document_id="synthetic-default", detector="model_direct", chat=chat)
    assert len(seen) == 1 and result["coverage"]["execution_complete"] is True


def test_explicit_opt_in_sends_local_hints_and_preserves_whole_source():
    text = "实现营业收入亿。其他业务收入1亿元。"
    seen = []
    def chat(messages, *, purpose):
        request = json.loads(messages[-1]["content"])
        assert request["contexts"][0]["text"] == text
        check, = request["source_integrity_checks"]
        assert check["business_error_proven"] is False and check["source"]["text"] == "实现营业收入亿"
        seen.append(request)
        return {"content": '{"errors":[]}', "trace": {}}
    result = detect_text(text, document_id="synthetic-opt-in", detector="model_direct", chat=chat,
                         source_integrity_hints=True)
    assert len(seen) == 1 and result["coverage"]["execution_complete"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_window_and_global_message_paths_share_experimental_setting(monkeypatch, enabled):
    original = text_review._messages
    calls = []
    def checked(*args, **kwargs):
        assert kwargs.get("source_integrity_hints") is enabled
        calls.append((bool(args[0]), kwargs.get("global_check", False)))
        return original(*args, **kwargs)
    monkeypatch.setattr(text_review, "_messages", checked)
    text = "\n\n".join(f"第{i}段：本期营业收入{i+1}亿元，同比增长5%。" + "经营情况持续稳定。" * 15 for i in range(8))
    def chat(messages, *, purpose):
        return {"content": '{"errors":[]}', "trace": {}}
    # Size the budget from the same full prompt, then force local windows and
    # repeated-metric global links without replacing token accounting.
    budget = text_review.estimated_input_tokens(original([], "synthetic-windows", "", [], source_integrity_hints=enabled)) + 450
    result = detect_text(text, document_id="synthetic-windows", chat=chat, max_input_tokens=budget,
                         source_integrity_hints=enabled)
    assert result["coverage"]["context_mode"] == "paragraph_windows_with_global_links"
    assert any(not has_parts for has_parts, _ in calls)
    assert any(global_check for _, global_check in calls)
    assert len(calls) > 3


@pytest.mark.parametrize("value", [1, "true", None])
def test_experimental_setting_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        detect_text("实现营业收入亿。", document_id="synthetic-setting", source_integrity_hints=value)
