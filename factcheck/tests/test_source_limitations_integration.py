import copy
import hashlib
import json

import pytest

from yjcheck.source_limitations import build_source_limitation_hints
from yjcheck.text_context import TextSlice, estimated_input_tokens
from yjcheck.text_review import SOURCE_LIMITATIONS_GUIDANCE, _messages, detect_text


def metadata(text, quote, *, role="non_source_marker"):
    start = text.index(quote)
    return {"schema_version": "source-limitations/1.0", "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "items": [{"kind": "occluded_text", "span": {"start": start, "end": start + len(quote), "text": quote},
                       "text_role": role, "evidence": {"source_sha256": "a" * 64, "page": 1,
                       "image_sha256": "b" * 64, "bbox": [12, 10, 40, 21], "bbox_units": "pdf_points",
                       "method": "visual_review"}}]}


def test_none_and_valid_empty_metadata_leave_actual_messages_and_report_identical():
    text = "利润〔原页遮挡〕亿元。"
    empty = metadata(text, "〔原页遮挡〕")
    empty["items"] = []
    seen = []
    def chat(messages, *, purpose):
        seen.append(copy.deepcopy(messages))
        return {"content": '{"errors":[]}', "trace": {}}
    first = detect_text(text, document_id="synthetic", detector="model_direct", chat=chat)
    second = detect_text(text, document_id="synthetic", detector="model_direct", chat=chat, source_limitations=empty)
    assert first == second and seen[0] == seen[1]
    assert "source_quality" not in first
    assert "source_limitations" not in json.loads(seen[0][-1]["content"])


def test_real_request_changes_only_quality_field_and_fixed_guidance_without_suppressing_candidates():
    first, second = "利润〔原页遮挡〕亿元。", "收入为 元。"
    text = first + second
    meta = metadata(text, "〔原页遮挡〕")
    original_meta = copy.deepcopy(meta)
    baseline = _messages([TextSlice(0, len(text), text)], "synthetic", "", [])
    errors = [{"error_type": "数值缺失", "spans": [{"text": quote}], "reason": "合成候选"} for quote in (first, second)]
    def chat(messages, *, purpose):
        payload = json.loads(messages[-1]["content"])
        hints = payload.pop("source_limitations")
        assert payload == json.loads(baseline[-1]["content"])
        assert messages[0]["content"] == baseline[0]["content"] + SOURCE_LIMITATIONS_GUIDANCE
        assert hints[0]["visible_spans"] == [meta["items"][0]["span"]]
        assert set(hints[0]["declared_span"]) == {"start", "end"}
        return {"content": json.dumps({"errors": errors}, ensure_ascii=False), "trace": {}}
    report = detect_text(text, document_id="synthetic", detector="model_direct", chat=chat, source_limitations=meta)
    assert report["coverage"]["execution_complete"] is True
    assert len(report["errors"]) == len(report["raw_candidates"]) == 2
    assert report["rejected_candidates"] == []
    assert all(error["status"] == "needs_review" for error in report["errors"])
    assert report["source_quality"]["candidate_suppression_allowed"] is False
    assert report["source_quality"]["limitations"][0]["declared_span"] == meta["items"][0]["span"]
    assert meta == original_meta


def test_bad_metadata_fails_before_model_call_even_if_bad_range_would_be_outside_window():
    text = "利润〔原页遮挡〕亿元。"
    meta = metadata(text, "〔原页遮挡〕")
    meta["input_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="input_sha256"):
        detect_text(text, document_id="synthetic", source_limitations=meta,
                    chat=lambda *args, **kwargs: pytest.fail("Invalid metadata must not incur a call"))


def test_wire_projection_does_not_leak_declared_quote_outside_context():
    text = "左界甲乙外窗隐藏词丙丁右界"
    hints = build_source_limitation_hints(text, metadata(text, "甲乙外窗隐藏词丙丁", role="source_fragment"))
    start = text.index("丙丁")
    context = TextSlice(start, start + 2, text[start:start + 2])
    messages = _messages([context], "synthetic", "", [], source_limitation_hints=hints)
    payload = json.loads(messages[-1]["content"])
    hint, = payload["source_limitations"]
    assert hint["visible_spans"] == [context.to_dict()]
    assert hint["context_complete"] is False
    assert "外窗隐藏词" not in messages[-1]["content"]
    assert hints[0]["declared_span"]["text"] == "甲乙外窗隐藏词丙丁"


def test_unrelated_window_has_exact_default_messages_and_no_quality_guidance():
    text = "利润〔原页遮挡〕亿元。另一项收入为空。"
    hints = build_source_limitation_hints(text, metadata(text, "〔原页遮挡〕"))
    start = text.index("另一项")
    context = TextSlice(start, len(text), text[start:])
    assert _messages([context], "synthetic", "", [], source_limitation_hints=hints) == _messages([context], "synthetic", "", [])


def test_extra_metadata_cannot_be_dropped_to_fake_a_complete_underbudget_review():
    text = "利润〔原页遮挡〕亿元。"
    cap = estimated_input_tokens(_messages([TextSlice(0, len(text), text)], "synthetic", "", []))
    report = detect_text(text, document_id="synthetic", detector="model_direct", max_input_tokens=cap,
                         source_limitations=metadata(text, "〔原页遮挡〕"),
                         chat=lambda *args, **kwargs: pytest.fail("Overbudget augmented messages must not be sent"))
    assert report["coverage"]["execution_complete"] is False
    assert report["coverage"]["processed_chars"] == 0
    assert report["source_quality"]["limitations"]
