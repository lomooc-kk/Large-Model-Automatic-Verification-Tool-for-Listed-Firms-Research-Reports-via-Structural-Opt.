"""Synthetic source relation tests; no benchmark IDs, gold or model calls."""
from copy import deepcopy

import pytest

from yjcheck.temporal_alignment import bind_reversed_year_range, same_reversed_year_issue


_REASON = "财年区间2025-2024起始年晚于结束年，时间范围倒置不成立。"
_LEGACY_REASON = "句中年份顺序倒置（晚于起始年份的表述早于起始年份），需人工核对期间。"


def candidate(text, reason=_REASON, *, quote=None, kind="时间信息非法"):
    quote = text if quote is None else quote
    start = text.index(quote)
    return {"error_type": kind, "status": "needs_review", "reason": reason,
            "spans": [{"start": start, "end": start + len(quote), "text": quote}]}


@pytest.mark.parametrize("text,reason", [
    ("2025年4月31日，公司发布2025-2024财年第四季度财报。", _REASON),
    ("公司发布2025—2024财年报告。", "2025-2024财年倒序。"),
    ("统计期间为2025年至2024年。", "期间2025年至2024年起始年份晚于结束年份。"),
    ("公司2025-2024年期间签订合同。", "2025-2024年期间顺序颠倒。"),
    ("报告覆盖2025-2024会计年度。", "会计年度起始年晚于结束年。"),
    ("项目周期：2025年到2024年。", "2025年晚于2024年，属于时间矛盾。"),
])
def test_single_explicit_period_binds_with_review_only_provenance(text, reason):
    c = candidate(text, reason)
    before = deepcopy(c)
    binding = bind_reversed_year_range(c, text)
    assert binding is not None
    assert c == before
    assert binding["canonical_type"] == "时间矛盾" and binding["status"] == "needs_review"
    assert binding["source_alignment"]["original_error_type"] == "时间信息非法"
    assert binding["source_alignment"]["original_reason"] == reason
    assert "verification_spans" not in binding
    for span in binding["source_structure"]["anchors"] + binding["spans"]:
        assert text[span["start"]:span["end"]] == span["text"]
    check = binding["source_structure"]["checks"][0]
    assert check["first_year"] == 2025 and check["last_year"] == 2024
    assert check["business_error_confirmed"] is False
    assert check["corrected_years_inferred"] is False


def test_exact_range_quote_can_use_same_statement_period_semantics():
    text = "公司发布2025-2024财年报告。"
    assert bind_reversed_year_range(candidate(text, quote="2025-2024"), text) is not None


@pytest.mark.parametrize("text,range_text", [
    ("公司2025-2024年间执行合同。", "2025-2024年"),
    ("公司2025年至2024年间执行合同。", "2025年至2024年"),
])
def test_between_years_role_preserves_the_full_source_token(text, range_text):
    c = candidate(text, "期间起始年晚于结束年。")
    before = deepcopy(c)
    binding = bind_reversed_year_range(c, text)
    assert binding is not None and c == before
    structure = binding["source_structure"]
    assert structure["anchors"][0]["text"] == range_text
    role = structure["checks"][0]["role_anchor"]
    assert role["text"] == "年间"
    assert text[role["start"]:role["end"]] == "年间"
    assert role["start"] == structure["anchors"][0]["end"] - 1
    assert binding["evidence"][1]["text"] == "年间"


@pytest.mark.parametrize("length,expected", [(799, True), (800, True), (801, False), (12523, False)])
@pytest.mark.parametrize("narrow", [False, True])
def test_statement_length_guard_abstains_without_truncating_candidate(length, expected, narrow):
    prefix = "公司发布2025-2024财年报告，"
    text = prefix + "业务" * ((length - len(prefix) - 1) // 2)
    text += "甲" * (length - len(text) - 1) + "。"
    assert len(text) == length
    c = candidate(text, quote="2025-2024" if narrow else None)
    before = deepcopy(c)
    binding = bind_reversed_year_range(c, text)
    assert (binding is not None) is expected
    assert c == before
    if binding:
        assert binding["spans"][0]["text"] == text


@pytest.mark.parametrize("reason", [
    "2025年4月31日不存在，4月只有30天。",
    _REASON + "2025年4月31日不存在。",
    "2025-2024财年倒序，而且营收计算错误。",
    "2025-2024财年倒序，应改为2024-2025财年。",
    "2024-2023财年倒序。",
    "并非2025-2024财年倒序。",
    "2025-2024财年可能倒序。",
    "2025-2024财年倒序的说法不正确。",
    "时间信息非法。", "年份不合法。", "", None,
])
def test_calendar_mixed_unknown_and_foreign_allegations_abstain(reason):
    text = "2025年4月31日，公司发布2025-2024财年第四季度财报。"
    assert bind_reversed_year_range(candidate(text, reason), text) is None


@pytest.mark.parametrize("text", [
    "公司营收在2025-2024年分别为3亿元和2亿元。",
    "历史数据从近到远列举2025-2024财年。",
    "按年份降序排列2025-2024财年。",
    "历史回溯2025-2024财年。",
    "例如错误地写成2025-2024财年。",
    "原文误写2025-2024财年，需要更正。",
    "公司发布2025-2024财年和2023-2022财年报告。",
    "公司发布2025-2024财年及2022-2023财年报告。",
    "计算2025-2024财年差值=1。",
    "列出2025、2024财年。",
    "公司发布2024-2025财年报告。",
    "公司发布1999-2001财年报告。",
    "公司发布2025-2025财年报告。",
    "公司发布2025-0000财年报告。",
    "公司发布12025-2024财年报告。",
    "公元前2025-2024年期间。",
])
def test_no_automatic_error_from_descending_year_strings_or_legal_chronology(text):
    assert bind_reversed_year_range(candidate(text), text) is None


def test_old_legal_years_do_not_use_a_modern_year_threshold():
    text = "统计期间为0180年至0179年。"
    reason = "期间起始年晚于结束年。"
    binding = bind_reversed_year_range(candidate(text, reason), text)
    assert binding is not None
    assert binding["source_structure"]["checks"][0]["first_year"] == 180
    assert binding["status"] == "needs_review"


def test_narrow_quote_does_not_hide_a_second_range_in_same_statement():
    text = "公司发布2025-2024财年及2023-2022财年报告。"
    assert bind_reversed_year_range(candidate(text, quote="2025-2024"), text) is None


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(error_type="数值不一致错误"),
    lambda c: c["spans"][0].update(text="不在原文"),
    lambda c: c["spans"][0].update(start=True),
    lambda c: c["spans"][0].update(start=1),
    lambda c: c.update(spans=[]),
    lambda c: c["spans"].append(deepcopy(c["spans"][0])),
])
def test_input_anchor_and_type_contract(mutate):
    text = "公司发布2025-2024财年报告。"
    c = candidate(text)
    mutate(c)
    assert bind_reversed_year_range(c, text) is None


def test_legacy_dedup_requires_same_exact_range_and_single_known_reason():
    text = "2025年4月31日，公司发布2025-2024财年第四季度财报。"
    binding = bind_reversed_year_range(candidate(text), text)
    legacy = {**candidate(text, _LEGACY_REASON, kind="时间矛盾"), "detector_id": "legacy.C.INTRINSIC.002"}
    before = deepcopy((legacy, binding))
    assert same_reversed_year_issue(legacy, binding, text) is True
    assert (legacy, binding) == before
    legacy["reason"] += "同时日期不存在。"
    assert same_reversed_year_issue(legacy, binding, text) is False


@pytest.mark.parametrize("mutation", [
    lambda l, b: l.update(detector_id="hybrid.model"),
    lambda l, b: l.update(status="confirmed_error"),
    lambda l, b: l["spans"][0].update(text="错误引文"),
    lambda l, b: b.update(source_issue_key="wrong"),
    lambda l, b: b.update(source_content_sha256="wrong"),
    lambda l, b: b.update(status="confirmed_error"),
    lambda l, b: b["source_structure"]["anchors"][0].update(start=0),
])
def test_legacy_dedup_rejects_mutated_or_mismatched_binding(mutation):
    text = "公司发布2025-2024财年报告。"
    binding = bind_reversed_year_range(candidate(text), text)
    legacy = {**candidate(text, _LEGACY_REASON, kind="时间矛盾"), "detector_id": "legacy.C.INTRINSIC.002"}
    mutation(legacy, binding)
    assert same_reversed_year_issue(legacy, binding, text) is False


def test_legacy_same_wording_in_another_statement_is_another_source_issue():
    text = "甲公司发布2025-2024财年报告。乙公司发布2025-2024财年报告。"
    first, second = text.split("。")[:2]
    binding = bind_reversed_year_range(candidate(text, quote=first + "。"), text)
    legacy = {**candidate(text, _LEGACY_REASON, quote=second + "。", kind="时间矛盾"),
              "detector_id": "legacy.C.INTRINSIC.002"}
    assert same_reversed_year_issue(legacy, binding, text) is False
