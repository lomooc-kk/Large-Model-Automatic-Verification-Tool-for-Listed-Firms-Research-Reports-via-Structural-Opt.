"""Synthetic source-bound calendar identity tests; no model or benchmark gold."""
from copy import deepcopy

import pytest

from yjcheck.calendar_alignment import calendar_allegation_matches


def case(text="2025年4月31日，公司发布2025-2024财年报告。", token="2025年4月31日",
         reason="2025年4月仅有30日，4月31日不是合法日期。", *, exact=False):
    start = text.index(token)
    proof = {"start": start, "end": start + len(token), "text": token}
    wide = {"start": 0, "end": len(text), "text": text}
    year, rest = token.split("年")
    month, day = rest.split("月")
    finding = {"error_type": "时间信息非法", "detector_id": "verified.calendar",
               "status": "confirmed_error", "spans": [wide], "verification_spans": [proof],
               "evidence": [{"kind": "source_text", **proof},
                            {"kind": "deterministic", "check": "calendar_date",
                             "year": int(year), "month": int(month), "day": int(day[:-1])}]}
    candidate = {"error_type": "时间信息非法", "status": "needs_review",
                 "spans": [deepcopy(proof if exact else wide)], "reason": reason}
    return candidate, finding, text


@pytest.mark.parametrize("reason", [
    "2025年4月仅有30日，4月31日不是合法日期。",
    "2025年4月31日不是有效公历日期。",
    "该年月日不是有效公历日期。",
    "原文中的2025年4月31日不存在。",
    "4月31日为无效日期。",
    "公历中不存在2025年4月31日。",
    "2025年4月只有30天，因此4月31日不存在。",
    "4月没有31日。",
    "“2025年04月31日”不是合法的公历日期。",
])
@pytest.mark.parametrize("exact", [False, True])
def test_calendar_only_allegation_binds_without_mutation(reason, exact):
    args = case(reason=reason, exact=exact)
    before = deepcopy(args)
    assert calendar_allegation_matches(*args) is True
    assert args == before


@pytest.mark.parametrize("reason", [
    "2025年2月30日不存在，不是合法日期。",
    "2025年2月30日不是有效公历日期，是不存在的日期。",
    "2025年2月30日不存在，因此不是合法日期。",
])
def test_subjectless_invalid_restatement_requires_preceding_bound_allegation(reason):
    assert calendar_allegation_matches(*case(
        "公司于2025年2月30日发布公告。", "2025年2月30日", reason)) is True


@pytest.mark.parametrize("reason", [
    "不是合法日期。",
    "不是合法日期，2025年2月30日不存在。",
    "2025年2月只有28天，不是合法日期。",
    "2025年2月30日不存在，2025年3月30日不是合法日期。",
    "2025年2月30日不存在，另一日期不是合法日期。",
    "2025年2月30日不存在，不是合法日期，且财年倒序。",
    "2025年2月30日不存在，不是合法日期的说法不正确。",
])
def test_subjectless_clause_does_not_introduce_or_switch_subject(reason):
    assert calendar_allegation_matches(*case(
        "公司于2025年2月30日发布公告。", "2025年2月30日", reason)) is False


@pytest.mark.parametrize("reason", [
    "财年区间2025-2024起始年晚于结束年，时间范围倒置不成立。",
    "2025年4月31日不存在，且2025-2024财年倒序。",
    "2025年4月31日不存在；起始年晚于结束年。",
    "2025年4月31日不存在，公司营收也有错误。",
    "2025年4月31日不存在，应改为2025年4月30日。",
    "2025年4月31日不存在，2026年6月31日也不存在。",
    "2026年6月31日不是合法日期。",
    "2025年4月31日晚于报告发布日。",
    "2025年4月31日不是错误。",
    "2025年4月31日是合法日期。",
    "并非2025年4月31日不存在。",
    "2025年4月31日可能不存在。",
    "2025年4月31日不存在的说法不正确。",
    "示例中2025年4月31日不存在。",
    "2025年4月共有31日，4月31日不存在。",
    "2026年4月只有30天，4月31日不存在。",
    "2025年4月只有30天。",
    "", "时间信息错误。",
])
@pytest.mark.parametrize("exact", [False, True])
def test_independent_or_mixed_allegation_never_inherits_calendar_proof(reason, exact):
    assert calendar_allegation_matches(*case(reason=reason, exact=exact)) is False


@pytest.mark.parametrize("token,reason,expected", [
    ("2023年2月29日", "2023年不是闰年，2月29日不存在。", True),
    ("2024年2月30日", "2024年是闰年，2月只有29天，2月30日不存在。", True),
    ("1900年2月29日", "1900年不是闰年，2月29日不存在。", True),
    ("2024年2月29日", "2月29日不存在。", False),
    ("2000年2月29日", "2000年不是闰年，2月29日不存在。", False),
    ("2023年2月29日", "2023年是闰年，2月29日不存在。", False),
    ("2025年4月30日", "4月30日不存在。", False),
    ("2025年13月1日", "2025年13月1日不是有效公历日期。", True),
])
def test_real_calendar_validation_and_explanatory_facts(token, reason, expected):
    assert calendar_allegation_matches(*case(token + "发布公告。", token, reason)) is expected


@pytest.mark.parametrize("other", ["2025年5月1日", "2025年6月31日", "2025年4月31日",
                                 "5月1日", "6月31日", "2025.6.31", "2025-05-01"])
def test_broad_quote_with_another_full_date_is_not_single_source_proof(other):
    args = case("2025年4月31日公告，后文另述" + other + "。")
    assert calendar_allegation_matches(*args) is False


@pytest.mark.parametrize("reason", [
    "2025年4月31日不存在，4月只有30天。",
    "2025年04月31日不是有效公历日期。",
])
def test_explicit_target_alone_can_bind_despite_other_legal_full_dates(reason):
    text = "2025年5月26日晚间发布公告，统计期为2025年4月31日至2025年5月26日。"
    assert calendar_allegation_matches(*case(text, reason=reason)) is True


@pytest.mark.parametrize("reason", [
    "该日期不存在，4月只有30天。",
    "4月31日不存在，4月只有30天。",
    "2025年5月26日不存在。",
    "2025年4月31日不存在，起始日期早于发布日。",
    "2025年4月31日不存在，2025年5月26日也不存在。",
])
def test_other_legal_dates_do_not_allow_ambiguous_or_compound_reasons(reason):
    text = "2025年5月26日晚间发布公告，统计期为2025年4月31日至2025年5月26日。"
    assert calendar_allegation_matches(*case(text, reason=reason)) is False


@pytest.mark.parametrize("other", ["2025年6月31日", "2025年4月31日", "5月26日", "2025-05-26"])
def test_explicit_target_does_not_allow_other_invalid_or_partial_dates(other):
    text = "2025年4月31日公告，后文另述" + other + "。"
    assert calendar_allegation_matches(*case(text, reason="2025年4月31日不存在。")) is False


def test_narrow_quote_remains_bound_when_other_date_is_outside_candidate():
    args = case("2025年4月31日公告，后文另述2025年6月31日。", exact=True)
    assert calendar_allegation_matches(*args) is True


@pytest.mark.parametrize("mutation", [
    lambda c, f: c["spans"][0].update(start=1),
    lambda c, f: c["spans"][0].update(text="不存在的原文"),
    lambda c, f: c["spans"][0].update(start=True),
    lambda c, f: c.update(spans=[]),
    lambda c, f: c["spans"].append(deepcopy(c["spans"][0])),
    lambda c, f: c.update(reason=None),
    lambda c, f: c.update(error_type="时间矛盾"),
    lambda c, f: f.update(detector_id="verified.month_range"),
    lambda c, f: f.update(status="needs_review"),
    lambda c, f: f.update(error_type="时间矛盾"),
    lambda c, f: f.update(verification_spans=[]),
    lambda c, f: f["verification_spans"][0].update(end=9),
    lambda c, f: f["spans"][0].update(text="不存在的原文"),
    lambda c, f: f.update(evidence=[]),
    lambda c, f: f["evidence"][0].update(text="错误源文本"),
    lambda c, f: f["evidence"][0].update(start=1),
    lambda c, f: f["evidence"].pop(0),
    lambda c, f: f["evidence"][1].update(day=30),
    lambda c, f: f["evidence"][1].update(month="4"),
    lambda c, f: f["evidence"].append(deepcopy(f["evidence"][1])),
])
def test_unverified_or_malformed_source_proof_is_rejected(mutation):
    c, f, text = case()
    mutation(c, f)
    assert calendar_allegation_matches(c, f, text) is False


def test_candidate_cannot_borrow_proof_outside_its_quote():
    c, f, text = case()
    c["spans"] = [{"start": 11, "end": len(text), "text": text[11:]}]
    assert calendar_allegation_matches(c, f, text) is False
