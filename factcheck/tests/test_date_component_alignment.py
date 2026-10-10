from copy import deepcopy
import json

import pytest

from yjcheck.date_component_alignment import date_component_allegation_matches
from yjcheck.text_review import _verified_rules, detect_text


def fixture(token="13月", reason="13月不存在。", exact=False):
    source = f"公司于2025年4月31日披露2025-2024财年业绩，并称{token}更新。"
    detector = "verified.month_range" if token.endswith("月") else "verified.day_range"
    finding = next(x for x in _verified_rules(source) if x["detector_id"] == detector)
    quote = token if exact else source
    start = source.index(quote)
    candidate = {"error_type": "时间信息非法", "reason": reason,
                 "spans": [{"start": start, "end": start + len(quote), "text": quote}]}
    return source, finding, candidate


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("token,reason", [
    ("13月", "13月不存在。"), ("13月", "公历中不存在13月，一年只有12个月。"),
    ("19月", "19月超过12月。"),
    ("2月32日", "32日超过31日。"),
    ("2月32日", "32日不存在，每月最多31天。"),
])
def test_only_matching_component_reason_can_inherit_proof(token, reason, exact):
    source, finding, candidate = fixture(token, reason, exact)
    before = deepcopy((finding, candidate))
    assert date_component_allegation_matches(candidate, finding, source)
    assert before == (finding, candidate)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("reason", [
    "2025-2024财年区间倒置。", "13月不存在，财年区间也倒置。", "13月不存在，但公司利润不一致。",
    "13月并非非法月份。", "应当改成12月。", "14月不存在。", "32日超过31日。",
    "13月不存在是假设。", "这个时间可能有问题。", "一年只有12个月。", "13月不代表错误。",
])
def test_mixed_unknown_or_wrong_issue_stays_independent(reason, exact):
    source, finding, candidate = fixture(reason=reason, exact=exact)
    assert not date_component_allegation_matches(candidate, finding, source)


@pytest.mark.parametrize("mutation", ["wrong_source", "bad_proof", "changed_check", "no_check", "extra_span", "unconfirmed"])
def test_untrusted_or_changed_evidence_cannot_authorize_merge(mutation):
    source, finding, candidate = fixture()
    if mutation == "wrong_source": candidate["spans"][0]["text"] += "x"
    elif mutation == "bad_proof": finding["verification_spans"][0]["start"] += 1
    elif mutation == "changed_check": finding["evidence"][1]["text"] = "14月"
    elif mutation == "no_check": finding["evidence"] = finding["evidence"][:1]
    elif mutation == "extra_span": candidate["spans"] *= 2
    elif mutation == "unconfirmed": finding["status"] = "needs_review"
    assert not date_component_allegation_matches(candidate, finding, source)


def test_two_invalid_month_positions_remain_ambiguous():
    source = "公司于13月和13月更新。"
    findings = _verified_rules(source)
    candidate = {"error_type": "时间信息非法", "reason": "13月不存在。",
                 "spans": [{"start": 0, "end": len(source), "text": source}]}
    for finding in findings:
        assert not date_component_allegation_matches(candidate, finding, source)


@pytest.mark.parametrize("token,reason", [("13月", "月份超出1-12范围。"), ("2月32日", "日期超过31日。")])
def test_anaphora_only_when_source_has_one_date_expression(token, reason):
    source = f"公司于{token}更新。"
    finding = _verified_rules(source)[0]
    candidate = {"error_type": "时间信息非法", "reason": reason,
                 "spans": [{"start": 0, "end": len(source), "text": source}]}
    assert date_component_allegation_matches(candidate, finding, source)


@pytest.mark.parametrize("token,reason", [("13月", "月份超出1-12范围。"), ("2月32日", "日期不存在。"),
                                         ("2月32日", "该日期无效。")])
def test_anaphora_among_multiple_dates_stays_pending(token, reason):
    source, finding, row = fixture(token, reason)
    assert not date_component_allegation_matches(row, finding, source)
    result = review(source, [row])
    model = [x for x in result["errors"] if x["detector_id"] == "hybrid.model"]
    assert len(model) == 1 and model[0]["status"] == "needs_review"
    assert model[0]["reason"] == reason


@pytest.mark.parametrize("month,day,ending", [(1, 41, "该日期本身不成立。"), (5, 32, "日期本身不成立，属于非法时间信息。"),
                                             (7, 35, "日期本身不成立。")])
def test_source_bound_month_has_no_out_of_range_day_merges_once(month, day, ending):
    source, finding, row = fixture(f"{month}月{day}日", f"{month}月没有{day}日，{ending}")
    assert date_component_allegation_matches(row, finding, source)
    result = review(source, [row])
    component = [e for e in result["errors"] if e["detector_id"] == "verified.day_range"]
    assert len(component) == 1 and "hybrid.model" in component[0]["detector_ids"]
    assert not any(e["detector_id"] == "hybrid.model" for e in result["errors"])


@pytest.mark.parametrize("reason", ["5月没有32日。", "2月没有32日，财年也倒置。", "2月没有32日，应改为28日。",
                                     "2月没有32日，但这不代表日期错误。", "2月没有32日，利润应为一亿元。"])
def test_month_qualified_reason_must_bind_same_source_and_only_that_allegation(reason):
    source, finding, row = fixture("2月32日", reason)
    assert not date_component_allegation_matches(row, finding, source)


def review(source, rows):
    def chat(messages, *, purpose):
        return {"content": json.dumps({"errors": rows}, ensure_ascii=False), "trace": {"provider": "unit-test"}}
    return detect_text(source, document_id="component-proof-scope", chat=chat)


@pytest.mark.parametrize("token", ["13月", "2月32日"])
@pytest.mark.parametrize("exact", [False, True])
def test_mixed_allegation_survives_promotion_and_deduplication(token, exact):
    reason = "2025年4月31日不存在，且2025-2024财年区间倒置。"
    source, finding, row = fixture(token, reason, exact)
    result = review(source, [row])
    pending = [x for x in result["errors"] if x["detector_id"] == "hybrid.model"]
    assert len(pending) == 1
    assert pending[0]["reason"] == reason
    assert pending[0]["status"] == "needs_review"
    assert not pending[0].get("verified_by")
    assert any(x["detector_id"] == finding["detector_id"] and x["status"] == "confirmed_error" for x in result["errors"])


@pytest.mark.parametrize("token,reason", [("13月", "13月不存在。"), ("2月32日", "32日超过31日。")])
@pytest.mark.parametrize("reverse", [False, True])
def test_pure_component_and_mixed_reason_keep_separate_representatives(token, reason, reverse):
    source, finding, pure = fixture(token, reason)
    mixed = deepcopy(pure)
    mixed["reason"] = reason + "2025-2024财年区间也倒置。"
    rows = [pure, mixed]
    result = review(source, rows[::-1] if reverse else rows)
    pending = [x for x in result["errors"] if x["detector_id"] == "hybrid.model" and x["status"] == "needs_review"]
    assert len(pending) == 1 and pending[0]["reason"] == mixed["reason"]
    component = next(x for x in result["errors"] if x["detector_id"] == finding["detector_id"])
    assert component["status"] == "confirmed_error"
    assert "hybrid.model" in component["detector_ids"]
