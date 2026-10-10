import json

import pytest

from yjcheck.text_review import detect_text


SOURCE = "公司于2025年4月31日发布2025-2024财年业绩。"
FISCAL_REASON = "财年区间2025-2024起始年晚于结束年，时间范围倒置不成立。"


def row(text, reason, kind="时间信息非法"):
    return {"error_type": kind, "spans": [{"text": text}], "reason": reason}


def run(text, rows, *, detector="hybrid"):
    def chat(messages, *, purpose):
        return {"content": json.dumps({"errors": rows}, ensure_ascii=False), "trace": {"provider": "unit-test"}}
    return detect_text(text, document_id="temporal-source-relations", detector=detector, chat=chat)


def test_fiscal_relation_retyped_without_calendar_promotion_or_guessed_replacement():
    raw = row(SOURCE, FISCAL_REASON)
    report = run(SOURCE, [raw])
    fiscal, = [error for error in report["errors"] if error["error_type"] == "时间矛盾"]
    assert fiscal["status"] == "needs_review"
    assert fiscal["reason"] == FISCAL_REASON
    assert fiscal["source_alignment"]["original_error_type"] == "时间信息非法"
    assert fiscal["source_alignment"]["original_reason"] == FISCAL_REASON
    assert fiscal["represented_model_candidates"][0]["candidate"] == raw
    assert not fiscal.get("verification_spans") and not fiscal.get("verified_by")
    assert len([error for error in report["errors"] if error["detector_id"] == "verified.calendar"]) == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_calendar_and_fiscal_claims_keep_separate_evidence_in_either_order(reverse):
    rows = [row(SOURCE, "2025年4月31日不存在，4月只有30天。"), row(SOURCE, FISCAL_REASON)]
    report = run(SOURCE, rows[::-1] if reverse else rows)
    assert len(report["errors"]) == 2
    assert {error["error_type"] for error in report["errors"]} == {"时间信息非法", "时间矛盾"}
    for error in report["errors"]:
        assert len(error["represented_model_candidates"]) == 1


@pytest.mark.parametrize("reason", [
    "2025年4月31日不存在，2025-2024财年也倒序。",
    "2025-2024财年倒序，而且营收计算错误。",
    "时间信息非法。",
])
def test_mixed_or_unknown_reason_is_not_retyped_or_hidden(reason):
    report = run(SOURCE, [row(SOURCE, reason)])
    model, = [error for error in report["errors"] if error["detector_id"] == "hybrid.model"]
    assert model["error_type"] == "时间信息非法"
    assert model["status"] == "needs_review" and model["reason"] == reason


def test_explicit_legacy_year_period_and_same_model_relation_merge_only_as_review():
    text = "统计期间为2025年至2024年。"
    report = run(text, [row(text, "期间2025年至2024年起始年份晚于结束年份。")])
    assert len(report["errors"]) == 1
    finding = report["errors"][0]
    assert finding["error_type"] == "时间矛盾" and finding["status"] == "needs_review"
    assert set(finding["detector_ids"]) == {"legacy.C.INTRINSIC.002", "hybrid.model"}


def test_direct_model_arm_keeps_original_type_for_honest_comparison():
    result = run(SOURCE, [row(SOURCE, FISCAL_REASON)], detector="model_direct")
    assert len(result["errors"]) == 1
    assert result["errors"][0]["error_type"] == "时间信息非法"
    assert "source_alignment" not in result["errors"][0]
