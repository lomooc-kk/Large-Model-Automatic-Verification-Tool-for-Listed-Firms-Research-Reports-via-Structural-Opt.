"""Calendar proof must not erase another allegation at the same source span."""
import json

import pytest

from yjcheck.text_review import detect_text


SOURCE = "公司于2025年4月31日披露2025-2024财年业绩。"
TOKEN = "2025年4月31日"


def candidate(reason, *, exact=False):
    quote = TOKEN if exact else SOURCE
    start = SOURCE.index(quote)
    return {"error_type": "时间信息非法", "reason": reason,
            "spans": [{"start": start, "end": start + len(quote), "text": quote}]}


def review(rows):
    def chat(messages, *, purpose):
        return {"content": json.dumps({"errors": rows}, ensure_ascii=False), "trace": {"provider": "unit-test"}}
    return detect_text(SOURCE, document_id="calendar-independent-allegations", chat=chat)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("reason", [
    "2025-2024财年区间起始年晚于结束年。",
    "2025年4月31日不存在，且2025-2024财年区间倒置。",
    "这是待进一步检查的时间问题。",
])
def test_independent_or_mixed_reason_survives_exact_and_containment_paths(reason, exact):
    result = review([candidate(reason, exact=exact)])
    calendar = [e for e in result["errors"] if e["detector_id"] == "verified.calendar"]
    pending = [e for e in result["errors"] if e["detector_id"] == "hybrid.model"]
    assert len(calendar) == 1 and calendar[0]["status"] == "confirmed_error"
    assert len(pending) == 1 and pending[0]["status"] == "needs_review"
    assert pending[0]["reason"] == reason
    assert not pending[0].get("verified_by")


@pytest.mark.parametrize("reverse", [False, True])
def test_two_model_allegations_on_same_sentence_keep_independent_fiscal_issue(reverse):
    calendar = candidate("2025年4月31日不存在，4月只有30天。")
    fiscal = candidate("2025-2024财年区间起始年晚于结束年。")
    rows = [calendar, fiscal]
    result = review(rows[::-1] if reverse else rows)
    confirmed = [e for e in result["errors"] if e["status"] == "confirmed_error"]
    pending = [e for e in result["errors"] if e["detector_id"] == "hybrid.model" and e["status"] == "needs_review"]
    assert len(confirmed) == 1
    assert "hybrid.model" in confirmed[0]["detector_ids"]
    assert len(pending) == 1 and pending[0]["reason"] == fiscal["reason"]


@pytest.mark.parametrize("exact", [False, True])
def test_calendar_only_candidate_still_merges_with_rule(exact):
    result = review([candidate("2025年4月31日不存在，4月只有30天。", exact=exact)])
    assert len(result["errors"]) == 1
    assert result["errors"][0]["status"] == "confirmed_error"
    assert set(result["errors"][0]["detector_ids"]) == {"hybrid.model", "verified.calendar"}
