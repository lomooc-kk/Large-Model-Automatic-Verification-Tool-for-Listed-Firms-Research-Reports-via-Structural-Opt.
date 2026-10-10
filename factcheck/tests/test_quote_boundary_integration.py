"""Boundary warnings must not change detection, messages, or source localization."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from yjcheck import text_review


def run(source, items, *, decisions=False, detector="hybrid", normalize=True, suppress=False):
    requests = []
    response = json.dumps({"decisions" if decisions else "errors": items}, ensure_ascii=False)

    def client(messages, *, purpose):
        requests.append(deepcopy(messages))
        return {"content": response, "trace": {"status": "ok", "purpose": purpose}}

    kwargs = {"document_id": "synthetic", "detector": detector, "chat": client,
              "decision_contract": decisions, "normalize_spans": normalize}
    if suppress:
        with patch.object(text_review, "numeric_quote_boundary_observations", return_value=[]):
            report = text_review.detect_text(source, **kwargs)
    else:
        report = text_review.detect_text(source, **kwargs)
    return report, requests


def candidate(quote, verdict=None, reason="检查原文数据。"):
    value = {"error_type": "格式错误", "spans": [{"text": quote}], "reason": reason}
    if verdict is not None:
        value["verdict"] = verdict
    return value


def assert_only_observations_change(source, items, **kwargs):
    result, messages = run(source, items, **kwargs)
    baseline, old_messages = run(source, items, suppress=True, **kwargs)
    assert messages == old_messages
    retained = deepcopy(result)
    observations = retained.pop("quote_boundary_observations")
    assert baseline.pop("quote_boundary_observations") == []
    assert retained == baseline
    return result, observations


@pytest.mark.parametrize("detector", ["hybrid", "model_direct"])
@pytest.mark.parametrize("normalize", [True, False])
def test_positive_quote_cut_only_adds_audit_not_failure(detector, normalize):
    source = "营业利润 | 3.1% | 48.3% | 524.9 | -92.8%"
    items = [candidate(source[:-1])]
    result, observations = assert_only_observations_change(source, items, detector=detector, normalize=normalize)
    assert len(observations) == 1
    item = observations[0]
    assert item["candidate_ref"] == "model:0:0" and item["decision_ref"] is None
    assert item["raw_index"] == item["span_index"] == item["job_index"] == 0
    assert item["omitted_span"]["text"] == "%" and item["severity"] == "warning"
    assert item["span_basis"] == "anchored_raw_span_order"
    assert result["coverage"]["execution_complete"] is True
    assert result["coverage"]["candidate_quality_complete"] is True
    assert result["raw_candidates"][0]["candidate"] == items[0]


def test_all_three_verdicts_keep_original_indices_and_own_quote_evidence():
    source = "甲增长率12.3%。乙增长率45.6%。丙增长率78.9%。"
    items = [candidate("甲增长率12.3", "no_error"), candidate("乙增长率45.6", "error_supported"),
             candidate("丙增长率78.9", "insufficient_evidence")]
    result, observations = assert_only_observations_change(source, items, decisions=True)
    assert [x["decision_ref"] for x in observations] == ["decision:0:0", "decision:0:1", "decision:0:2"]
    assert [x["candidate_ref"] for x in observations] == [None, "model:0:1", None]
    assert [x["verdict"] for x in observations] == [x["verdict"] for x in items]
    assert [x["raw_index"] for x in observations] == [0, 1, 2]
    assert [x["candidate_ref"] for x in result["raw_candidates"]] == ["model:0:1"]
    assert result["decision_contract"]["nonerror_decision_count"] == 2
    assert result["coverage"]["candidate_quality_complete"] is True


def test_deduplication_retains_two_original_candidate_observations():
    source = "增幅为12.3%。"
    item = candidate("增幅为12.3")
    result, observations = assert_only_observations_change(source, [item, deepcopy(item)])
    assert [x["candidate_ref"] for x in observations] == ["model:0:0", "model:0:1"]
    assert len(result["raw_candidates"]) == 2
    represented = [x["candidate_ref"] for e in result["errors"] for x in e.get("represented_model_candidates", [])]
    assert sorted(represented) == ["model:0:0", "model:0:1"]
    assert len(result["errors"]) == 1


def test_adjacent_span_normalization_does_not_erase_raw_span_index():
    source = "标题甲\n\n金额12.3万元。"
    item = candidate("标题甲")
    item["spans"].append({"text": "金额12.3"})
    result, observations = assert_only_observations_change(source, [item])
    assert observations[0]["span_index"] == 1
    assert observations[0]["quoted_span"]["text"] == "金额12.3"
    assert result["errors"][0]["spans"][0]["text"] == "标题甲\n\n金额12.3"


@pytest.mark.parametrize("quote", ["12.3", "%", "EPS(X)"])
def test_legal_local_number_unit_and_label_do_not_add_observation(quote):
    source = "增长12.3%。每股收益标签EPS(X)。"
    result, observations = assert_only_observations_change(source, [candidate(quote)])
    assert observations == [] and result["coverage"]["candidate_quality_complete"] is True


def test_bad_nonerror_anchor_stays_rejected_and_has_no_false_boundary_evidence():
    result, observations = assert_only_observations_change("增长12.3%。", [candidate("不存在的引文", "no_error")], decisions=True)
    assert observations == []
    assert result["model_decision_dispositions"][0]["disposition"] == "anchor_rejected"
    assert result["coverage"]["candidate_quality_complete"] is False


def test_warning_nonerror_does_not_remove_confirmed_rule():
    source = "2025年2月30日公告，增幅12.3%。"
    result, observations = assert_only_observations_change(source, [candidate("增幅12.3", "no_error")], decisions=True)
    assert len(observations) == 1
    assert any(e["status"] == "confirmed_error" and e["error_type"] == "时间信息非法" for e in result["errors"])
    assert result["raw_candidates"] == [] and result["coverage"]["execution_complete"] is True


def test_observation_has_no_prompt_or_external_source_effect():
    source = "变化1.234%。"
    result, requests = run(source, [candidate("变化1.23")])
    user = json.loads(requests[0][-1]["content"])
    assert "quote_boundary_observations" not in user
    assert result["coverage"]["external_source_documents_used"] is False
    assert result["quote_boundary_observations"][0]["omitted_span"]["text"] == "4%"
