"""Production-node synthetic regression tests; no benchmark gold or paid calls."""
from copy import deepcopy
import json

import pytest

from yjcheck.review_hints import all_review_hints
from yjcheck.text_review import detect_text


def raw(text, kind, reason):
    return {"error_type": kind, "spans": [{"text": text}], "reason": reason}


def run(text, candidates, *, detector="hybrid"):
    def chat(messages, *, purpose):
        return {"content": json.dumps({"errors": candidates}, ensure_ascii=False), "trace": {}}
    return detect_text(text, document_id="synthetic-source-fusion", detector=detector, chat=chat)


def test_unnamed_clause_gap_does_not_invent_a_missing_numeric_field():
    text = "第一档次企业8家，销售额均值12亿元。第二档次企业6家，。"
    candidate = raw("第二档次企业6家，。", "数值缺失", "前项有销售额均值，后项句末，。缺少销售额均值。")
    original = deepcopy(candidate)
    report = run(text, [candidate])
    assert len(report["errors"]) == 1
    issue = report["errors"][0]
    assert issue["error_type"] == "金融要素缺失"
    assert issue["source_structure"]["family"] == "missing_unnamed_clause"
    assert issue["status"] == "needs_review" and "verification_spans" not in issue
    assert issue["source_original_verdict"]["status"] == "confirmed_error"
    assert any(c.get("check") == "surface_gap_only" and c["business_field_missing_confirmed"] is False
               for c in issue["source_structure"]["checks"])
    assert "均值" not in issue["reason"]
    assert issue["represented_model_candidates"][0]["candidate"] == original
    assert report["raw_candidates"][0]["candidate"] == original
    assert set(issue["detector_ids"]) == {"verified.suspended_punctuation", "hybrid.model"}


@pytest.mark.parametrize("text,kind,family", [
    ("本期营业收入为，。", "数值缺失", "missing_numeric_slot"),
    ("本次信用评级为，。", "属性值缺失错误", "empty_attribute"),
])
def test_named_empty_field_uses_this_clause_not_sibling_template(text, kind, family):
    report = run(text, [raw(text, "金融要素缺失", "明确字段后为空，缺少字段值。")])
    assert len(report["errors"]) == 1
    issue = report["errors"][0]
    assert issue["error_type"] == kind and issue["source_structure"]["family"] == family
    assert issue["status"] == "confirmed_error"
    assert issue["source_structure"]["checks"][-1]["field_explicitly_named"] is True
    assert "replacement" not in issue


def test_same_type_same_quote_empty_title_gets_one_representative_hint():
    text = "管理部门印发《》，要求加强管理。"
    candidates = [raw(text, "属性值缺失错误", "书名号内文件名称缺失。")]
    report = run(text, candidates)
    assert len(report["errors"]) == len(all_review_hints(report)["errors"]) == 1
    issue = report["errors"][0]
    assert issue["verification_spans"][0]["text"] == "《》"
    assert len(report["model_candidate_representation"]) == 1
    assert report["model_candidate_representation"][0]["error_id"] == issue["id"]
    assert issue["represented_model_candidates"][0]["candidate"] == candidates[0]


def test_same_sentence_two_empty_slots_are_not_merged_by_same_type_and_span():
    text = "本月先印发《》，再印发《》。"
    report = run(text, [raw(text, "属性值缺失错误", "书名号内文件名称缺失。")])
    rules = [e for e in report["errors"] if e["detector_id"] == "verified.empty_placeholder"]
    model = [e for e in report["errors"] if e["detector_id"] == "hybrid.model"]
    assert len(rules) == 2 and len({e["source_issue_key"] for e in rules}) == 2
    assert len(model) == 1 and model[0]["status"] == "needs_review"
    assert model[0]["source_alignment"]["status"] == "ambiguous_source_structures"
    assert len(all_review_hints(report)["errors"]) == 3


def test_explicit_unique_placeholder_can_select_one_of_two_distinct_slots():
    text = "本月印发《》，并将评级填写为（）。"
    report = run(text, [raw(text, "属性值缺失错误", "《》内文件名称缺失。")])
    assert len(report["errors"]) == 2
    chosen = [e for e in report["errors"] if e.get("represented_model_candidates")]
    assert len(chosen) == 1 and chosen[0]["verification_spans"][0]["text"] == "《》"


def test_full_sentence_about_other_filled_metric_is_not_replaced_by_empty_title():
    text = "公司印发《》，净利润为60万元。"
    report = run(text, [raw(text, "数值缺失", "净利润缺少具体数值。")])
    assert len(report["errors"]) == 2
    model = next(e for e in report["errors"] if e["detector_id"] == "hybrid.model")
    assert model["error_type"] == "数值缺失" and model["status"] == "needs_review"
    assert "source_issue_key" not in model


@pytest.mark.parametrize("detector", ["hybrid", "model_direct"])
def test_two_distinct_term_errors_in_same_wide_quote_are_not_swallowed(detector):
    text = "公司建设人行机器人供应链，并采用Jeston Thor芯片。"
    candidates = [raw(text, "术语误用", "人行机器人应为人形机器人。"),
                  raw(text, "术语误用", "Jeston Thor中的Jeston拼写应核为Jetson。")]
    report = run(text, candidates, detector=detector)
    assert len(report["errors"]) == 2
    assert len({e["id"] for e in report["errors"]}) == 2
    assert {e["reason"] for e in report["errors"]} == {e["reason"] for e in candidates}
    assert len(all_review_hints(report)["errors"]) == 2
    assert len({item["error_id"] for item in report["model_candidate_representation"]}) == 2


def test_exact_duplicate_model_allegations_still_collapse_without_losing_raw_occurrences():
    text = "这里使用某个专业术语。"
    candidate = raw(text, "术语误用", "某个专业术语需要核查。")
    report = run(text, [candidate, candidate])
    assert len(report["errors"]) == 1
    assert len(report["raw_candidates"]) == len(report["model_candidate_representation"]) == 2
    assert len(report["errors"][0]["represented_model_candidates"]) == 2


def test_paired_count_uses_verified_counts_and_merges_broader_quote():
    text = "公司收入8万元/9万元，YOY依次为1%/2%；归母净利润2万元/3万元，YOY依次为1%/2%/3%。"
    candidate = raw(text, "数值不一致错误", "归母净利润的序列数量不一致，两个金额对应三个同比。")
    report = run(text, [candidate])
    assert len(report["errors"]) == 1
    issue = report["errors"][0]
    assert issue["status"] == "confirmed_error"
    proof = next(p for p in issue["source_structure"]["checks"] if p["check"] == "aligned_sequence_cardinality")
    assert (proof["value_count"], proof["rate_count"]) == (2, 3)
    assert issue["spans"][0]["text"].startswith("归母净利润")
    assert issue["represented_model_candidates"][0]["candidate"] == candidate


def test_two_bad_series_do_not_share_a_single_broad_unqualified_proof():
    text = "收入8万元/9万元，YOY依次为1%/2%/3%；利润2万元/3万元，YOY依次为1%/2%/3%。"
    report = run(text, [raw(text, "数值不一致错误", "金额与同比序列数量不一致。")])
    rules = [e for e in report["errors"] if e["detector_id"] == "verified.aligned_series_cardinality"]
    model = [e for e in report["errors"] if e["detector_id"] == "hybrid.model"]
    assert len(rules) == 2 and len(model) == 1
    assert model[0]["status"] == "needs_review"


def test_good_income_series_does_not_inherit_bad_profit_series_proof():
    text = "营业收入8万元/9万元，YOY依次为1%/2%；归母净利润2万元/3万元，YOY依次为1%/2%/3%。"
    report = run(text, [raw(text, "数值不一致错误", "营业收入序列数量不一致。")])
    assert len(report["errors"]) == 2
    model = next(e for e in report["errors"] if e["detector_id"] == "hybrid.model")
    assert model["status"] == "needs_review" and "source_issue_key" not in model


def test_legal_but_reversed_dates_change_type_without_confirming_a_replacement():
    text = "公司2024年8-2月收入增长。"
    original = raw(text, "时间信息非法", "2024年8-2月期间倒序，应改为2-8月。")
    report = run(text, [original])
    assert len(report["errors"]) == 1
    issue = report["errors"][0]
    assert issue["error_type"] == "时间矛盾" and issue["status"] == "needs_review"
    assert "2-8月" not in issue["reason"] and "verification_spans" not in issue
    assert issue["represented_model_candidates"][0]["candidate"] == original


@pytest.mark.parametrize("text", ["跨年期间为2024年12-1月。", "公司2024年13月收入增长。", "更正示例：2024年8-2月。"])
def test_cross_period_illegal_endpoint_and_example_do_not_get_reversed_range_alignment(text):
    report = run(text, [raw(text, "时间信息非法", "月份区间倒序，期间不成立。")])
    assert not any(e.get("source_structure", {}).get("family") == "reversed_legal_date_range" for e in report["errors"])
    assert not any(a.get("original_error_type") == "时间信息非法" for e in report["errors"] for a in e.get("source_alignments", []))


def test_same_sentence_date_and_gap_are_independent():
    text = "公司于2023年2月30日印发《》。"
    candidates = [raw(text, "时间信息非法", "2023年2月30日不存在。"), raw(text, "属性值缺失错误", "书名号内文件名称缺失。")]
    report = run(text, candidates)
    assert len(report["errors"]) == 2
    assert {e["error_type"] for e in report["errors"]} == {"时间信息非法", "属性值缺失错误"}
    assert all(len(e["represented_model_candidates"]) == 1 for e in report["errors"])


def test_rejected_hint_stays_visible_beside_fused_valid_hint():
    text = "管理部门印发《》。"
    report = run(text, [raw(text, "属性值缺失错误", "书名号内名称缺失。"), raw("原文不存在的引文", "属性值缺失错误", "另有名称缺失。")])
    assert len(report["errors"]) == 1 and len(report["rejected_candidates"]) == 1
    hints = all_review_hints(report)["errors"]
    assert len(hints) == 2 and sum(bool(e.get("invalid_anchor")) for e in hints) == 1
    assert len(report["model_candidate_representation"]) == 1


@pytest.mark.parametrize("text", ["板块在申万31个一级行业中排名第50。", "板块表现在31个一级行业中位居第40位。", "板块在申万31个一级子行业中排名第41位。"])
def test_rank_over_source_total_has_a_numeric_proof_not_a_type_alias(text):
    candidate = raw(text, "语义逻辑矛盾", "行业排名超过所列总数，名次不可能。")
    report = run(text, [candidate])
    assert len(report["errors"]) == 1
    issue = report["errors"][0]
    assert issue["error_type"] == "数值不一致错误" and issue["status"] == "confirmed_error"
    proof = next(p for p in issue["source_structure"]["checks"] if p["check"] == "rank_within_explicit_total")
    assert proof["total_count"] == 31 and proof["rank"] > 31
    assert issue["represented_model_candidates"][0]["candidate"] == candidate


@pytest.mark.parametrize("text", [
    "在31个一级行业中排名第30位。", "在前31个一级行业中排名第50位。", "在前 31个一级行业中排名第50位。",
    "前31名中第50家公司另列。", "31个一级行业中甲公司排名第50。", "在31个一级行业中排名第50家公司。",
    "在31个一级行业中排名第50.5位。", "在31个一级行业中排名第50-60位。", "错误示例：在31个一级行业中排名第50位。",
])
def test_rank_rule_rejects_other_populations_top_n_and_unsupported_forms(text):
    report = run(text, [])
    assert not any(e["detector_id"] == "verified.rank_within_total" for e in report["errors"])


def test_short_year_is_not_declared_illegal_without_period_evidence():
    report = run("129年年初至今公司收入稳定。", [])
    assert not any(e["error_type"] == "时间信息非法" for e in report["errors"])


@pytest.mark.parametrize("token,missing", [
    ("2025年7月日", ["day"]), ("2025年月18日", ["month"]),
    ("2025年月日", ["month", "day"]), ("2025年 7月 日", ["day"]),
])
def test_explicit_date_blank_has_narrow_source_proof_without_guessing(token, missing):
    text = f"公司截至{token}已完成项目。"
    report = run(text, [raw(text, "时间信息非法", f"{token}日期不完整，缺少月份或日的数值。")])
    assert len(report["errors"]) == 1
    issue = report["errors"][0]
    assert issue["error_type"] == "数值缺失" and issue["status"] == "confirmed_error"
    assert issue["verification_spans"][0]["text"] == token
    proof = next(e for e in issue["source_structure"]["checks"] if e["check"] == "missing_date_numeric_slot")
    assert proof["missing_fields"] == missing
    assert len(issue["represented_model_candidates"]) == 1
    assert "replacement" not in issue and "不推定" in issue["reason"]


@pytest.mark.parametrize("text", [
    "公司2025年7月日均销量稳定。", "公司2025年7月日常运营正常。",
    "公司2025年7月日产能达到100吨。", "公司2025年7月日内涨幅超过1%。",
    "公司2025年7月日报已发布。", "公司2025年7月日本市场收入增长。",
    "公司2025年7月日期安排已确定。", "公司2025年7月日交易额增长。",
    "公司2024年12月日 均产量稳定。", "公司2024年12月日\n均产量稳定。",
    "公司2024年12月日\t交易额增长。", "公司2024年12月日 \n 本市场收入增长。",
    "2024年12月日经225指数上涨3%。", "2024年12月日立公司发布新产品。",
    "2024年12月日清食品收入增长。", "2024年12月日月股份收入增长。",
    "2024年12月日和食品公司发布报告。", "2024年12月日某新品牌收入增长。",
    "请按2025年月日格式填写。", "模板内容：2025年7月日。", "错误示例：2025年月日。",
    "公司2025年13月日公告。", "公司2025年月32日公告。", "公司2025年7月18日公告。",
    "公司二零二五年七月十八日公告。", "年月日均应按实际情况填写。",
])
def test_date_blank_abstains_on_daily_words_templates_and_invalid_filled_fields(text):
    assert not any(e["detector_id"] == "verified.date_numeric_gap" for e in run(text, [])["errors"])


def test_date_gap_does_not_swallow_impossible_filled_date_or_unrelated_field():
    text = "公司2025年7月日公告，交付日为2025年2月30日，营业收入为100万元。"
    report = run(text, [raw(text, "时间信息非法", "2025年2月30日不存在，不是合法日期。"),
                        raw(text, "数值缺失", "营业收入缺少数值。")])
    assert len(report["errors"]) == 3
    blank = next(e for e in report["errors"] if e["detector_id"] == "verified.date_numeric_gap")
    assert not blank.get("represented_model_candidates")


@pytest.mark.parametrize("separator", [" ", "\n", "\t"])
def test_date_gap_whitespace_does_not_hide_an_actual_day_blank(separator):
    text = f"公司2024年12月日{separator}公告项目已经完工。"
    gaps = [e for e in run(text, [])["errors"] if e["detector_id"] == "verified.date_numeric_gap"]
    assert len(gaps) == 1 and gaps[0]["verification_spans"][0]["text"] == "2024年12月日"


@pytest.mark.parametrize("tail", ["。", "", "和11月28日转入正式运行。", "已完成项目。", "将发布公告。", "拟召开会议。"])
def test_day_blank_requires_a_date_closure_connection_or_explicit_event(tail):
    text = "试点于2024年11月日" + tail
    gaps = [e for e in run(text, [])["errors"] if e["detector_id"] == "verified.date_numeric_gap"]
    assert len(gaps) == 1 and gaps[0]["verification_spans"][0]["text"] == "2024年11月日"


def test_date_gap_does_not_select_one_of_two_unqualified_blanks():
    text = "公司2025年7月日公告，2025年8月日交付。"
    report = run(text, [raw(text, "时间信息非法", "日期缺少具体日数。")])
    assert len(report["errors"]) == 3
    model = next(e for e in report["errors"] if e["detector_id"] == "hybrid.model")
    assert model["source_alignment"]["status"] == "ambiguous_source_structures"


@pytest.mark.parametrize("reason", [
    "2025年7月日日期缺少具体日数，而且2028年开工晚于2027年竣工，期间存在矛盾。",
    "2025年7月日日期缺少具体日数，工期的前后顺序不一致。",
    "2025年7月日日期缺少具体日数，2028年开工时间需核对。",
    "2025年7月日日期缺少具体日数，另外项目的期间也存在错误。",
])
def test_date_gap_preserves_extra_chronology_or_date_allegations(reason):
    text = "公司于2025年7月日发布公告，拟2028年开工，预计2027年竣工。"
    candidate = raw(text, "时间信息非法", reason)
    report = run(text, [candidate])
    gap = next(e for e in report["errors"] if e["detector_id"] == "verified.date_numeric_gap")
    model = next(e for e in report["errors"] if e["detector_id"] == "hybrid.model")
    assert gap["status"] == "confirmed_error" and not gap.get("represented_model_candidates")
    assert model["status"] == "needs_review" and model["reason"] == reason
    assert "source_issue_key" not in model and "verification_spans" not in model
    assert model["represented_model_candidates"][0]["candidate"] == candidate
    assert len(all_review_hints(report)["errors"]) == 2


def test_date_blank_does_not_absorb_same_year_other_filled_date_claim():
    text = "公司于2025年7月日发布公告，交付日期为2025年8月12日。"
    reason = "2025年7月日日期缺少具体日数，2025年8月12日交付日期待补充。"
    report = run(text, [raw(text, "时间信息非法", reason)])
    assert len(all_review_hints(report)["errors"]) == 2
    assert next(e for e in report["errors"] if e["detector_id"] == "hybrid.model")["reason"] == reason


def test_source_structure_does_not_confirm_external_unit_allegations():
    text = "企业营业收入12830亿元，净利润39万元。"
    candidate = raw(text, "数值单位错误", "营业收入很大，所以净利润单位应为亿元。")
    report = run(text, [candidate])
    issue = next(e for e in report["errors"] if e["detector_id"] == "hybrid.model")
    assert issue["status"] == "needs_review" and "source_issue_key" not in issue
    assert issue["reason"] == candidate["reason"]
