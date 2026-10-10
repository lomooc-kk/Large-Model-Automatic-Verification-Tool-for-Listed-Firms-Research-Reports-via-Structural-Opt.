from copy import deepcopy

import pytest

from yjcheck.quote_boundaries import numeric_quote_boundary_observations as observe


def anchored(source, quote):
    assert source.count(quote) == 1
    start = source.index(quote)
    return {"start": start, "end": start + len(quote), "text": quote}


def observations(source, quote):
    return observe(source, [anchored(source, quote)])


def check_evidence(source, result):
    for item in result:
        assert item["is_anchor_failure"] is False and item["business_error_proven"] is False
        for field in ("quoted_span", "numeric_span", "token_span", "omitted_span"):
            span = item[field]
            assert source[span["start"]:span["end"]] == span["text"]


@pytest.mark.parametrize("unit", ["%", "％", "‰", "‱", "万元", "亿元", "元/股", "万吨", "倍", "pct", "bps", "个百分点"])
def test_context_omits_explicit_adjacent_unit(unit):
    source = f"预测增量为12.3{unit}。"
    result = observations(source, "预测增量为12.3")
    assert len(result) == 1 and result[0]["code"] == "contextual_quantity_unit_cut"
    assert result[0]["omitted_span"]["text"] == unit
    check_evidence(source, result)


@pytest.mark.parametrize("source,quote,omitted", [
    ("利润率为-92.8%。", "利润率为-92.8", "%"),
    ("利润率为１２．３％。", "利润率为１２．３", "％"),
    ("金额为1,234.56万元。", "金额为1,234", ".56万元"),
    ("金额为123.45万元。", "金额为123.4", "5万元"),
    ("变动为1.2e-3%。", "变动为1.2e-", "3%"),
    ("预测值-12.3%低于去年。", "12.3%低于去年。", "-"),
    ("金额为12.3万元。", "金额为12.3万", "元"),
    ("每股收益12.3元/股。", "每股收益12.3元", "/股"),
    ("收益率为12.3 %。", "收益率为12.3", " %"),
])
def test_numeric_sign_decimal_and_unit_cuts(source, quote, omitted):
    result = observations(source, quote)
    assert len(result) == 1 and result[0]["omitted_span"]["text"] == omitted
    check_evidence(source, result)


def test_two_table_quotes_cut_percent_are_separate_and_no_rewrite():
    source = "营业利润 | 3.1% | 48.3% | 524.9 | -92.8%\n\n归属于母公司净利 | 2.8% | 48.0% | 517.8 | -93.1%"
    spans = [anchored(source, row[:-1]) for row in source.split("\n\n")]
    before = deepcopy(spans)
    result = observe(source, spans)
    assert [r["span_index"] for r in result] == [0, 1]
    assert all(r["omitted_span"]["text"] == "%" for r in result)
    assert spans == before
    check_evidence(source, result)


@pytest.mark.parametrize("source,quote", [
    ("利率为12.3%。", "12.3"),
    ("利率为12.3%。", "12"),  # Intentional digit/precision selection remains legal.
    ("利率为12.3%。", ".3"),
    ("利率为-12.3%。", "12.3%"),
    ("利率为12.3%。", "%"),
    ("金额12.3万元。", "万元"),
    ("金额12.3万元。", "元"),
    ("EPS(X) | 5.78 | 6.21", "EPS(X)"),
    ("PE(X) | 12.3", "PE(X)"),
    ("收入为12.3亿元。", "收入为12.3亿元"),
    ("目标价为12.3\n单位：元", "目标价为12.3"),
    ("读者12.3日到访。", "读者12.3"),
    ("编号123项目已备案。", "编号123"),
    ("合作方123元宵公司。", "合作方123"),
    ("第123股东提出意见。", "第123"),
    ("计数12.3bpm。", "计数12.3"),
    ("版本v12.34修复。", "版本v12"),
    ("版本12.34.56修复。", "版本12.3"),
    ("金额12,3456万元。", "金额12,345"),
    ("变化从3-5亿元区间调整。", "5亿元区间调整。"),
    ("截至2025年。", "截至2025"),
    ("利率为12.3，下一项。", "利率为12.3"),
])
def test_legal_local_labels_units_and_ambiguous_prose_are_not_flagged(source, quote):
    assert observations(source, quote) == []


def test_left_and_right_cut_preserve_two_observations_and_exact_offsets():
    source = "上期-12.3%增长至456.78万元。"
    result = observations(source, "12.3%增长至456.7")
    assert [r["boundary"] for r in result] == ["left", "right"]
    assert [r["omitted_span"]["text"] for r in result] == ["-", "8万元"]
    check_evidence(source, result)


@pytest.mark.parametrize("bad", [
    [{"start": False, "end": 2, "text": "12"}],
    [{"start": -1, "end": 2, "text": "12"}],
    [{"start": 0, "end": 2, "text": "99"}],
    [{"start": 0, "end": 9, "text": "12%"}],
    [{"text": "12%"}],
    [{"start": 0, "end": 0, "text": ""}],
])
def test_unanchored_input_is_programmer_error_not_candidate_rejection(bad):
    with pytest.raises(ValueError, match="exact anchored"):
        observe("12%", bad)


def test_empty_input_is_not_a_completeness_certificate():
    assert observe("", []) == []
    assert observe("原文", []) == []


def test_extra_span_fields_and_duplicate_input_indices_unchanged():
    source = "增幅12.3%。"
    item = {**anchored(source, "增幅12.3"), "metadata": {"untouched": True}}
    spans = [item, deepcopy(item)];before = deepcopy(spans)
    result = observe(source, spans)
    assert [x["span_index"] for x in result] == [0, 1] and spans == before
