from decimal import Decimal
import json

import pytest

from yjcheck.proportion_growth_context import proportion_growth_checks


SOURCE = "其中AIGC内容贡献比例由上年的31%提升至44%，同比增长44%。"


def test_rounded_display_allows_reported_growth_and_preserves_sources():
    check, = proportion_growth_checks(SOURCE, offset=50)
    assert check["rounding_compatible"] is True and check["relation_proven"] is False
    assert Decimal(check["possible_growth_percent"][0]) < 44 < Decimal(check["possible_growth_percent"][1])
    assert check["point_percentage_point_change"] == "13"
    for span in [check["source"], *check["operand_sources"].values()]:
        assert SOURCE[span["start"]-50:span["end"]-50] == span["text"]
    # A constructive witness, independent of the interval formula.
    assert Decimal("30.7").quantize(Decimal("1")) == 31
    assert Decimal("44.208").quantize(Decimal("1")) == 44
    assert (Decimal("44.208") / Decimal("30.7") - 1) * 100 == 44
    json.dumps(check)


@pytest.mark.parametrize("source,compatible", [
    (SOURCE.replace("31%", "31.0%"), True),
    ("比例由31.0%提升至44.0%，同比增长44%。", False),
    ("占比由上年的31%提升至44%，同比增长80%。", False),
    ("比例从上月的40%下降至30%，环比下降25%。", True),
    ("比例从上月的40%下降至30%，环比增长-25%。", True),
    ("毛利率由上年的20.00%提升至30.00%，同比增长50.00%。", True),
])
def test_display_precision_and_signed_relative_change(source, compatible):
    result, = proportion_growth_checks(source)
    assert result["rounding_compatible"] is compatible


@pytest.mark.parametrize("source", [
    "比例由31%提升至44%，同比增长44个百分点。",
    "比例由31%提升至44%，同比增长44pct。",
    "比例由上年的31%提升至44%，环比增长44%。",
    "比例由上月的31%提升至44%，同比增长44%。",
    "预计比例由31%提升至44%，同比增长44%。",
    "预计，比例由31%提升至44%，同比增长44%。",
    "例如，比例由31%提升至44%，同比增长44%。",
    "比例由约31%提升至44%，同比增长44%。",
    "精确比例由31%提升至44%，同比增长44%。",
    "比例由31%提升至44%，同比增长44%至50%。",
    "比例由31%提升至44%，同比增长44%，50%。",
    "比例由31%提升至44%，同比增长44%，约50%。",
    "比例由31%提升至44%，同比增长44%/50%。",
    "比例由31%提升至44%，同比增长44% to 50%。",
    "比例由31%提升至44%，同比增长44%个百分点。",
    "比例由31%提升至44%，同比增长44%及50%。",
    "比例由31%提升至44%，同比增长44.123456789%。",
    "比例由31%提升至144%，同比增长44%。",
    "比例由0%提升至44%，同比增长44%。",
    "比例由-31%提升至44%，同比增长44%。",
    "比例由40%下降至30%，环比下降-25%。",
    "比例由31%提升至44%。收入同比增长44%。",
    "比例由31%提升至44%，\n同比增长44%。",
])
def test_incomplete_ambiguous_or_different_dimension_abstains(source):
    assert proportion_growth_checks(source) == []


def test_empty_limit_and_multiple_bounded_relations():
    assert proportion_growth_checks(SOURCE, limit=0) == []
    assert len(proportion_growth_checks(SOURCE + SOURCE, limit=1)) == 1
    assert len(proportion_growth_checks(SOURCE + SOURCE)) == 2


def test_arithmetic_compatibility_does_not_hide_opposite_direction_word():
    text = "比例由40%提升至30%，环比下降25%。"
    check, = proportion_growth_checks(text)
    assert check["rounding_compatible"] is True
    assert check["endpoint_direction"] == "提升至"
    assert check["endpoint_direction_interval_compatible"] is False
    span = check["operand_sources"]["direction"]
    assert text[span["start"]:span["end"]] == "提升至"


@pytest.mark.parametrize("kwargs", [{"offset": True}, {"offset": -1}, {"limit": True}, {"limit": -1}])
def test_bad_offsets_and_limits_fail(kwargs):
    with pytest.raises(ValueError):
        proportion_growth_checks(SOURCE, **kwargs)
