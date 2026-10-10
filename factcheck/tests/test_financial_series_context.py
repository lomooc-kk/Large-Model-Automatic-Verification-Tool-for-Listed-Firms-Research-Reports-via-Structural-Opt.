"""Known reasoning traps expressed as synthetic source relations, without gold."""
from decimal import Decimal
import json

import pytest

from yjcheck.financial_series_context import financial_series_checks
from yjcheck.text_context import TextSlice
from yjcheck.text_review import _messages


def checks(text):
    return financial_series_checks(text)


def test_cumulative_five_is_not_all_six_or_five_consecutive():
    text = "销售收入累计五个季度同比下降。收入分别同比增-14%、5%、-7%、-6%、-18%、-8%。"
    result, = checks(text)
    assert result["negative_count"] == result["matching_sign_count"] == result["claim"]["count"] == 5
    assert result["positive_count"] == 1
    assert result["longest_matching_run_in_printed_order"] == 4
    assert result["claim"]["operator"] == "累计"
    assert result["claim_to_series_binding_proven"] is False
    assert result["period_contiguity_proven"] is False
    assert "error_type" not in result and "verdict" not in result


def test_consecutive_word_and_zero_are_preserved_not_reinterpreted_as_cumulative():
    result, = checks("营收连续三个季度同比增长。营收分别同比增长1%、0%、2%、3%。")
    assert result["positive_count"] == 3 and result["zero_count"] == 1
    assert result["longest_matching_run_in_printed_order"] == 2
    assert result["claim"]["operator"] == "连续"
    assert not result["period_contiguity_proven"]


def test_explicit_decline_magnitudes_translate_to_signed_changes():
    result, = checks("营收累计两个季度同比下降。营收分别同比下降3%、4%。")
    assert result["signed_growth_percent"] == ["-3", "-4"]
    assert [member["displayed"] for member in result["members"]] == ["3%", "4%"]


@pytest.mark.parametrize("text", [
    "营收累计两个季度同比下降。营收分别同比下降-3%、4%。",
    "营收累计两个季度同比下降。营收分别同比增-3%、4%、%。",
    "营收累计两个季度同比下降。营收分别同比增-3%、4%、约-5%。",
    "营收累计两个季度同比下降。营收分别同比增-3%、4%、不低于-5%。",
    "营收累计两个季度同比下降。\n营收分别同比增-3%、4%。",
    "营收累计两个季度同比下降。营收分别同比增-3%、4%。收入分别同比增-5%、2%。",
    "营收累计两个季度同比下降，累计三个季度同比增长。营收分别同比增-3%、4%。",
    "营收累计两个季度同比下降。营收分别同比增3、4。",
    "营收分别同比增-3%、4%。",
])
def test_incomplete_ambiguous_or_unbound_series_abstains(text):
    assert checks(text) == []


def test_conditional_growth_can_have_two_opposite_sign_bases():
    text = "归母净利润分别为2/3亿元，同比+150%/+200%。"
    result, = checks(text)
    assert result["formula_used_by_source_proven"] is False
    assert result["unique_previous_base_proven"] is False
    first = result["pairs"][0]
    assert first["two_base_signs_possible_at_displayed_points"] is True
    assert Decimal(first["positive_previous_base_point_if_absolute_formula"]) == Decimal("0.8")
    assert Decimal(first["negative_previous_base_point_if_absolute_formula"]) == Decimal("-4")
    # Independent reconstruction: both hypotheses yield the same stated growth.
    for key in ("positive_previous_base_point_if_absolute_formula", "negative_previous_base_point_if_absolute_formula"):
        prior = Decimal(first[key])
        assert (Decimal(2) - prior) / abs(prior) == Decimal("1.5")


@pytest.mark.parametrize("text", [
    "归母净利润分别为2/3亿元，同比+50%/+20%。",
    "归母净利润分别为2/3亿元，同比+150%/+200%/+300%。",
    "归母净利润分别为2/3亿元，同比+150%。",
    "企业家数分别为2/3家，同比+150%/+200%。",
    "归母净利润分别为约2/3亿元，同比+150%/+200%。",
    "归母净利润分别为2/3美元，同比+150%/+200%。",
    "归母净利润分别为2/3亿元，同比+150%/+200%，+300%。",
    "归母净利润分别为2/3亿元，同比+150%/+200%，%。",
    "归母净利润分别为2/3亿元，同比+150%/+200%，300。",
])
def test_growth_without_negative_base_hypothesis_or_complete_pairing_is_not_augmented(text):
    assert checks(text) == []


def test_singular_formulas_do_not_divide_by_zero():
    result, = checks("归母净利润分别为-1/2/0亿元，同比-100%/+100%/+100%。")
    assert result["pairs"][0]["positive_previous_base_point_if_absolute_formula"] is None
    assert result["pairs"][1]["negative_previous_base_point_if_absolute_formula"] is None
    assert result["pairs"][2]["negative_previous_base_point_if_absolute_formula"] is None
    assert result["pairs"][2]["negative_base_underdetermined_at_displayed_point"] is True
    assert result["display_rounding_analyzed"] is False
    assert result["null_point_does_not_exclude_rounded_solution"] is True


@pytest.mark.parametrize("growth,flag,prior", [
    ("+100", "negative_base_underdetermined_at_displayed_point", Decimal("-7")),
    ("-100", "positive_base_underdetermined_at_displayed_point", Decimal("7")),
])
def test_zero_current_singular_growth_allows_any_base_of_one_sign(growth, flag, prior):
    result, = checks(f"归母净利润分别为0/2亿元，同比{growth}%/+50%。")
    assert result["pairs"][0][flag] is True
    assert (0 - prior) / abs(prior) * 100 == Decimal(growth)


@pytest.mark.parametrize("metric", ["扣非归母净利润", "扣非后归母净利润"])
def test_full_profit_metric_is_not_silently_changed_to_its_suffix(metric):
    text = f"{metric}累计两个季度同比下降。{metric}分别同比增-3%、-4%。"
    result, = checks(text)
    assert result["metric"] == metric
    assert result["source"]["text"].startswith(metric)


def test_every_member_and_claim_preserves_exact_global_source_coordinates():
    source = "营收累计两个季度同比下降。营收分别同比增－3％、＋4％、−5％。归母净利润分别为2/3亿元，同比+150%/+200%。"
    whole = "前缀" + source
    output = financial_series_checks(source, offset=2)
    def validate(value):
        if isinstance(value, dict):
            if {"start", "end", "text"} <= value.keys():
                assert whole[value["start"]:value["end"]] == value["text"]
            for child in value.values():
                validate(child)
        elif isinstance(value, list):
            for child in value:
                validate(child)
    validate(output)
    assert len(output) == 2
    assert financial_series_checks(source, limit=0) == []
    assert len(financial_series_checks(source, limit=1)) == 1


def test_model_payload_receives_source_intermediates_without_answers():
    text = "归母净利润分别为2/3亿元，同比+150%/+200%。"
    payload = json.loads(_messages([TextSlice(0, len(text), text)], "synthetic", "", [])[-1]["content"])
    new = [row for row in payload["source_arithmetic_checks"] if row["kind"] == "conditional_profit_growth_base_sign"]
    assert len(new) == 1
    assert payload["contexts"][0]["text"] == text
    assert "errors" not in payload and "gold" not in payload


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"offset": True}, {"limit": -1}, {"limit": True}])
def test_invalid_offsets_and_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        financial_series_checks("源文", **kwargs)
