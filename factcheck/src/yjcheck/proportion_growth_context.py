"""Conditional growth intervals for an explicitly repeated percentage metric.

Displayed percentages can be rounded. A change in percentage points and a
relative percent change are different quantities. Source-only intermediates
here neither identify a missing value nor adjudicate a business statement.
"""
from decimal import Decimal, localcontext
import re

from .arithmetic_context import _interval, _number


_N = r"\d{1,3}(?:\.\d{1,8})?"
_CHANGE = re.compile(
    rf"(?P<metric>贡献比例|占比|比例|比重|渗透率|覆盖率|利用率|毛利率|净利率)"
    rf"\s*(?:由|从)\s*(?:(?P<base_period>上年|去年|上期|上月|上季度|上周)(?:的)?)?\s*"
    rf"(?P<base>{_N})\s*[%％]\s*(?P<direction>提升至|提高至|上升至|增长至|增至|下降至|降低至|降至)\s*"
    rf"(?P<current>{_N})\s*[%％]\s*[，,]\s*"
    rf"(?P<period>同比|环比)\s*(?P<growth_direction>增长|增加|提高|上升|下降|减少|降低)\s*"
    rf"(?P<growth>[+＋\-−－]?{_N})\s*[%％](?![\d.%％])")
_UNCERTAIN = re.compile(r"预计|预期|预测|目标|约|近|超|逾|不足|至少|至多|精确|未舍入|未四舍五入|截尾|截断|假设|示例|例如")
_TAIL = re.compile(r"\s*(?:[-—~～至到/／±∓]|[、,，]\s*(?:约|近|超过|不足)?\s*[+＋\-−－]?\d|(?i:to)\b|个百分点|百分比点|(?i:pct)|(?:及|或|和)\s*\d)")
_DECLINE = {"下降", "减少", "降低"}


def proportion_growth_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    """Read complete bounded constructions, retaining unproved basis assumptions."""
    if not isinstance(text, str) or type(offset) is not int or offset < 0 or type(limit) is not int or limit < 0:
        raise ValueError("text, nonnegative integer offset and limit required")
    result = []
    for match in _CHANGE.finditer(text):
        if len(result) >= limit:
            break
        if "\n" in match.group() or "\r" in match.group() or _TAIL.match(text, match.end()):
            continue
        # Qualifiers in the same local clause remain relevant even when the
        # narrow metric matcher starts after them.
        prefix_start = max(text.rfind(mark, 0, match.start()) for mark in "。；;\n，,") + 1
        prior_qualifier = re.search(
            r"(?:预计|预期|预测|目标|假设|示例|例如)[，,：:][^。；;\n，,:：]{0,40}$", text[:match.start()])
        if prior_qualifier or _UNCERTAIN.search(text[prefix_start:match.end()]):
            continue
        base_period = match["base_period"]
        if ((base_period in {"上年", "去年"} and match["period"] != "同比")
                or (base_period in {"上月", "上季度", "上周", "上期"} and match["period"] != "环比")):
            continue
        with localcontext() as context:
            context.prec = 40
            base, current, growth = [_number(match[k]) for k in ("base", "current", "growth")]
            if not (0 < base <= 100 and 0 <= current <= 100):
                continue
            if match["growth_direction"] in _DECLINE and growth < 0:
                # "下降-5%" admits competing sign conventions.
                continue
            blo, bhi = _interval(match["base"])
            clo, chi = _interval(match["current"])
            rlo, rhi = _interval(match["growth"])
            if blo <= 0 or clo < 0 or chi > 100 or bhi > 100:
                continue
            if match["growth_direction"] in _DECLINE:
                rlo, rhi = -rhi, -rlo
            low, high = (clo / bhi - 1) * 100, (chi / blo - 1) * 100
            overlap_low, overlap_high = max(low, rlo), min(high, rhi)

            def source(start, end):
                return {"start": offset + start, "end": offset + end, "text": text[start:end]}

            result.append({
                "kind": "conditional_displayed_proportion_growth",
                "scope": "calculation_only_not_business_verdict",
                "source": source(match.start(), match.end()),
                "operand_sources": {key: source(*match.span(key)) for key in ("metric", "base", "current", "growth", "period", "direction")},
                "metric": match["metric"], "base_period": base_period, "comparison_period": match["period"],
                "endpoint_direction": match["direction"],
                "endpoint_direction_interval_compatible": (clo < bhi if match["direction"] in {"下降至", "降低至", "降至"} else chi > blo),
                "base_percent_raw": match["base"], "current_percent_raw": match["current"],
                "reported_growth_raw": match["growth"], "reported_growth_direction": match["growth_direction"],
                "formula": "relative_growth_percent=(current_percent/base_percent-1)*100",
                "point_growth_percent": str((current / base - 1) * 100),
                "point_percentage_point_change": str(current - base),
                "base_rounding_interval": [str(blo), str(bhi)],
                "current_rounding_interval": [str(clo), str(chi)],
                "possible_growth_percent": [str(low), str(high)],
                "reported_growth_interval": [str(rlo), str(rhi)],
                "rounding_compatible": overlap_low <= overlap_high,
                "boundary_only_overlap": overlap_low == overlap_high,
                "relation_proven": False,
                "assumptions": ["round_to_last_displayed_digit", "relative_change_of_same_proportion_metric"],
                "required_business_checks": ["same_measurement_basis", "period_and_denominator_alignment", "reported_growth_refers_to_proportion_not_underlying_count"],
                "interpretation": "An overlap means point arithmetic cannot establish a contradiction. It does not validate the printed endpoint direction. Neither overlap nor non-overlap proves business correctness. Percentage point change is not relative growth. Keep other allegations independent.",
            })
    return result
