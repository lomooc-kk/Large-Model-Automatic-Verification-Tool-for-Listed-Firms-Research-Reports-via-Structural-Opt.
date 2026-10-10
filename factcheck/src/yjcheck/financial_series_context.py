"""Source-only financial series intermediates, with conditional semantics.

Printed order does not prove consecutive periods. A displayed current value
and growth rate do not prove a positive previous base. No helper adjudicates a
candidate, reads a benchmark, or substitutes an unreported financial value.
"""
from decimal import Decimal, localcontext
import re


_NUMBER = r"[+＋\-−－]?\d{1,12}(?:\.\d{1,8})?"
_PERCENT = rf"{_NUMBER}\s*[%％]"
_GROWTH_SERIES = re.compile(
    rf"(?P<metric>扣非后归母净利润|扣非归母净利润|销售收入|营业收入|收入|营收|归母净利润|扣非净利润|净利润)"
    rf"(?:分别同比|分别较上年同期)(?P<direction>增长|增加|增|下降|减少|降)?"
    rf"(?P<rates>{_PERCENT}(?:\s*[、,，/／]\s*{_PERCENT}){{1,11}})"
    rf"(?=[。；;\n]|$)")
_CLAIM = re.compile(r"(?P<operator>累计|连续)(?P<count>\d{1,2}|[一二三四五六七八九十两]{1,3})个季度同比(?P<direction>下降|增长)")
_PROFIT_SERIES = re.compile(
    rf"(?P<metric>扣非后归母净利润|扣非归母净利润|归母净利润|扣非净利润|净利润)"
    rf"分别(?:为|是)\s*(?P<amounts>{_NUMBER}(?:\s*[/／]\s*{_NUMBER}){{1,7}})"
    rf"\s*(?P<unit>亿元|万元|元)\s*[，,]\s*同比\s*"
    rf"(?P<rates>{_PERCENT}(?:\s*[/／]\s*{_PERCENT}){{1,7}})"
    rf"(?=[，,。；;\n]|$)")
_SIGN_TRANSLATION = str.maketrans({"＋": "+", "−": "-", "－": "-"})


def _number(raw):
    return Decimal(re.sub(r"\s+", "", raw).translate(_SIGN_TRANSLATION))


def _span(text, start, end, offset):
    return {"start": offset + start, "end": offset + end, "text": text[start:end]}


def _count(raw):
    if raw.isdigit():
        return int(raw)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if raw in digits:
        return digits[raw]
    if raw == "十":
        return 10
    if re.fullmatch(r"十[一二三四五六七八九]", raw):
        return 10 + digits[raw[-1]]
    return None


def _series_members(text, group_start, raw, offset, *, percent):
    pattern = _PERCENT if percent else _NUMBER
    return [{"displayed": match.group(),
             "value": str(_number(match.group().rstrip("%％").strip())),
             "source": _span(text, group_start + match.start(), group_start + match.end(), offset)}
            for match in re.finditer(pattern, raw)]


def financial_series_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    if not isinstance(text, str) or type(offset) is not int or offset < 0 or type(limit) is not int or limit < 0:
        raise ValueError("text, nonnegative integer offset and limit required")
    checks = []
    for match in _GROWTH_SERIES.finditer(text):
        members = _series_members(text, match.start("rates"), match["rates"], offset, percent=True)
        numbers = [_number(member["value"]) for member in members]
        if match["direction"] in {"下降", "减少", "降"}:
            # A direction word plus negative printed magnitude has competing
            # readings; leave it for review rather than applying two minuses.
            if any(value < 0 for value in numbers):
                continue
            numbers = [-value for value in numbers]
        paragraph_start = text.rfind("\n", 0, match.start()) + 1
        paragraph_end = text.find("\n", match.end())
        if paragraph_end < 0:
            paragraph_end = len(text)
        if paragraph_end - paragraph_start > 1200:
            continue
        claims = list(_CLAIM.finditer(text, paragraph_start, paragraph_end))
        if len(claims) != 1 or _count(claims[0]["count"]) is None:
            continue
        # Multiple possible series in one paragraph do not establish which
        # sequence the cumulative claim summarizes.
        if len(list(_GROWTH_SERIES.finditer(text, paragraph_start, paragraph_end))) != 1:
            continue
        claim = claims[0]
        sign = -1 if claim["direction"] == "下降" else 1
        run = longest = 0
        for value in numbers:
            run = run + 1 if value * sign > 0 else 0
            longest = max(longest, run)
        checks.append({
            "kind": "printed_growth_series_count", "scope": "calculation_only_not_business_verdict",
            "source": _span(text, match.start(), match.end(), offset), "metric": match["metric"],
            "members": members, "signed_growth_percent": [str(value) for value in numbers],
            "negative_count": sum(value < 0 for value in numbers), "positive_count": sum(value > 0 for value in numbers),
            "zero_count": sum(value == 0 for value in numbers),
            "claim": {"source": _span(text, claim.start(), claim.end(), offset),
                      "operator": claim["operator"], "count": _count(claim["count"]), "direction": claim["direction"]},
            "matching_sign_count": sum(value * sign > 0 for value in numbers),
            "longest_matching_run_in_printed_order": longest,
            "claim_to_series_binding_proven": False, "period_contiguity_proven": False,
            "interpretation": "累计 counts matching observations; it does not require every listed observation or a consecutive run. 连续 requires period-order evidence. These counts do not validate incomplete quarter labels or prove scope alignment.",
        })
    for match in _PROFIT_SERIES.finditer(text):
        # A comma followed by another number or an empty percent slot is not
        # evidence that the slash series ended. Do not silently truncate it.
        if re.match(r"[，,]\s*(?:[+＋\-−－]?\d|[%％])", text[match.end():]):
            continue
        amounts = _series_members(text, match.start("amounts"), match["amounts"], offset, percent=False)
        rates = _series_members(text, match.start("rates"), match["rates"], offset, percent=True)
        if len(amounts) != len(rates):
            continue
        pairs = []
        with localcontext() as context:
            context.prec = 40
            for amount, rate in zip(amounts, rates):
                current, growth = _number(amount["value"]), _number(rate["value"]) / 100
                signed = current / (1 + growth) if growth != -1 else None
                negative = current / (1 - growth) if growth != 1 else None
                positive_base = signed if signed is not None and signed > 0 else None
                negative_base = negative if negative is not None and negative < 0 else None
                positive_underdetermined = current == 0 and growth == -1
                negative_underdetermined = current == 0 and growth == 1
                pairs.append({"current": amount, "growth_percent": rate,
                              "positive_previous_base_point_if_absolute_formula": str(positive_base) if positive_base is not None else None,
                              "negative_previous_base_point_if_absolute_formula": str(negative_base) if negative_base is not None else None,
                              "positive_base_underdetermined_at_displayed_point": positive_underdetermined,
                              "negative_base_underdetermined_at_displayed_point": negative_underdetermined,
                              "two_base_signs_possible_at_displayed_points": positive_base is not None and negative_base is not None})
        if not any(pair["negative_previous_base_point_if_absolute_formula"] is not None
                   or pair["negative_base_underdetermined_at_displayed_point"]
                   or pair["positive_base_underdetermined_at_displayed_point"] for pair in pairs):
            continue
        checks.append({
            "kind": "conditional_profit_growth_base_sign", "scope": "calculation_only_not_business_verdict",
            "source": _span(text, match.start(), match.end(), offset), "metric": match["metric"], "unit": match["unit"],
            "conditional_formula": "growth=(current-previous)/abs(previous)",
            "positive_base_formula": "previous=current/(1+growth)",
            "negative_base_formula": "previous=current/(1-growth)", "pairs": pairs,
            "formula_used_by_source_proven": False, "unique_previous_base_proven": False,
            "display_rounding_analyzed": False, "null_point_does_not_exclude_rounded_solution": True,
            "interpretation": "These are conditional point solutions, not recovered actual financial data. A current positive profit with growth above 100% can have a positive or negative previous base under the absolute-base convention. At current=0, growth=+100% permits any negative base; growth=-100% permits any positive base. Null means no unique admissible point was computed, not that a rounded source value rules out that sign. Do not force every quarter to a positive base or declare an annual-sum contradiction without establishing growth convention, base signs, rounding and period alignment.",
        })
    return sorted(checks, key=lambda check: (check["source"]["start"], check["kind"]))[:limit]
