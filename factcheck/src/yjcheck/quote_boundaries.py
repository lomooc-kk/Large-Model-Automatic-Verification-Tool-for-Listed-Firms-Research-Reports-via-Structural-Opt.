"""Source-only observations about contextual quotes cutting numeric tokens.

Exact anchoring and completeness of a quotation are different properties.  This
module accepts already anchored spans and reports a narrow lexical boundary
observation.  It does not expand spans, reject a candidate, or establish that its
business allegation is false.  A deliberately local number/unit selection is
allowed, including selecting only the erroneous digits of a number.

Coverage is intentionally limited: Arabic/fullwidth digits, conventional decimal
and thousands separators, scientific notation, and a finite set of explicit
financial/physical units.  No cross-line unit inference or general Chinese word
segmentation is attempted.  No observation is not a completeness certificate.
"""
from __future__ import annotations

import re

_D = r"[0-9０-９]"
_NUMBER = re.compile(
    rf"(?:{_D}{{1,3}}(?:[,，]{_D}{{3}})+|{_D}+)(?:[.．]{_D}+)?"
    rf"(?:[eE][+\-＋－−]?{_D}+)?"
)
_SIGNS = "+-＋－−"
_CURRENCY_SIGNS = "$¥￥"
_SYMBOL_UNITS = {"%", "％", "‰", "‱"}
_UNITS = (
    "个百分点", "基点", "亿元", "万元", "千元", "元",
    "亿美元", "万美元", "美元", "亿港元", "万港元", "港元",
    "亿欧元", "万欧元", "欧元", "亿股", "万股", "股",
    "万吨", "吨", "万千瓦时", "千瓦时", "万千瓦", "千瓦",
    "千克", "公斤", "公里", "平方米", "倍", "bps", "bp", "pct",
    "%", "％", "‰", "‱",
)
_UNIT = re.compile("|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)))
_PER_UNIT = re.compile(r"[/／](?:平方米|千瓦时|千克|公斤|股|吨|年|月|kWh)")
_HSPACE = " \t\u00a0\u202f"
_TERMINATORS = " \t\r\n\u00a0\u202f，,。.;；:：、|｜!?！？()（）[]【】{}“”‘’\"'=＝<>≤≥"
# A unit followed by an arbitrary Chinese word is ambiguous (元宵, 股东, 倍增).
# Only these explicit continuations or punctuation justify treating it as a unit.
_UNIT_CONTINUATIONS = ("同比", "环比", "较", "增长", "下降", "减少", "增加", "左右", "以上", "以下", "以内", "以外", "至", "到", "的")
_LOCAL_SEPARATORS = re.compile(r"[\s+\-＋－−$¥￥.,，．%％‰‱/／=＝×÷*:：()（）\[\]【】~～—–、;；|｜]+")


def _local_selection(text: str) -> bool:
    """A number/unit-only selection is legitimate even within a larger token."""
    remainder = _NUMBER.sub("", text)
    remainder = _UNIT.sub("", remainder)
    remainder = _PER_UNIT.sub("", remainder)
    return not _LOCAL_SEPARATORS.sub("", remainder)


def _unit_end(content: str, number_end: int) -> int:
    cursor = number_end
    while cursor < len(content) and content[cursor] in _HSPACE and cursor - number_end < 3:
        cursor += 1
    match = _UNIT.match(content, cursor)
    if not match:
        return number_end
    end = match.end()
    per = _PER_UNIT.match(content, end)
    if per:
        end = per.end()
    if match.group() in _SYMBOL_UNITS and not per:
        return end
    if end == len(content) or content[end] in _TERMINATORS:
        return end
    if content.startswith(_UNIT_CONTINUATIONS, end):
        return end
    return number_end


def _number_start(content: str, digit_start: int) -> int:
    start = digit_start
    if start and content[start - 1] in _SIGNS:
        before_sign = content[start - 2] if start > 1 else ""
        # Do not turn range separators or alphanumeric identifiers into signs.
        if not before_sign or not (before_sign.isascii() and before_sign.isalnum()
                                   or before_sign in "０１２３４５６７８９)）]】%％‰‱"):
            start -= 1
    if start and content[start - 1] in _CURRENCY_SIGNS:
        start -= 1
    return start


def _source_span(content: str, start: int, end: int) -> dict:
    return {"start": start, "end": end, "text": content[start:end]}


def numeric_quote_boundary_observations(content: str, spans: list[dict]) -> list[dict]:
    """Return advisory source evidence for cuts in contextual quotations.

    Input offsets use Python Unicode ``[start,end)``. Every input must already
    match the source exactly; invalid inputs raise ValueError for the caller to
    handle separately from model quality. Output order is input span, then left
    boundary, then right boundary. All returned source text/offsets are exact.

    ``contextual_numeric_lexeme_cut`` means a contextual quote omits part of a
    written number/sign; ``contextual_quantity_unit_cut`` means it splits a
    recognized number-and-unit expression. Both are observations, not rejection
    reasons. Pure numeric/unit selections produce no observations by design.
    """
    if not isinstance(content, str) or not isinstance(spans, (list, tuple)):
        raise ValueError("numeric boundary check requires source text and anchored spans")
    for span in spans:
        if (not isinstance(span, dict) or type(span.get("start")) is not int
                or type(span.get("end")) is not int or not isinstance(span.get("text"), str)
                or not 0 <= span["start"] < span["end"] <= len(content)
                or content[span["start"]:span["end"]] != span["text"]):
            raise ValueError("numeric boundary check requires exact anchored source spans")
    if not spans:
        return []
    tokens = []
    for match in _NUMBER.finditer(content):
        # Version strings/identifiers need their own tokenizer. Do not partially
        # interpret one as a financial number merely because it contains digits.
        if match.start() and (content[match.start() - 1] in ".．_" or content[match.start() - 1].isdecimal()
                              or content[match.start() - 1].isascii() and content[match.start() - 1].isalpha()):
            continue
        if match.end() < len(content) and (content[match.end()].isdecimal()
                or content[match.end()] in ".．" and match.end() + 1 < len(content)
                and content[match.end() + 1].isdecimal()):
            continue
        start = _number_start(content, match.start())
        tokens.append((start, match.end(), _unit_end(content, match.end())))
    observations = []
    for index, span in enumerate(spans):
        if _local_selection(span["text"]):
            continue
        for side, boundary in (("left", span["start"]), ("right", span["end"])):
            for start, number_end, token_end in tokens:
                if not start < boundary < token_end:
                    continue
                omitted_start, omitted_end = ((start, boundary) if side == "left" else (boundary, token_end))
                observations.append({
                    "span_index": index, "boundary": side, "boundary_offset": boundary,
                    "code": "contextual_numeric_lexeme_cut" if boundary < number_end else "contextual_quantity_unit_cut",
                    "quoted_span": _source_span(content, span["start"], span["end"]),
                    "numeric_span": _source_span(content, start, number_end),
                    "token_span": _source_span(content, start, token_end),
                    "omitted_span": _source_span(content, omitted_start, omitted_end),
                    "is_anchor_failure": False, "business_error_proven": False,
                })
    return observations
