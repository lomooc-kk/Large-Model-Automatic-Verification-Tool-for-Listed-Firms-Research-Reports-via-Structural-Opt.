"""Bind a single month/day range allegation before inheriting its proof.

A quote containing an invalid month does not prove that an unrelated fiscal
claim is wrong. Unknown and mixed reasons keep their own review candidates.
This helper changes neither source detection nor error types.
"""
from __future__ import annotations

import re

from .calendar_alignment import _source_span


DETECTORS = frozenset({"verified.month_range", "verified.day_range"})
_MONTH = re.compile(r"(?<!\d)(?P<value>1[3-9]|[2-9]\d)\s*月")
_DAY = re.compile(r"月\s*(?P<value>3[2-9]|[4-9]\d)\s*日")
_TEMPORAL = re.compile(
    r"(?<!\d)\d{4}\s*年?\s*[-—–~～至到]\s*\d{4}(?:\s*年)?"
    r"|(?<!\d)(?:\d{4}\s*年\s*)?\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
    r"|(?<!\d)\d{4}\s*年")
_QUOTES = str.maketrans("", "", "\"'“”‘’「」『』")


def _single_reason(reason: str, value: int, *, month: bool, source_month: int | None = None) -> bool:
    compact = re.sub(r"\s+", "", reason.translate(_QUOTES))
    if not compact or len(compact) > 240:
        return False
    token = rf"0*{value}{'月' if month else '日'}"
    noun = r"月份|月数" if month else r"日期|日数"
    limit = 12 if month else 31
    subject = rf"(?:{token}|(?:该|上述|原文)?(?:{noun}))"
    over = rf"(?:超出|超过|大于)(?:合法|有效)?(?:的)?(?:1[-—至]{limit}(?:范围)?|{limit}{'月' if month else '日'}|{limit})(?:的)?(?:范围|上限)?"
    invalid = r"(?:不是(?:合法|有效)(?:的)?(?:公历)?(?:月份|日期)|(?:为|是)?(?:非法|无效)(?:的)?(?:月份|日期)?|不存在|不合法|(?:本身)?不成立)"
    accused = re.compile(rf"{subject}(?:{over}|{invalid})")
    fact = (re.compile(r"(?:一年|公历一年|每年)(?:仅有|只有|有|最多有)12(?:个)?月") if month
            else re.compile(r"(?:一个月|每月|月份)(?:最多有|最多|不超过)31(?:天|日)"))
    seen = False
    for clause in filter(None, re.split(r"[，,；;。.!！?？]+", compact)):
        clause = re.sub(r"^(?:原文中|原文|因此|所以|故)", "", clause)
        source_month_denial = (not month and source_month is not None
                               and re.fullmatch(rf"0*{source_month}月(?:没有|不存在){token}", clause))
        if accused.fullmatch(clause) or source_month_denial or re.fullmatch(rf"(?:公历中)?(?:不存在|没有){token}", clause):
            seen = True
        elif fact.fullmatch(clause):
            continue
        elif seen and clause in {"属于时间信息非法", "属于非法时间信息", "不是有效公历日期", "该年月日不是有效公历日期"}:
            continue
        else:
            return False
    return seen


def date_component_allegation_matches(candidate: dict, finding: dict, content: str) -> bool:
    """Accept only exact source proof and a solely matching month/day reason.

    Both promotion and deduplication must use this guard, including exact-span
    shortcuts. It deliberately abstains when two invalid tokens share a quote.
    """
    if not isinstance(candidate, dict) or not isinstance(finding, dict) or not isinstance(content, str):
        return False
    detector = finding.get("detector_id")
    if (detector not in DETECTORS or finding.get("status") != "confirmed_error"
            or finding.get("error_type") != "时间信息非法" or candidate.get("error_type") != "时间信息非法"):
        return False
    groups = [finding.get("verification_spans"), finding.get("spans"), candidate.get("spans")]
    if not all(isinstance(x, list) and len(x) == 1 for x in groups):
        return False
    validated = [_source_span(x[0], content) for x in groups]
    if any(x is None for x in validated):
        return False
    proof, display, quote = validated
    ps, pe, text = proof
    ds, de, _ = display
    cs, ce, _ = quote
    if not ds <= cs <= ps < pe <= ce <= de:
        return False
    month = detector == "verified.month_range"
    pattern = _MONTH if month else _DAY
    parsed = pattern.fullmatch(text)
    if parsed is None:
        return False
    # Match on the full source, not on a slice whose edge could turn a suffix
    # of a longer number into an apparent invalid month.
    matches = [m for m in pattern.finditer(content) if cs <= m.start() < m.end() <= ce]
    if len(matches) != 1 or (matches[0].start(), matches[0].end()) != (ps, pe):
        return False
    evidence = finding.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(e, dict) for e in evidence):
        return False
    source = [e for e in evidence if e.get("kind") == "source_text"]
    check = "month_range" if month else "day_range"
    checks = [e for e in evidence if e.get("check") == check]
    if (len(source) != 1 or _source_span(source[0], content) != proof or len(checks) != 1
            or checks[0].get("kind") != "deterministic" or checks[0].get("text") != text):
        return False
    reason = candidate.get("reason")
    if not isinstance(reason, str):
        return False
    value = int(parsed["value"])
    other_time = any(not (m.start() < pe and ps < m.end())
                     for m in _TEMPORAL.finditer(content) if cs <= m.start() < m.end() <= ce)
    explicit = re.search(rf"(?<!\d)0*{value}\s*{'月' if month else '日'}", reason)
    # Promotion rewrites a proved single allegation to the rule's standard
    # wording. Only the exact inherited proof can disambiguate that internal
    # wording; raw model metadata is not copied into these internal fields.
    inherited = (candidate.get("status") == "confirmed_error"
                 and candidate.get("verified_by") == detector
                 and candidate.get("verification_spans") == finding["verification_spans"]
                 and candidate.get("evidence") == evidence
                 and reason == finding.get("reason"))
    if other_time and not explicit and not inherited:
        return False
    source_month = None
    if not month:
        prefixes = [m for m in re.finditer(r"(?<!\d)(\d{1,2})\s*(?=月)", content) if m.end() == ps]
        if len(prefixes) == 1 and 1 <= int(prefixes[0][1]) <= 12:
            source_month = int(prefixes[0][1])
    return _single_reason(reason, value, month=month, source_month=source_month)
