"""Conservative source binding for one impossible-calendar-date allegation.

This helper does not detect new errors or change candidates.  A false result
means that the model allegation must keep its own review status; it does not
invalidate an independently verified calendar finding.  The small positive
reason grammar deliberately abstains on unfamiliar or compound explanations.
"""
from __future__ import annotations

import calendar
from datetime import date
import re


_DATE = re.compile(r"(?<!\d)(?P<y>\d{4})年\s*(?P<m>\d{1,2})月\s*(?P<d>\d{1,2})日")
_MONTH_DAY = re.compile(r"(?<!\d)\d{1,2}月\s*\d{1,2}日")
_NUMERIC_DATE = re.compile(r"(?<!\d)\d{4}[-./]\d{1,2}[-./]\d{1,2}(?!\d)")
_PUNCTUATION = re.compile(r"[，,；;。.!！?？]+")
_QUOTES = str.maketrans("", "", "\"'“”‘’「」『』")


def _source_span(span: object, content: str) -> tuple[int, int, str] | None:
    if not isinstance(span, dict):
        return None
    start, end, text = span.get("start"), span.get("end"), span.get("text")
    if (type(start) is not int or type(end) is not int or not isinstance(text, str)
            or not 0 <= start < end <= len(content) or content[start:end] != text):
        return None
    return start, end, text


def _date_aliases(year: int, month: int, day: int) -> str:
    # Source positions, rather than spelling alone, identify the proof. These
    # aliases allow an explanation to omit the year or use zero padding.
    y = rf"0*{year}年" if year else r"0{1,4}年"
    m, d = rf"0*{month}月", rf"0*{day}日"
    return rf"(?:{y})?{m}{d}"


def _single_calendar_reason(reason: str, year: int, month: int, day: int) -> bool:
    compact = re.sub(r"\s+", "", reason.translate(_QUOTES))
    if not compact or len(compact) > 320:
        return False
    target = _date_aliases(year, month, day)
    subject = rf"(?:{target}|该日期|该年月日|上述日期|原文日期|日期)"
    invalid = (
        r"(?:不是(?:有效|合法)(?:的)?(?:公历)?日期"
        r"|(?:是|为)?(?:无效|非法)(?:的)?(?:公历)?日期"
        r"|(?:是|为)?不存在的日期|(?:在公历中)?不存在"
        r"|不符合公历(?:规则|日期规则)|(?:不合法|无效|非法))"
    )
    accusation = re.compile(rf"(?:原文(?:中)?(?:的|标注的|写作的)?)?{subject}{invalid}")
    reverse_accusation = re.compile(rf"(?:公历中)?(?:不存在|没有){target}(?:这一日期|这个日期)?")

    month_days = None
    if 1 <= year <= 9999 and 1 <= month <= 12:
        month_days = calendar.monthrange(year, month)[1]
    year_alias = rf"0*{year}年"
    facts = []
    if month_days is not None:
        facts.append(re.compile(
            rf"(?:{year_alias})?0*{month}月(?:仅有|只有|最多有|共有|有|最多){month_days}(?:天|日)"))
        if day > month_days or day < 1:
            # These are source-bound assertions about this particular day, not
            # arbitrary statements about a second date in the same sentence.
            facts.append(re.compile(rf"(?:{year_alias})?0*{month}月(?:没有|不存在)0*{day}日"))
        if month == 2:
            facts.append(re.compile(
                rf"{year_alias}(?:是|为){'闰年' if calendar.isleap(year) else '平年'}"))
            if not calendar.isleap(year):
                facts.append(re.compile(rf"{year_alias}不是闰年"))

    has_invalid_assertion = False
    clauses = [part for part in _PUNCTUATION.split(compact) if part]
    for clause in clauses:
        clause = re.sub(r"^(?:因此|所以|故|因|由于)", "", clause)
        if accusation.fullmatch(clause) or reverse_accusation.fullmatch(clause):
            has_invalid_assertion = True
        elif has_invalid_assertion and re.fullmatch(invalid, clause):
            # A subjectless restatement can only continue an already bound
            # invalid-date allegation, never introduce or switch its subject.
            pass
        elif any(pattern.fullmatch(clause) for pattern in facts):
            # A correctly stated month's length alone is not an allegation.
            if re.search(r"没有|不存在", clause):
                has_invalid_assertion = True
        else:
            # No negative-word blacklist: every remainder must be accounted for
            # by the bounded calendar-only grammar. Fiscal, order, corrections,
            # other dates, negations and mixed claims consequently abstain.
            return False
    return has_invalid_assertion


def calendar_allegation_matches(candidate: dict, finding: dict, content: str) -> bool:
    """Whether a candidate states only the finding's source-proved invalid date.

    Call this before *both* deterministic promotion and calendar deduplication.
    Only exact source spans and a matching ``calendar_date`` proof are accepted.
    A wider quote may contain a fiscal interval, but its reason must describe
    solely the one invalid calendar date. The function never mutates inputs.
    """
    if not isinstance(candidate, dict) or not isinstance(finding, dict) or not isinstance(content, str):
        return False
    if (finding.get("detector_id") != "verified.calendar"
            or finding.get("status") != "confirmed_error"
            or finding.get("error_type") != "时间信息非法"
            or candidate.get("error_type") != "时间信息非法"):
        return False
    proof_spans = finding.get("verification_spans")
    candidate_spans = candidate.get("spans")
    display_spans = finding.get("spans")
    if not all(isinstance(spans, list) and len(spans) == 1
               for spans in (proof_spans, candidate_spans, display_spans)):
        return False
    proof = _source_span(proof_spans[0], content)
    quoted = _source_span(candidate_spans[0], content)
    display = _source_span(display_spans[0], content)
    if proof is None or quoted is None or display is None:
        return False
    ps, pe, text = proof
    cs, ce, _ = quoted
    ds, de, _ = display
    if not ds <= cs <= ps < pe <= ce <= de:
        return False
    parsed = _DATE.fullmatch(text)
    if parsed is None:
        return False
    year, month, day = (int(parsed[key]) for key in ("y", "m", "d"))
    try:
        date(year, month, day)
    except ValueError:
        pass
    else:
        return False
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return False
    source_proofs = [item for item in evidence
                     if isinstance(item, dict) and item.get("kind") == "source_text"]
    if len(source_proofs) != 1 or _source_span(source_proofs[0], content) != proof:
        return False
    calendar_proofs = [item for item in evidence
                       if isinstance(item, dict) and item.get("check") == "calendar_date"]
    if len(calendar_proofs) != 1:
        return False
    fields = calendar_proofs[0]
    if (fields.get("kind") != "deterministic"
            or any(type(fields.get(key)) is not int or fields[key] != value
                   for key, value in (("year", year), ("month", month), ("day", day)))):
        return False
    dates = list(_DATE.finditer(content, cs, ce))
    if not any((match.start(), match.end()) == (ps, pe) for match in dates):
        return False
    has_other_dates = False
    for match in dates:
        if (match.start(), match.end()) == (ps, pe):
            continue
        # Another legal complete date can be event context. A second invalid
        # date is another possible proof, even when its text is identical.
        try:
            date(*(int(match[key]) for key in ("y", "m", "d")))
        except ValueError:
            return False
        has_other_dates = True
    if any(not any(full.start() <= match.start() < match.end() <= full.end() for full in dates)
           for match in _MONTH_DAY.finditer(content, cs, ce)):
        return False
    if _NUMERIC_DATE.search(content, cs, ce):
        return False
    reason = candidate.get("reason")
    if not isinstance(reason, str):
        return False
    if has_other_dates and not any(
            tuple(int(match[key]) for key in ("y", "m", "d")) == (year, month, day)
            for match in _DATE.finditer(reason)):
        # Anaphora or just a month/day is insufficient among multiple dates.
        # The bounded grammar below still rejects other or mixed allegations.
        return False
    return _single_calendar_reason(reason, year, month, day)
