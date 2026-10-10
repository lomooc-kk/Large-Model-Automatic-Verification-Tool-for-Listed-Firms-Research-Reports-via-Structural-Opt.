"""Small source-grounded structures used by the original fifteen-type reviewer.

These helpers neither read benchmark labels nor infer the missing/correct value.
Offsets always address the unchanged source, and bounded parsers abstain on
unsupported lists rather than comparing a convenient numeric prefix.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
import re


_NUMBER = r"[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT = r"(?:亿元|万元|元/股|元|万股|股|万吨|吨|万件|件)"
_AMOUNT = rf"{_NUMBER}\s*{_UNIT}"
_RATE = rf"{_NUMBER}\s*[%％]"
_SEPARATOR = r"\s*[/／、]\s*"
_PAIRED_SERIES = re.compile(
    rf"(?<![\d.,/／、+\-])(?P<values>{_AMOUNT}(?:{_SEPARATOR}{_AMOUNT}){{1,11}})"
    rf"\s*[，,]\s*(?P<relation>(?:YOY|同比增速|同比增长率|同比增幅|同比变化率)"
    rf"\s*(?:依次|分别)(?:为|是)?)\s*"
    rf"(?P<rates>{_RATE}(?:{_SEPARATOR}{_RATE}){{1,11}})"
    rf"(?![\d.%％]|\s*[/／、])", re.I)
_FULL_STATEMENT = re.compile(r"[^。！？\r\n]+[。！？]?")
_ORDINAL_SERIES = re.compile(
    r"^\s*第(?:[一二三四五六七八九十百]+|\d+)"
    r"(?P<name>[\u4e00-\u9fff]{1,6}?)(?=[（(：:]|企业|公司|机构|项目|分别|为|是|有)")
_MONTH_RANGE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})年\s*"
    r"(?P<first>0?[1-9]|1[0-2])\s*[-—～~至]\s*(?P<last>0?[1-9]|1[0-2])月")
_DAY_RANGE = re.compile(
    r"(?<!\d)(?:(?P<year>(?:19|20)\d{2})年\s*)?"
    r"(?P<month>0?[1-9]|1[0-2])月\s*(?P<first>[0-3]?\d)日?"
    r"\s*[-—～~至]\s*(?P<last>[0-3]?\d)日")
_RANK_WITH_TOTAL = re.compile(
    r"(?<![\d.第前])(?P<total>[1-9]\d{0,5})个"
    r"(?P<scope>(?:一级(?:子)?|二级|三级)?行业)中\s*"
    r"(?P<relation>排名|位居|位列|排行)\s*第(?P<rank>[1-9]\d{0,5})(?:位|名)?(?!\d)")
_DATE_SLOT = re.compile(
    r"(?<![\d.])(?P<year>(?:19|20)\d{2})年[ \t]*"
    r"(?P<month>\d{1,2})?月[ \t]*(?P<day>\d{1,2})?日")
_DAY_WORD = re.compile(
    r"均|常|度|内|历|报|益|渐|用|化|产|语|光|元|记|夜|趋|后|前|子|间|期|程|照|线|"
    r"量|增|耗|销|赚|交易|营业|成交|出|落|晒|本")
_DATE_DAY_CONTINUATION = re.compile(
    r"(?:$|[，,。；;！？、：:）)】\]]|"
    r"(?:和|及|与|或|至|到)\s*(?:(?:19|20)\d{2}年\s*)?"
    r"(?:0?[1-9]|1[0-2])月\s*(?:0?[1-9]|[12]\d|3[01])日|"
    r"(?:已|将|拟)?(?:完成|公告|发布|披露|召开|启动|转入|投产|开工|竣工|交付|生效|上市))")


def missing_date_slots(content: str) -> list[dict]:
    """Explicit year-month-day grammar with an unfilled month or day.

    A missing numeral is distinct from an impossible filled calendar date.
    Ordinary month-level daily measures and date-format instructions are not
    blanks. No unknown numeral or replacement date is inferred.
    """
    out = []
    for match in _DATE_SLOT.finditer(content):
        month, day = match['month'], match['day']
        if month is not None and day is not None:
            continue
        if (month is not None and not 1 <= int(month) <= 12
                or day is not None and not 1 <= int(day) <= 31):
            continue
        if day is None:
            suffix = content[match.end():].lstrip()
            # "日" may begin an entity or an ordinary daily measure. Unknown
            # continuations abstain rather than expanding a blacklist of names.
            if _DAY_WORD.match(suffix) or not _DATE_DAY_CONTINUATION.match(suffix):
                continue
        left = max((content.rfind(c, 0, match.start()) for c in "。；;！？\r\n"), default=-1) + 1
        right = min((p for c in "。；;！？\r\n" if (p := content.find(c, match.end())) >= 0), default=len(content))
        clause = content[left:right]
        if re.search(r"模板|格式|占位|空白表|填表|填写|填报|示例|例如|假设|误写|更正|纠正", clause):
            continue
        out.append({'start': match.start(), 'end': match.end(),
                    'year': int(match['year']),
                    'month': int(month) if month is not None else None,
                    'day': int(day) if day is not None else None,
                    'missing_fields': [name for name, value in (('month', month), ('day', day)) if value is None]})
    return out


def paired_series(content: str) -> list[dict]:
    """Compare only two adjacent, explicitly aligned, fully unit-bearing lists."""
    result = []
    for match in _PAIRED_SERIES.finditer(content):
        # A match must not start in the middle of an unsupported preceding list.
        prefix = content[max(0, match.start() - 2):match.start()].rstrip()
        if prefix.endswith(("/", "／", "、")):
            continue
        suffix = content[match.end():match.end() + 12].lstrip()
        if suffix and suffix[0] not in "，,。；;！？\r\n)）":
            continue
        values = list(re.finditer(rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})", match['values']))
        rates = list(re.finditer(_RATE, match['rates']))
        # Mixed units/objects and per-item qualifiers need a richer parser.
        if len({v['unit'] for v in values}) != 1 or len(values) == len(rates):
            continue
        result.append({"start": match.start(), "end": match.end(),
                       "value_count": len(values), "rate_count": len(rates),
                       "value_tokens": [v.group() for v in values],
                       "rate_tokens": [v.group() for v in rates],
                       "relation": match['relation'], "unit": values[0]['unit']})
    return result


def rank_exceeds_explicit_total(content: str) -> list[dict]:
    """The same tightly bound industry ranking cannot exceed its stated size.

    No top-N selection, changing subject, unrelated company count, or inferred
    industry taxonomy is compared. The original statement supplies both numbers.
    """
    result = []
    for match in _RANK_WITH_TOTAL.finditer(content):
        total, rank = int(match["total"]), int(match["rank"])
        prefix = content[max(0, match.start() - 12):match.start()]
        if rank <= total or re.search(r"前\s*(?:申万)?$|选出\s*$|入选\s*$|选取\s*$", prefix):
            continue
        # Do not parse a decimal or interval's convenient integer prefix.
        if re.match(r"[.．\d]|\s*[-—~至]|家|公司|企业|个公司", content[match.end():]):
            continue
        result.append({"start": match.start(), "end": match.end(), "total_count": total,
                       "rank": rank, "scope": match["scope"], "relation": match["relation"]})
    return result


def closed_statement_context(content: str, start: int, end: int, *, max_chars: int = 800) -> tuple[int, int]:
    """One bounded, terminated statement for display, never preceding siblings.

    Semicolon members may share a date or subject. The containing statement is
    context only; callers retain their narrower source proof independently.
    """
    for statement in _FULL_STATEMENT.finditer(content):
        if statement.start() <= start and end <= statement.end():
            if len(statement.group()) <= max_chars and statement.group()[-1:] in "。！？":
                return statement.start(), statement.end()
            break
    return start, end


def closed_list_context(content: str, start: int, end: int) -> tuple[int, int]:
    """Keep the subject of a semicolon list, plus an explicit preceding sibling.

    A preceding sentence is included only for matching numbered item names;
    paragraphs, unrelated prose and unbounded lookback are never joined.
    """
    previous = None
    for statement in _FULL_STATEMENT.finditer(content):
        if statement.start() <= start and end <= statement.end():
            first = statement.start()
            label = _ORDINAL_SERIES.match(statement.group())
            old_label = _ORDINAL_SERIES.match(previous.group()) if previous else None
            if (label and old_label and label['name'] == old_label['name']
                    and not content[previous.end():statement.start()].strip()
                    and '\n' not in content[previous.end():statement.start()]
                    and statement.end() - previous.start() <= 800):
                first = previous.start()
            return first, statement.end()
        previous = statement
    return start, end


def reversed_date_ranges(content: str) -> list[dict]:
    """Candidate relations between legal dates, never a guessed correction."""
    out = []
    for pattern, granularity in ((_MONTH_RANGE, 'month'), (_DAY_RANGE, 'day')):
        for match in pattern.finditer(content):
            first, last = int(match['first']), int(match['last'])
            if first <= last:
                continue
            neighborhood = content[max(0, match.start() - 12):match.end() + 16]
            if re.search(r"跨年|跨月|次年|翌年|次月|翌月|倒序|逆序|降序|错误|示例|请勿|不应|误写|更正|纠正|假设|假如", neighborhood):
                continue
            if granularity == 'day':
                try:
                    # 2000 admits Feb 29 without assuming an unknown year.
                    year = int(match['year'] or 2000)
                    date(year, int(match['month']), first)
                    date(year, int(match['month']), last)
                except ValueError:
                    continue
            out.append({'start': match.start(), 'end': match.end(),
                        'first': first, 'last': last, 'granularity': granularity,
                        'same_period_basis': 'explicit_year' if granularity == 'month'
                        else 'abbreviated_same_month', 'requires_review': True})
    return out


def display_interval(value: str, scale: Decimal) -> tuple[Decimal, Decimal]:
    """Conservative interval under ordinary rounding at the displayed precision.

    The interval is used only to abstain from an alleged contradiction. It does
    not verify equality or assign either printed value as the true number.
    """
    number = Decimal(value.replace(',', ''))
    half_step = Decimal(1).scaleb(number.as_tuple().exponent) * scale / 2
    center = number * scale
    return center - half_step, center + half_step
