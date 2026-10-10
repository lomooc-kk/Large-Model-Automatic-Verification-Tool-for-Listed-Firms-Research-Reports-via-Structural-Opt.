"""Source-only unit comparisons between explicitly labelled EPS table rows.

No output is a confirmed financial fact. A monetary reference must independently
match at least two exact year/status columns; the suspect label alone is not
enough. The local label is the alleged error, while rows/headers are evidence.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

_YEAR = re.compile(r"(?:19|20)\d{2}(?:[AaEe]|年)?")
_NUMBER = re.compile(r"[+\-−－]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?")
_LABEL = re.compile(r"(?P<name>EPS|(?:基本|稀释|摊薄|调整后|调整前|扣非)?每股收益)"
                    r"\s*[（(](?P<unit>[^()（）]{1,12})[）)]", re.I)
_MONEY_UNITS = frozenset({'元', '元/股', '人民币元', '人民币元/股', '港元', '港元/股',
                         '美元', '美元/股', '摊薄/元', '基本/元', '稀释/元'})
_RATIO_UNITS = frozenset({'x', '倍'})
_SECTIONS = frozenset({'成长能力', '获利能力', '偿债能力', '营运能力', '估值比率',
                       '每股指标', '每股指标（元）', '每股指标(元)', '盈利能力'})
_NONASSERTION = re.compile(r'示例|例如|举例|例题|假设|假如|请勿|不得|误写|误填|更正|纠正|不应|不是|并非')


def _span(content, start, end):
    return {'start': start, 'end': end, 'text': content[start:end]}


def _cells(line, offset):
    """Delimited cells with source positions, never a whitespace column guess."""
    if '|' not in line and '\t' not in line:
        return []
    values, left = [], 0
    for right in [m.start() for m in re.finditer(r'[|\t]', line)] + [len(line)]:
        raw = line[left:right]
        token = raw.strip()
        start = offset + left + len(raw) - len(raw.lstrip())
        # Include consecutive separators: interior missing columns must not
        # disappear and accidentally realign later values to different years.
        values.append((token, start, start + len(token)))
        left = right + 1
    # Strip only table framing whitespace, not an interior missing value.
    if line.lstrip().startswith('|') and values and not values[0][0]:
        values.pop(0)
    if line.rstrip().endswith('|') and values and not values[-1][0]:
        values.pop()
    return values


def _years(cells):
    texts = [cell[0] for cell in cells]
    candidates = texts if texts and _YEAR.fullmatch(texts[0]) else texts[1:]
    if len(candidates) < 2 or not all(_YEAR.fullmatch(x) for x in candidates):
        return None
    years = [x.upper().removesuffix('年') for x in candidates]
    return years if len(set(years)) == len(years) else None


def _decimal(value):
    if not _NUMBER.fullmatch(value):
        return None
    try:
        return Decimal(value.replace(',', '').replace('，', '').replace('−', '-').replace('－', '-'))
    except InvalidOperation:
        return None


def eps_table_rows(content: str) -> list[dict]:
    """Read pipe/tab EPS rows only within explicit contiguous table contexts."""
    rows, header, offset, example_context = [], None, 0, False
    for full in content.splitlines(keepends=True):
        line = full.rstrip('\r\n')
        stripped = line.strip()
        cells = _cells(line, offset)
        years = _years(cells)
        if years:
            header = {'years': years, 'span': _span(content, offset, offset + len(line)),
                      'example_context': example_context or bool(_NONASSERTION.search(line))}
        elif not stripped or re.fullmatch(r'[|:\-\s]+', line):
            pass
        elif cells and header and len(cells) == len(header['years']) + 1:
            label = _LABEL.fullmatch(cells[0][0])
            values = [_decimal(cell[0]) for cell in cells[1:]]
            if label and all(value is not None for value in values) and not header['example_context']:
                unit = ''.join(label['unit'].split()).replace('／', '/')
                rows.append({'label': _span(content, cells[0][1], cells[0][2]),
                             'row': _span(content, offset, offset + len(line)),
                             'header': header['span'], 'unit': unit,
                             'values': dict(zip(header['years'], values))})
        elif stripped not in _SECTIONS:
            # A prose paragraph/new title or malformed row breaks the table.
            # The next explicit year header is needed to establish new columns.
            header = None
            example_context = bool(_NONASSERTION.search(stripped))
        offset += len(full)
    return rows


def eps_unit_comparisons(content: str) -> list[dict]:
    """Return review candidates backed by same-source monetary EPS rows.

    The output proposes neither a currency nor a corrected numeric value. A/E
    statuses are part of the key; 2025A does not match 2025E or bare 2025.
    """
    if not isinstance(content, str):
        raise TypeError('content must be source text')
    rows = eps_table_rows(content)
    references = [row for row in rows if row['unit'] in _MONEY_UNITS]
    output = []
    for target in rows:
        if target['unit'].lower() not in _RATIO_UNITS:
            continue
        matches = []
        for reference in references:
            common = sorted(set(target['values']) & set(reference['values']))
            if len(common) >= 2 and all(target['values'][year] == reference['values'][year] for year in common):
                matches.append({'label': reference['label'], 'row': reference['row'],
                                'header': reference['header'], 'unit': reference['unit'],
                                'matching_years': common,
                                'matching_values': {year: str(reference['values'][year]) for year in common}})
        if matches:
            output.append({'label': target['label'], 'row': target['row'], 'header': target['header'],
                           'unit': target['unit'], 'references': matches,
                           'check': 'same_source_eps_unit_dimension_comparison',
                           'business_confirmation': False})
    return output
