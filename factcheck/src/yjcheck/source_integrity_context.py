"""Narrow source-surface warnings, never error labels or missing-value claims.

These signals describe the supplied text, which may itself be an excerpt.
They do not reconstruct a PDF, repair text, suppress candidates, or prove that
the author made a business error. Unmarked text is not certified as intact.
"""
from __future__ import annotations

import re


_NUMBER = r"[+＋\-−－]?\d{1,12}(?:\.\d{1,8})?"
_MONEY_UNIT = r"(?:亿元|万元|亿美元|万美元|元|亿|万)"
_POWER_UNIT = r"(?:GW|MW|千瓦|万千瓦)"
_LIST_UNIT = r"(?:亿元|万元|万吨|万台|万套|GW|MW)"
_PRICE_UNIT = r"(?:万元|美元|元|万)\s*[/／]\s*(?:吨|公斤|千克|桶)"

# A direct reporting verb is required; unit-only table headings, zero values,
# approximate amounts and phrases such as "亿元级" are deliberately excluded.
_BARE_MAGNITUDE = re.compile(
    r"(?P<verb>实现|录得|达到)\s*"
    r"(?P<metric>营业总收入|营业收入|销售收入|归母净利润|扣非归母净利润|净利润)"
    r"\s*(?P<unit>亿元|万元|亿|万)(?=[，,；;。\n]|$)")

# Only known metric labels/prefixes after an already quantified financial metric
# are considered. A bare metric heading or an arbitrary unfinished sentence
# is not evidence of truncation.
_TAIL_PREFIX = re.compile(
    r"(?P<separator>[，,；;])\s*"
    r"(?P<prefix>加权|毛|扣非归母净利|归母净利|营业总收|营业收|净资产收益)\s*$")
_QUANTIFIED_METRIC = re.compile(
    rf"(?P<metric>基本每股收益|每股收益|营业总收入|营业收入|营收占比|营收|"
    rf"归母净利润|归母净利|毛利率|净利率)\s*(?:为|达|达到)?\s*"
    rf"(?P<number>{_NUMBER})\s*(?P<unit>{_MONEY_UNIT}|[%％])")

_CAPACITY_PRICE = re.compile(
    rf"(?P<metric>并网容量|装机容量|并网规模|装机规模|并网|装机)\s*"
    rf"(?:为|达|达到)?\s*(?P<number>{_NUMBER})\s*(?P<unit>{_PRICE_UNIT})"
    r"(?=[，,；;。\n]|$)")
_MONEY_TO_POWER = re.compile(
    rf"(?P<metric>累计投资|总投资|投资额|投资金额|投资|营业收入|营收)\s*"
    rf"(?:为|达|达到)?\s*(?P<amount>{_NUMBER})\s*(?P<amount_unit>亿元|万元|元)"
    rf"\s*[，,]\s*(?P<relation>同比|环比)(?:增长|增加|新增|下降|减少|增|降)\s*"
    rf"(?P<change>{_NUMBER})\s*(?P<change_unit>{_POWER_UNIT})(?=[，,；;。\n]|$)")

# An explicitly bounded ranking/list is required; open examples and ordinary
# prose lists do not imply any required item. We report the separator itself,
# not an inferred number or identity of missing elements.
_DANGLING_LIST = re.compile(
    r"(?P<intro>(?:涨幅|跌幅|排名)?前[一二三四五六七八九十1-9]\s*(?:为|是)|"
    r"(?:收入|营收|增速|增长率|利润|金额|销量|产量)(?:同比增速)?分别(?:为|是)?)"
    r"(?P<items>[^。；;\n，,：:]{1,140}、[^。；;\n，,：:]{1,140})"
    r"(?P<trailing_separator>、)(?P<terminator>[；;，,。])")
_OPEN_LIST = re.compile(r"例如|比如|譬如|包括|包含|等|部分|举例|示例|不限于|详见|略|…")

# A total followed by a colon, then slash-separated components with identical
# units and a repeated *unchanged* role after another colon. A period/metric
# switch or labelled subset breaks the construction and is not paired.
_REPEATED_COMPONENT = re.compile(
    rf"(?P<total_metric>合计|总计|总额|总装机|招标|装机|销量|产量|收入)\s*"
    rf"(?:为|达|达到)?\s*(?P<total>{_NUMBER})\s*(?P<unit>{_LIST_UNIT})"
    rf"(?:\s*[，,]\s*(?:同比|环比)\s*{_NUMBER}[%％])?\s*[:：]\s*"
    rf"(?P<first_role>[\u4e00-\u9fff]{{2,8}})\s*(?P<first>{_NUMBER})\s*(?P=unit)\s*[/／]\s*"
    rf"(?P<second_role>[\u4e00-\u9fff]{{2,8}})\s*(?P<second>{_NUMBER})\s*(?P=unit)\s*[:：]\s*"
    rf"(?P<repeated_role>(?P=first_role))\s*(?P<repeated>{_NUMBER})\s*(?P=unit)"
    r"(?=[，,；;。\n]|$)")
_ROLE_QUALIFIER = re.compile(r"本周|上周|本月|上月|本年|去年|当期|前期|累计|新增|存量|其中|分项|小计")

# Signals may describe an input line ending, but must not create a relation by
# consuming a blank line between unrelated paragraphs or table headings.
_BARE_MAGNITUDE, _CAPACITY_PRICE, _MONEY_TO_POWER, _REPEATED_COMPONENT = (
    re.compile(pattern.pattern.replace(r"\s", r"[^\S\r\n]"))
    for pattern in (_BARE_MAGNITUDE, _CAPACITY_PRICE, _MONEY_TO_POWER, _REPEATED_COMPONENT)
)


def _span(text: str, start: int, end: int, offset: int) -> dict:
    return {"start": start + offset, "end": end + offset, "text": text[start:end]}


def _warning(text: str, match: re.Match, offset: int, kind: str,
             observation: str, groups: tuple[str, ...], *, start: int | None = None) -> dict:
    return {
        "kind": kind,
        "scope": "source_structure_only_not_business_verdict",
        "source": _span(text, match.start() if start is None else start, match.end(), offset),
        "surface_signal": observation,
        "tokens": {name: _span(text, *match.span(name), offset) for name in groups},
        "relationship_binding_proven": False,
        "missing_field_proven": False,
        "business_error_proven": False,
        "interpretation": (
            "仅是该局部原文的表面结构信号。先复核相邻数值、单位和角色是否确实属于同一关系，"
            "再使用算术；不能据此确认缺失字段、补造内容、判业务错误或删除候选。"
            "未触发的文本不代表完整，其他段落不受此局部信号裁定。"
        ),
    }


def source_integrity_checks(text: str, *, offset: int = 0, limit: int = 12) -> list[dict]:
    """Return bounded, exactly anchored surface signals from this text alone.

    ``offset`` only translates positions; no surrounding text or document ID is
    consulted. A supplied excerpt ending at a metric prefix can be a slicing
    boundary rather than damage in the original document.
    """
    if not isinstance(text, str) or type(offset) is not int or offset < 0 or type(limit) is not int or limit < 0:
        raise ValueError("text, nonnegative integer offset and limit required")
    if limit == 0:
        return []
    checks = []
    for match in _BARE_MAGNITUDE.finditer(text):
        checks.append(_warning(text, match, offset, "printed_magnitude_after_metric_without_number",
            "报告动词后的指标直接连接量级/单位词，中间没有打印数字；可能是表述或输入损坏，尚未证实缺值。",
            ("verb", "metric", "unit")))
    for paragraph in re.finditer(r"[^\r\n]+", text):
        match = _TAIL_PREFIX.search(text, paragraph.start(), paragraph.end())
        if match is None:
            continue
        previous = list(_QUANTIFIED_METRIC.finditer(text, max(paragraph.start(), match.start() - 180), match.start()))
        if not previous:
            continue
        quantified = previous[-1]
        # A sentence boundary separates a complete disclosure from a heading.
        if re.search(r"[。！？!?]", text[quantified.end():match.start()]):
            continue
        check = _warning(text, match, offset, "printed_metric_prefix_at_text_boundary",
            "已有量化指标之后，当前行在分隔符和另一指标词/前缀处结束；无法确定是切片、标题还是未完表述。",
            ("separator", "prefix"), start=quantified.start())
        check["tokens"]["preceding_quantified_metric"] = _span(text, quantified.start(), quantified.end(), offset)
        check["supplied_text_boundary_only"] = True
        checks.append(check)
    for match in _CAPACITY_PRICE.finditer(text):
        checks.append(_warning(text, match, offset, "capacity_phrase_directly_followed_by_price_unit",
            "并网/装机词紧接按质量或体积计价的单位，未出现新的价格指标标签；需核对相邻片段绑定。",
            ("metric", "number", "unit")))
    for match in _MONEY_TO_POWER.finditer(text):
        checks.append(_warning(text, match, offset, "money_metric_growth_followed_by_power_unit",
            "金额指标后紧接同比/环比变化和功率单位，中间没有新的指标标签；需核对是否发生文本拼接。",
            ("metric", "amount", "amount_unit", "relation", "change", "change_unit")))
    for match in _DANGLING_LIST.finditer(text):
        local_start = max((text.rfind(mark, 0, match.start()) + 1 for mark in "。；;\n"), default=0)
        if _OPEN_LIST.search(text[max(local_start, match.start() - 60):match.end()]):
            continue
        checks.append(_warning(text, match, offset, "bounded_list_separator_before_terminator",
            "带明确排名/分别语法的列举，在顿号后立即结束；仅确认尾部分隔结构，不推断缺哪一字段。",
            ("intro", "items", "trailing_separator", "terminator")))
    for match in _REPEATED_COMPONENT.finditer(text):
        roles = [match[name] for name in ("first_role", "second_role")]
        if roles[0] == roles[1] or any(_ROLE_QUALIFIER.search(role) for role in roles):
            continue
        checks.append(_warning(text, match, offset, "repeated_role_inside_explicit_component_sequence",
            "总量后的同单位分解序列，在另一冒号后重复同一个分项标签；各数的层级或期间绑定尚未确定。",
            ("total_metric", "total", "first_role", "first", "second_role", "second", "repeated_role", "repeated")))
    return sorted(checks, key=lambda check: (check["source"]["start"], check["kind"]))[:limit]
