"""Bind one source-explicit backwards year period to a review-only relation.

This is candidate alignment, not a new detector.  It requires valid source year
endpoints, explicit period semantics and a single matching allegation.  It does
not infer corrections, reinterpret historical lists, or automatically confirm
business errors.  Full dates and month/day relations are deliberately outside
this module's scope.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
import re


_RANGE = re.compile(
    r"(?<![\d.])(?P<first>\d{4})\s*年?\s*[-—–~～至到]\s*"
    r"(?P<last>\d{4})(?:\s*年)?(?!\d)")
_ROLE_SUFFIX = re.compile(r"\s*(?:财年|会计年度|期间|年间)")
_ROLE_PREFIX = re.compile(
    r"(?:统计|报告|考核|执行|服务|项目|会计|业绩|数据|有效)?"
    r"(?:期间|区间|周期|起止年份|起止年度)(?:为|是|[:：])?\s*$")
_STATEMENT_BREAK = re.compile(r"[。！？!?\n]")
_NON_ASSERTED_PERIOD = re.compile(
    r"从近到远|由近及远|从后往前|从晚到早|倒序(?:列|排)|逆序(?:列|排)|"
    r"降序(?:列|排)|按年份降序|历史回溯|逆年代|公元前|"
    r"示例|例如|譬如|假设|假如|误写|误填|更正|纠正|不应|并非|不是|请勿|禁止|"
    r"相减|减去|差值|年差|相差|计算式|算式|[=＝]")
_LEGACY_REASON = "句中年份顺序倒置（晚于起始年份的表述早于起始年份），需人工核对期间。"
_FAMILY = "reversed_explicit_year_period"
_MAX_STATEMENT_CHARS = 800


def _span(content: str, start: int, end: int) -> dict:
    return {"start": start, "end": end, "text": content[start:end]}


def _valid_span(value: object, content: str) -> bool:
    return (isinstance(value, dict) and type(value.get("start")) is int
            and type(value.get("end")) is int and isinstance(value.get("text"), str)
            and 0 <= value["start"] < value["end"] <= len(content)
            and content[value["start"]:value["end"]] == value["text"])


def _statement(content: str, start: int, end: int) -> dict:
    previous = list(_STATEMENT_BREAK.finditer(content, 0, start))
    left = previous[-1].end() if previous else 0
    following = _STATEMENT_BREAK.search(content, end)
    right = following.end() if following else len(content)
    return _span(content, left, right)


def _source_period(candidate: dict, content: str) -> dict | None:
    spans = candidate.get("spans")
    if not isinstance(spans, list) or len(spans) != 1 or not _valid_span(spans[0], content):
        return None
    quoted = spans[0]
    matches = list(_RANGE.finditer(content, quoted["start"], quoted["end"]))
    if len(matches) != 1:
        return None
    match = matches[0]
    first, last = int(match["first"]), int(match["last"])
    try:
        date(first, 1, 1)
        date(last, 1, 1)
    except ValueError:
        return None
    if first <= last:
        return None
    statement = _statement(content, match.start(), match.end())
    if statement["end"] - statement["start"] > _MAX_STATEMENT_CHARS:
        return None
    if not (statement["start"] <= quoted["start"] < quoted["end"] <= statement["end"]):
        return None
    if (_NON_ASSERTED_PERIOD.search(statement["text"])
            or len(list(_RANGE.finditer(content, statement["start"], statement["end"]))) != 1):
        return None
    # The year-range parser includes the final 年. For 年间 that character is
    # also part of the period-role token: retain the full exact role evidence,
    # rather than inventing an isolated 间 role or truncating the range quote.
    suffix_start = match.end()
    if content[match.end() - 1:match.end() + 1] == "年间":
        suffix_start -= 1
    suffix = _ROLE_SUFFIX.match(content, suffix_start, statement["end"])
    prefix_start = max(statement["start"], match.start() - 24)
    prefix = _ROLE_PREFIX.search(content[prefix_start:match.start()])
    if suffix:
        role = _span(content, suffix_start, suffix.end())
        semantics = "explicit_suffix_period"
    elif prefix:
        role = _span(content, prefix_start + prefix.start(), match.start())
        semantics = "explicit_prefix_period"
    else:
        return None
    return {"anchor": _span(content, match.start(), match.end()), "display": statement,
            "role_anchor": role, "first_year": first, "last_year": last,
            "period_semantics": semantics}


def _single_year_order_reason(reason: str, first: int, last: int) -> bool:
    compact = re.sub(r"\s+", "", reason.translate(str.maketrans("", "", "\"'“”‘’「」")))
    if not compact or len(compact) > 320:
        return False
    year_range = rf"{first}年?[-—–~～至到]{last}年?"
    role = r"(?:财年区间|财年|会计年度|时间范围|时间区间|年份区间|年度区间|统计期间|期间|区间|起止年份|起止年度)"
    subject = rf"(?:(?:该|上述)?{role}(?:{year_range})?|{year_range}(?:{role})?)"
    order = (r"(?:的)?(?:起始年份|开始年份|起始年度|起始年|开始年|起始时间|开始时间|起点)"
             r"(?:晚于|大于)(?:结束年份|终止年份|结束年度|结束年|终止年|结束时间|终止时间|终点)")
    reversed_order = (r"(?:的)?(?:年份|起止年份|起止年度|起止顺序|时间顺序|顺序)?"
                      r"(?:倒序|倒置|颠倒)(?:不成立|不合理)?")
    accusation = re.compile(rf"{subject}(?:{order}|{reversed_order})")
    explicit_order = re.compile(rf"{first}年(?:晚于|大于){last}年")
    seen = False
    for clause in [x for x in re.split(r"[，,；;。.!！?？]+", compact) if x]:
        clause = re.sub(r"^(?:原文中|原文|因此|所以)", "", clause)
        if accusation.fullmatch(clause) or explicit_order.fullmatch(clause):
            seen = True
        elif seen and clause in {"属于时间矛盾", "需人工核对期间", "需核对期间"}:
            continue
        else:
            return False
    return seen


def bind_reversed_year_range(candidate: dict, content: str) -> dict | None:
    """Return an auditable, review-only binding; never mutate ``candidate``.

    Integrators may apply the returned canonical type, source identity, display,
    evidence and alignment while retaining raw candidate provenance.  Calendar
    or mixed reasons return None even if their quote contains a reversed range.
    """
    if not isinstance(candidate, dict) or not isinstance(content, str):
        return None
    if candidate.get("error_type") not in {"时间信息非法", "时间矛盾"}:
        return None
    reason = candidate.get("reason")
    if not isinstance(reason, str):
        return None
    period = _source_period(candidate, content)
    if period is None or not _single_year_order_reason(reason, period["first_year"], period["last_year"]):
        return None
    anchor = period["anchor"]
    key = sha256(json.dumps([_FAMILY, anchor], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    check = {"check": "valid_year_endpoints_in_reverse_order", "first_year": period["first_year"],
             "last_year": period["last_year"], "endpoints_are_valid_calendar_years": True,
             "period_semantics": period["period_semantics"], "role_anchor": deepcopy(period["role_anchor"]),
             "business_error_confirmed": False, "corrected_years_inferred": False}
    structure = {"family": _FAMILY, "anchors": [deepcopy(anchor)], "canonical_type": "时间矛盾",
                 "proof_status": "needs_review", "checks": [check]}
    return {"source_issue_key": key, "canonical_type": "时间矛盾", "status": "needs_review",
            "validation": "source_bound_temporal_relation", "reason": reason,
            "spans": [deepcopy(period["display"])],
            "evidence": [{"kind": "source_text", **deepcopy(anchor)},
                         {"kind": "source_text", **deepcopy(period["role_anchor"])},
                         {"kind": "source_relation", **deepcopy(check)}],
            "source_structure": structure,
            "source_content_sha256": sha256(content.encode()).hexdigest(),
            "source_alignment": {"status": "unique_source_structure", "source_issue_key": key,
                                 "original_error_type": candidate["error_type"], "original_reason": reason,
                                 "original_spans": deepcopy(candidate["spans"])}}


def same_reversed_year_issue(legacy: dict, binding: dict, content: str) -> bool:
    """Allow only the known legacy year-order finding for this exact range.

    The legacy detector's broad sentence quote alone is not enough. The same
    unique source period must be independently recovered, and its known single
    year-order reason must be intact. Keep both rule and model provenance when
    the caller merges; neither finding becomes confirmed.
    """
    if not isinstance(legacy, dict) or not isinstance(binding, dict) or not isinstance(content, str):
        return False
    if (legacy.get("detector_id") != "legacy.C.INTRINSIC.002"
            or legacy.get("error_type") != "时间矛盾"
            or legacy.get("status") != "needs_review"
            or legacy.get("reason") != _LEGACY_REASON
            or binding.get("source_content_sha256") != sha256(content.encode()).hexdigest()
            or binding.get("canonical_type") != "时间矛盾" or binding.get("status") != "needs_review"):
        return False
    original = binding.get("source_alignment", {})
    rebuilt = bind_reversed_year_range({"error_type": original.get("original_error_type"),
                                       "reason": original.get("original_reason"),
                                       "spans": original.get("original_spans")}, content)
    if rebuilt is None or rebuilt != binding:
        return False
    period = _source_period(legacy, content)
    return period is not None and period["anchor"] == binding["source_structure"]["anchors"][0]
