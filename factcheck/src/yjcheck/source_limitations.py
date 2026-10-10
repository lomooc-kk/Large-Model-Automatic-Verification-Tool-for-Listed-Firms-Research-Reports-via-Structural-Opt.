"""Explicit, exactly bound source-quality metadata; never a detection verdict.

This module does not discover limitations from prose, read files, authenticate
images, repair source text, or inspect/remove error candidates. Artifact hashes
are caller-supplied provenance. Only the UTF-8 input hash and character anchors
are checked here. Source pages are 1-based and boxes use PDF points with a
top-left origin, matching the caller's rendered-page coordinate mapping.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import math
import re


SCHEMA_VERSION = "source-limitations/1.0"
_KINDS = frozenset({"text_overlap", "occluded_text", "unreadable_text",
                    "extraction_gap", "reading_order_uncertain"})
_TEXT_ROLES = frozenset({"non_source_marker", "source_fragment", "layout_separator"})
_METHODS = frozenset({"visual_review", "parser_evidence"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ITEMS = 64

_INTERPRETATION = (
    "这是调用者提供并与输入精确绑定的局部来源质量记录，不是报告业务错误的证明。"
    "只能对 visible_spans 中本次可见的受影响片段使用该提示；declared_span 记录原始范围，"
    "不表示本次已经看见或判定整个范围。不得仅凭来源缺口断言数值、属性或金融要素缺失，"
    "不得补造被遮挡文字或把可辨片段当完整精确操作数。"
    "此提示也不证明原报告正确，不授权删除、隐藏或自动降级任何候选。"
    "附近可读的真实空字段、其他数值关系、独立错误仍须按其自身证据审查，不能整句或整表豁免。"
)
_ROLE_NOTES = {
    "non_source_marker": "该范围是整理端的非原件结构标记，不是作者留下的空值或原报告字句。",
    "source_fragment": "该范围保留来源的可辨片段；调用者声明其可读性或排列仍有限制，不保证完整。",
    "layout_separator": "该范围是布局分隔或空白的表示；不能仅由其存在判定作者遗漏必需字段。",
}


def _object(value: object, keys: set[str], name: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _ranges(value: object) -> list[tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("context_ranges must be a sequence of (start, end) pairs")
    ranges = []
    for pair in value:
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray)) or len(pair) != 2
                or type(pair[0]) is not int or type(pair[1]) is not int
                or not 0 <= pair[0] < pair[1]):
            raise ValueError("context range must contain nonnegative integer start < end")
        ranges.append((pair[0], pair[1]))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def project_source_limitation_hints(hints: Sequence[dict], context_ranges: Sequence[tuple[int, int]]) -> list[dict]:
    """Project ``build_source_limitation_hints`` results onto actual context ranges.

    Coordinates are Unicode character indices into the same original content.
    Only overlapping portions are exposed as ``visible_spans``; a gap between
    two windows stays a gap. Projection never modifies its input and can be
    repeated from an earlier projection because ``declared_span`` is retained.
    This accepts normalized builder results, not raw caller metadata.

    This is an internal/audit structure: ``declared_span.text`` still contains
    the full declared quote. Before putting it on a model request, the caller
    must copy the hint and retain only start/end in ``declared_span``; source
    text on that wire projection must come only from ``visible_spans``.
    """
    ranges = _ranges(context_ranges)
    projected = []
    for hint in hints:
        # The original bounds, rather than a prior visible projection, control
        # the next projection. No prose or neighbouring field is consulted.
        original = hint["declared_span"]
        start, end, text = original["start"], original["end"], original["text"]
        visible = []
        for left, right in ranges:
            lo, hi = max(start, left), min(end, right)
            if lo < hi:
                visible.append({"start": lo, "end": hi, "text": text[lo - start:hi - start]})
        if not visible:
            continue
        result = deepcopy(hint)
        result["visible_spans"] = visible
        result["context_complete"] = sum(s["end"] - s["start"] for s in visible) == end - start
        projected.append(result)
    return projected


def build_source_limitation_hints(content: str, metadata: dict | None = None, *,
                                  context_ranges: Sequence[tuple[int, int]] | None = None) -> list[dict]:
    """Validate all explicit metadata, then return local source-quality hints.

    ``None`` and a valid envelope with no items produce no hints. Nothing is
    inferred from keywords, placeholders, document names, or model output.
    Invalid metadata fails atomically, including invalid items outside selected
    windows; there is no partial acceptance, clipping of bad anchors, fuzzy
    reanchoring, or silent item limit. The caller must account for any added
    hints in its ordinary request budget and omit the payload field when empty.
    """
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if context_ranges is not None:
        ranges = _ranges(context_ranges)
        if any(end > len(content) for _, end in ranges):
            raise ValueError("context range extends beyond content")
    else:
        ranges = [(0, len(content))] if content else []
    if metadata is None:
        return []
    obj = _object(metadata, {"schema_version", "input_sha256", "items"}, "metadata")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported source limitations schema")
    if _hash(obj["input_sha256"], "input_sha256") != sha256(content.encode("utf-8")).hexdigest():
        raise ValueError("source limitations input_sha256 does not match content")
    if not isinstance(obj["items"], list) or len(obj["items"]) > _MAX_ITEMS:
        raise ValueError(f"items must be a list with at most {_MAX_ITEMS} entries")
    hints, seen = [], set()
    for index, value in enumerate(obj["items"]):
        item = _object(value, {"kind", "span", "text_role", "evidence"}, "item")
        kind, role = item["kind"], item["text_role"]
        if not isinstance(kind, str) or kind not in _KINDS:
            raise ValueError("unknown source limitation kind")
        if not isinstance(role, str) or role not in _TEXT_ROLES:
            raise ValueError("unknown source text_role")
        span = _object(item["span"], {"start", "end", "text"}, "span")
        start, end, text = span["start"], span["end"], span["text"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content)
                or not isinstance(text, str) or content[start:end] != text):
            raise ValueError("source limitation span must exactly match content")
        if role == "layout_separator" and not text.isspace():
            raise ValueError("layout_separator span must contain only source whitespace")
        if role != "layout_separator" and not text.strip():
            raise ValueError("blank spans require explicit layout_separator role")
        key = (kind, role, start, end)
        if key in seen:
            raise ValueError("duplicate source limitation item")
        seen.add(key)
        evidence = _object(item["evidence"], {"source_sha256", "page", "image_sha256", "bbox",
                                                "bbox_units", "method"}, "evidence")
        _hash(evidence["source_sha256"], "source_sha256")
        _hash(evidence["image_sha256"], "image_sha256")
        if type(evidence["page"]) is not int or evidence["page"] < 1:
            raise ValueError("source page must be a 1-based integer")
        if evidence["bbox_units"] != "pdf_points":
            raise ValueError("bbox_units must be pdf_points with top-left origin")
        box = evidence["bbox"]
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in box)
                or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
            raise ValueError("bbox must be finite nonnegative x0 < x1 and y0 < y1")
        if not isinstance(evidence["method"], str) or evidence["method"] not in _METHODS:
            raise ValueError("unknown source evidence method")
        hints.append({
            "schema_version": SCHEMA_VERSION,
            "limitation_index": index,
            "kind": kind,
            "text_role": role,
            "declared_span": dict(span),
            "visible_spans": [dict(span)],
            "context_complete": True,
            "source_evidence": {**evidence, "bbox": list(box)},
            "input_sha256": obj["input_sha256"],
            "scope": "source_quality_only_not_business_verdict",
            "evidence_verification": "caller_supplied_provenance_not_independently_authenticated",
            "business_error_proven": False,
            "business_error_excluded": False,
            "candidate_suppression_allowed": False,
            "interpretation": _ROLE_NOTES[role] + _INTERPRETATION,
        })
    return project_source_limitation_hints(hints, ranges)
