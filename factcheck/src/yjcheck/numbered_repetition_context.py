"""Source-only adjacent numbered-member copies for the reviewer context.

This module emits exact source comparisons, not errors or business decisions.
Additional sentences after a copied prefix remain explicitly outside the pair.
"""
import re

_PAREN = re.compile(r"(?<![\w])(?P<open>[（(]?)(?P<n>[1-9]\d?)(?P<close>[）)])")
_NO_CONTEXT = re.compile(r"示例|例如|譬如|假设|假如|练习|例题|模板|请勿|不应|误写|更正|纠正|不得|禁止")


def _span(content, start, end):
    return {"start": start, "end": end, "text": content[start:end]}


def adjacent_numbered_pairs(content):
    """Parse a narrow uninterrupted ordinal sequence within one paragraph.

    Bare leading numbers are not stripped: quantities, periods and ranks need
    a richer chapter parser. Nested, reset, mixed-style, nonconsecutive and threefold repeated sequences
    abstain. A later item may have additional complete sentences; only its exact
    leading copy of the previous whole body belongs to this pair.
    """
    output = []
    for paragraph in re.finditer(r"(?:[^\r\n]|\r?\n(?![ \t]*\r?\n))+", content):
        text, offset = paragraph.group(), paragraph.start()
        markers = []
        for match in _PAREN.finditer(text):
            opening, closing = match["open"], match["close"]
            if opening and {"（": "）", "(": ")"}[opening] != closing:
                continue
            markers.append({"start": match.start(), "end": match.end(), "number": int(match["n"]),
                            "style": opening + closing})
        markers.sort(key=lambda m: m["start"])
        if (not 2 <= len(markers) <= 32 or len({m["style"] for m in markers}) != 1
                or any(b["number"] != a["number"] + 1 for a, b in zip(markers, markers[1:]))):
            continue
        prefix = text[:markers[0]["start"]]
        # Keep the nearest complete sentence, including its terminal mark.
        # Splitting after the last full stop would lose a punctuated speaker
        # introduction or example warning immediately before the first item.
        introductions = [m for m in re.finditer(r"[^。！？\r\n]+(?:[。！？][”’」』]*)?", prefix)
                         if m.group().strip()]
        introduction = introductions[-1] if introductions else None
        intro_span = _span(content, offset + (introduction.start() if introduction else len(prefix)),
                           offset + (introduction.end() if introduction else len(prefix)))
        intro = intro_span["text"]
        preceding_context = None
        if not prefix.strip():
            preceding_end = len(content[:offset].rstrip())
            preceding_start = content.rfind("\n", 0, preceding_end) + 1
            if preceding_end > preceding_start:
                preceding_context = _span(content, preceding_start, preceding_end)
                intro += preceding_context["text"]
        if _NO_CONTEXT.search(intro):
            continue
        items = []
        for i, marker in enumerate(markers):
            end = markers[i + 1]["start"] if i + 1 < len(markers) else len(text)
            raw = text[marker["end"]:end]
            body_start = marker["end"] + len(raw) - len(raw.lstrip())
            body_end = end - (len(raw) - len(raw.rstrip()))
            items.append({**marker, "body_start": body_start, "body_end": body_end,
                          "body": text[body_start:body_end]})
        for i, (left, right) in enumerate(zip(items, items[1:])):
            body = left["body"]
            if (not 10 <= len(body) <= 1500 or body[-1:] not in "。！？；;"
                    or not right["body"].startswith(body) or _NO_CONTEXT.search(body)):
                continue
            # A third equal prefix makes a two-member allegation ambiguous.
            if sum(item["body"].startswith(body) for item in items) != 2:
                continue
            a, b = offset + left["body_start"], offset + right["body_start"]
            members = [_span(content, offset + left["start"], a + len(body)),
                       _span(content, offset + right["start"], b + len(body))]
            full = _span(content, members[0]["start"], members[1]["end"])
            output.append({"family": "literal_adjacent_numbered_members", "body": body,
                           "list_intro": intro_span, "preceding_context": preceding_context,
                           "members": members, "body_members": [_span(content, a, a + len(body)), _span(content, b, b + len(body))],
                           "numbers": [left["number"], right["number"]], "style": left["style"],
                           "full_span": full, "right_member_has_additional_text": len(right["body"]) > len(body),
                           "excluded_right_tail": _span(content, b + len(body), offset + right["body_end"]) if len(right["body"]) > len(body) else None,
                           "localization_only": True, "source_occurrence_count_within_list": 2,
                           "business_correctness_confirmed": False})
    return output


def source_numbered_repeat_checks(content, *, offset=0, limit=6):
    """Keep original text and map all local positions to their input window."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 0:
        raise ValueError("offset and limit must be nonnegative integers")
    def shifted(value):
        if isinstance(value, dict):
            return {key: item + offset if key in {"start", "end"} and type(item) is int else shifted(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [shifted(item) for item in value]
        return value
    return [{"kind": "source_numbered_member_copy", "source": shifted(pair["full_span"]),
             **shifted(pair), "scope": "current_visible_context_only",
             "instruction": "Only text equality and ordinal positions are observed. Review discourse roles and additional information; do not infer a business verdict."}
            for pair in adjacent_numbered_pairs(content)[:limit]]
