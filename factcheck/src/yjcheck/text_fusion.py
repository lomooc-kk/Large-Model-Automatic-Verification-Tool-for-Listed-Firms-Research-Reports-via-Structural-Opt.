"""Conservative identity and type alignment from unique source structures.

Model labels/reasons select the alleged issue; they are never its proof. Only
bounded source parsers and their exact anchors establish the structure. No gold,
document-specific rules, inferred replacement values or external facts enter.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re

from .text_structure import closed_statement_context


_MISSING_TYPES = {"金融要素缺失", "数值缺失", "属性值缺失错误"}
_NAMED_NUMERIC = re.compile(
    r"(?P<field>营业总收入|营业收入|销售额均值|销售额|收入|归母净利润|净利润|"
    r"毛利率|净利率|增长率|增速|占比|比例|数量|均价|每股收益|市盈率)"
    r"\s*(?:约为|达到|为|是|达)\s*$")
_NAMED_ATTRIBUTE = re.compile(r"(?P<field>文件名称|法规名称|公司名称|证券代码|信用评级|评级|名称|类别)\s*(?:为|是)\s*$")
_DETECTORS = {
    "verified.empty_placeholder": ("empty_attribute", _MISSING_TYPES),
    "verified.suspended_punctuation": ("missing_unnamed_clause", _MISSING_TYPES),
    "verified.percent_gap": ("missing_numeric_slot", _MISSING_TYPES),
    "verified.ordinal_gap": ("missing_numeric_slot", _MISSING_TYPES),
    "verified.unit_gap": ("missing_numeric_slot", _MISSING_TYPES),
    "verified.date_numeric_gap": ("missing_date_numeric_slot", {"数值缺失", "时间信息非法", "属性值缺失错误"}),
    "verified.aligned_series_cardinality": ("aligned_sequence_cardinality", {"数值不一致错误", "计算错误", "语义逻辑矛盾"}),
    "verified.rank_within_total": ("rank_within_explicit_total", {"数值不一致错误", "计算错误", "语义逻辑矛盾"}),
    "structured.reversed_date_range": ("reversed_legal_date_range", {"时间矛盾", "时间信息非法"}),
    "verified.redundant_sentence": ("literal_adjacent_sentence_pair", {"冗余语句"}),
}

# This is a validator for an existing independent proof, not a new detector.
# Both members must include the same terminator; newline-separated passages do
# not become a pair merely because their text is identical.
_EXACT_SENTENCE_PAIR = re.compile(
    r"(?P<unit>[^。！？；\r\n]{8,200})(?P<terminator>[。！？；])(?P<gap>[ \t]*)"
    r"(?P=unit)(?P=terminator)")
_EXACT_PARAGRAPH_PAIR = re.compile(
    r"(?P<unit>[^。！？；\r\n]{8,200})(?P<terminator>[。！？])"
    r"(?P<gap>[ \t]*(?:\r?\n[ \t]*){1,3})(?P=unit)(?P=terminator)")
_EXACT_PHRASE_PAIR = re.compile(
    r"(?P<unit>[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9.%％+＋\-－×*÷/=＝]{3,39})"
    r"(?P<separator>[，,、；;])(?P<gap>[ \t]*)(?P=unit)")
_REPEAT_FAMILIES = {"literal_adjacent_sentence_pair", "literal_adjacent_paragraph_pair", "literal_adjacent_phrase_pair"}
_REASON_QUOTE = re.compile(r'“([^”]+)”|「([^」]+)」|『([^』]+)』|"([^"\n]+)"')
_REPEAT_DESCRIPTOR = r"(?:主体|对象|期间|时点|日期|状态|产能状态|风险)"
_SAME_ORIGINAL_REASON = re.compile(
    r"同一" + _REPEAT_DESCRIPTOR + r"(?:、" + _REPEAT_DESCRIPTOR + r"){0,4}的(?:原句|句子|语句)")
_SAME_REPEAT_REASON = re.compile(r"同一(?:风险|信息|内容)不必要地出现两次")
_REPEAT_REASON_WORDS = re.compile("|".join(re.escape(word) for word in sorted((
    "未新增信息", "没有新增信息", "无新增信息", "没有新增含义", "未提供新信息", "未提供新增信息", "没有新信息",
    "没有新增内容", "未增加信息", "未增加新信息", "不含新增信息", "无新增内容", "没有信息增量", "无信息增量",
    "重复出现", "重复表述", "重复陈述", "重复描述", "重复表达", "重复列出", "原样重复",
    "完全重复", "内容重复", "连续重复", "不必要的冗余", "同一完整句子", "同一完整句",
    "同一个句子", "同一句子", "同一句话", "同一句", "在原文中", "在相邻段落中", "同一内容", "同一信息", "相同内容", "相同信息",
    "两个相邻句子", "相邻两句", "两处", "两个", "两句", "整句", "完整句子", "句子",
    "语句", "文字", "信息", "内容", "该句", "这两句", "上述", "前后", "相邻", "连续",
    "出现两次", "出现一次", "重复一次", "重复两次", "重复", "冗余", "出现", "属于", "构成",
    "一致", "完全相同", "逐字相同", "完全", "逐字", "不必要", "的", "了", "是",
), key=len, reverse=True)))


def _single_repeat_allegation(reason, unit, *, require_unit_quote):
    """Reason selects one allegation; only the independently parsed source proves it.

    Residual wording must describe repetition only. Unknown extra clauses are
    deliberately not interpreted, so another error cannot inherit this proof.
    """
    reason = str(reason)
    quoted = [next(group for group in match.groups() if group is not None)
              for match in _REASON_QUOTE.finditer(reason)]
    # A bounded cause assertion can be named by its source subject instead of
    # quoted wholesale: "X主要系Y所致" -> "同一X原因". The subject is copied
    # from the proved unit, never guessed from a model's finance knowledge.
    cause = re.fullmatch(r"(?P<subject>[^，,。！？；;：:\r\n]{2,40})主要系[^。！？；\r\n]+所致", unit)
    source_description = "同一" + cause["subject"] + "原因" if cause else None
    named_source_cause = bool(source_description and source_description in reason and not quoted)
    if (require_unit_quote and len(quoted) != 1 and not named_source_cause) or len(quoted) > 1:
        return False
    if quoted and quoted[0].rstrip("。！？；") != unit:
        return False
    remainder = _REASON_QUOTE.sub("", reason)
    if named_source_cause:
        remainder = remainder.replace(source_description, "同一信息")
    # Finite descriptions of the same repeated assertion, not arbitrary noun
    # removal. Extra prices, chronology or other alleged issues remain residual.
    remainder = _SAME_ORIGINAL_REASON.sub("同一句话", remainder)
    remainder = _SAME_REPEAT_REASON.sub("同一信息出现两次", remainder)
    remainder = remainder.replace("连续重复两次", "连续重复")
    if not re.search(r"重复|冗余", remainder):
        return False
    remainder = re.sub(r"[\s，,。；;：:、（）()]", "", remainder)
    return not _REPEAT_REASON_WORDS.sub("", remainder)


def literal_sentence_pair(content, finding):
    """Return the two exact source members of a narrow, confirmed proof or abstain."""
    if finding.get("detector_id") != "verified.redundant_sentence" or finding.get("status") != "confirmed_error":
        return None
    anchors = finding.get("verification_spans", [])
    if len(anchors) != 1 or not _valid(anchors, content):
        return None
    span = anchors[0]
    match = _EXACT_SENTENCE_PAIR.fullmatch(span["text"])
    if not match or match["unit"] != match["unit"].strip():
        return None
    start = span["start"]
    if start and content[start - 1] not in "。！？；：:，,\r\n":
        return None
    left = max((content.rfind(char, 0, start) for char in "。！？；\r\n"), default=-1) + 1
    if re.search(r"示例|例如|譬如|比如|假设|假如|练习|例题|请勿|不应|并非|不是|误写|误填|更正|纠正|不得|禁止",
                 content[left:span["end"]]):
        return None
    unit = match["unit"]
    # Count the unit too, so a third occurrence without its final punctuation
    # cannot be hidden by counting only the two complete quoted sentences.
    occurrences = [m.start() for m in re.finditer("(?=" + re.escape(unit) + ")", content)]
    width = len(unit) + 1
    right_start = start + width + len(match["gap"])
    if occurrences != [start, right_start]:
        return None
    members = [{"start": pos, "end": pos + width, "text": content[pos:pos + width]}
               for pos in occurrences]
    return {"unit": unit, "terminator": match["terminator"], "members": members,
            "full_span": {key: span[key] for key in ("start", "end", "text")},
            "source_occurrence_count": 2, "same_paragraph": True,
            "independent_detector": finding["detector_id"]}


def _paragraph_sentence_pair(content, finding):
    """Locate two whole short paragraphs; paragraph roles still need review."""
    if (finding.get("detector_id") != "verified.redundant_sentence"
            or finding.get("status") != "confirmed_error"
            and finding.get("validation") != "paragraph_repetition_requires_review"):
        return None
    spans = finding.get("verification_spans") or finding.get("surface_spans", [])
    if len(spans) != 1 or not _valid(spans, content):
        return None
    span = spans[0]
    match = _EXACT_PARAGRAPH_PAIR.fullmatch(span["text"])
    if not match or match["unit"] != match["unit"].strip():
        return None
    unit, width = match["unit"], len(match["unit"]) + 1
    starts = [span["start"], span["start"] + width + len(match["gap"])]
    if [m.start() for m in re.finditer("(?=" + re.escape(unit) + ")", content)] != starts:
        return None
    members = [{"start": start, "end": start + width, "text": content[start:start + width]} for start in starts]
    for member in members:
        left = content.rfind("\n", 0, member["start"]) + 1
        right = content.find("\n", member["end"])
        if right < 0:
            right = len(content)
        if content[left:member["start"]].strip() or content[member["end"]:right].strip():
            return None
    return {"unit": unit, "terminator": match["terminator"], "members": members,
            "full_span": deepcopy(span), "source_occurrence_count": 2, "same_paragraph": False,
            "localization_only": True, "independent_detector": finding["detector_id"]}


def _discourse_prefixed_paragraph_pair(content, finding):
    """Locate an exact pair after a bounded discourse prefix, without confirming.

    The first source sentence may start with "其中"; it adds no second subject
    to the literal unit. The second member must begin the next line and finish
    its own sentence. Following sentences remain outside this issue. This is
    narrower than treating arbitrary titles, subjects, or summary prefixes as
    interchangeable, and needs the existing independent literal proof.
    """
    if (finding.get("detector_id") != "verified.redundant_sentence"
            or finding.get("status") != "confirmed_error"
            and finding.get("validation") != "paragraph_repetition_requires_review"):
        return None
    spans = finding.get("verification_spans") or finding.get("surface_spans", [])
    if len(spans) != 1 or not _valid(spans, content):
        return None
    proof = spans[0]
    match = _EXACT_PARAGRAPH_PAIR.fullmatch(proof["text"])
    if not match or match["unit"] != match["unit"].strip():
        return None
    unit, width = match["unit"], len(match["unit"]) + 1
    starts = [proof["start"], proof["start"] + width + len(match["gap"])]
    if [m.start() for m in re.finditer("(?=" + re.escape(unit) + ")", content)] != starts:
        return None
    left = max((content.rfind(mark, 0, starts[0]) for mark in "。！？；\r\n"), default=-1) + 1
    if content[left:starts[0]] != "其中":
        return None
    second_left = content.rfind("\n", 0, starts[1]) + 1
    if content[second_left:starts[1]].strip():
        return None
    members = [{"start": start, "end": start + width, "text": content[start:start + width]} for start in starts]
    prefix = {"start": left, "end": starts[0], "text": content[left:starts[0]]}
    first_context = {"start": left, "end": members[0]["end"], "text": content[left:members[0]["end"]]}
    return {"unit": unit, "terminator": match["terminator"], "members": members,
            "full_span": {"start": left, "end": proof["end"], "text": content[left:proof["end"]]},
            "literal_proof_span": deepcopy(proof), "source_member_contexts": [first_context, deepcopy(members[1])],
            "discourse_prefix": prefix, "source_occurrence_count": 2, "same_paragraph": False,
            "localization_only": True, "independent_detector": finding["detector_id"]}


def _member_covering_quotes(raw_spans, pair, content, context_ranges):
    """A unique assignment of unequal quotes to two already proved members.

    A quote may include a same-statement prefix, but must cover exactly one
    member. It cannot bridge into the other member or another sentence.
    """
    options = []
    for raw in raw_spans:
        hits = []
        for match in re.finditer("(?=" + re.escape(raw["text"]) + ")", content):
            start, end = match.start(), match.start() + len(raw["text"])
            if not any(left <= start < end <= right for left, right in context_ranges):
                continue
            for index, member in enumerate(pair["members"]):
                left, right = closed_statement_context(content, member["start"], member["end"])
                other = pair["members"][1 - index]
                if (left <= start <= member["start"] < member["end"] <= end <= right
                        and not start <= other["start"] < other["end"] <= end):
                    hits.append((index, start, end))
        options.append(hits)
    assignments = {(a, b) for a in options[0] for b in options[1] if a[0] != b[0]}
    if len(assignments) != 1:
        return None
    return [{"member_index": index, "start": start, "end": end, "text": content[start:end]}
            for index, start, end in assignments.pop()]


def _pair_covering_quotes(raw_spans, pair, content, context_ranges):
    """One unique wide quote locates the pair; the narrow quote stays joint.

    A narrow quote found at both members does not get an invented occurrence
    offset. Its two source matches are retained, while the independently proved
    pair is the issue location. No quote may absorb a following sentence.
    """
    first, second = pair["members"]
    lower, _ = closed_statement_context(content, pair["full_span"]["start"], first["end"])
    matches_by_quote = []
    wide_indices = []
    for index, raw in enumerate(raw_spans):
        matches = [{"start": m.start(), "end": m.start() + len(raw["text"]), "text": raw["text"]}
                   for m in re.finditer("(?=" + re.escape(raw["text"]) + ")", content)]
        if not matches or not all(any(left <= s["start"] < s["end"] <= right for left, right in context_ranges)
                                  for s in matches):
            return None
        if len(matches) == 1 and lower <= matches[0]["start"] <= first["start"] and matches[0]["end"] == second["end"]:
            wide_indices.append(index)
        elif raw["text"] == pair["unit"] + pair["terminator"] and matches == pair["members"]:
            pass
        else:
            return None
        matches_by_quote.append({"quote_index": index, "text": raw["text"], "source_matches": matches})
    if len(wide_indices) != 1:
        return None
    return {"quoted_source_matches": matches_by_quote, "wide_quote_index": wide_indices[0],
            "member_assignment": "joint_pair_not_individual_offset"}


def anchor_repeated_sentence_pair(raw, content, context_ranges, findings):
    """Resolve two missing offsets together, never two arbitrary ambiguous quotes.

    Called by both arms. It restores localization only; direct-model candidates
    still need review. Existing explicit offsets retain the ordinary protocol.
    """
    spans = raw.get("spans")
    if (raw.get("error_type") != "冗余语句" or not isinstance(spans, list) or len(spans) != 2
            or any(not isinstance(s, dict) or not isinstance(s.get("text"), str) or not s["text"]
                   or "start" in s or "end" in s for s in spans)):
        return None
    possibilities = []
    for finding in findings:
        pair = (literal_sentence_pair(content, finding) or _paragraph_sentence_pair(content, finding)
                or _discourse_prefixed_paragraph_pair(content, finding))
        if (not pair or not all(any(left <= s["start"] and s["end"] <= right for left, right in context_ranges)
                                for s in pair["members"])):
            continue
        equal = spans[0]["text"] == spans[1]["text"]
        if not _single_repeat_allegation(raw.get("reason", ""), pair["unit"], require_unit_quote=not equal):
            continue
        if equal and all(s["text"] == spans[0]["text"] for s in pair["members"]):
            possibilities.append({**pair, "anchor_method": "unique_adjacent_paragraph_pair" if pair.get("localization_only")
                                  else "unique_adjacent_sentence_pair"})
        elif not equal:
            quoted = (_member_covering_quotes(spans, pair, content, context_ranges)
                      if pair["same_paragraph"] or pair.get("discourse_prefix") else None)
            if quoted:
                possibilities.append({**pair, "quoted_source_spans": quoted,
                                      "anchor_method": "unique_member_covering_quotes"})
            else:
                joint = _pair_covering_quotes(spans, pair, content, context_ranges)
                if joint:
                    possibilities.append({**pair, **joint, "anchor_method": "unique_pair_covering_quote"})
    if len(possibilities) != 1:
        return None
    return {**possibilities[0],
            "source_content_sha256": sha256(content.encode()).hexdigest()}


def _valid(spans, content):
    return bool(spans) and all(isinstance(s, dict) and type(s.get("start")) is int
        and type(s.get("end")) is int and isinstance(s.get("text"), str)
        and 0 <= s["start"] < s["end"] <= len(content)
        and content[s["start"]:s["end"]] == s["text"] for s in spans)


def _key(family, anchors):
    payload = [family, [(s["start"], s["end"], s["text"]) for s in anchors]]
    return sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]


def _gap_field(content, start):
    """Only a still-unfilled named field in this comma-delimited clause counts.

    A field in an earlier sibling, or the already-filled '13家' before an empty
    following clause, cannot establish what that following clause should say.
    """
    left = max((content.rfind(char, 0, start) for char in "，,。；;！？\r\n"), default=-1) + 1
    clause = content[left:start]
    for pattern, family, kind in ((_NAMED_NUMERIC, "missing_numeric_slot", "数值缺失"),
                                  (_NAMED_ATTRIBUTE, "empty_attribute", "属性值缺失错误")):
        match = pattern.search(clause)
        if match:
            return family, kind, {"field": match["field"], "start": left + match.start(),
                                  "end": start, "text": content[left + match.start():start], "field_explicitly_named": True}
    return "missing_unnamed_clause", "金融要素缺失", {"field_explicitly_named": False}


def _repeat_display_spans(content, pair):
    proof = pair["full_span"]
    start, end = proof["start"], proof["end"]
    if pair["terminator"] == "；":
        start, end = closed_statement_context(content, start, end)
    return [{"start": start, "end": end, "text": content[start:end]}]


def _same_phrase_members(content, finding, pair):
    """The phrase proof is exactly these two bare members, not a nested suffix."""
    spans = finding.get("verification_spans", [])
    if (finding.get("detector_id") != "verified.redundant_phrase"
            or finding.get("status") != "confirmed_error" or pair["terminator"] != "；"
            or len(spans) != 1 or not _valid(spans, content)):
        return False
    first, second = pair["members"]
    return (spans[0]["start"] == first["start"]
            and spans[0]["end"] == second["end"] - len(pair["terminator"])
            and content[first["start"]:first["end"] - 1] == pair["unit"]
            and content[second["start"]:second["end"] - 1] == pair["unit"])


def _rule_proof(finding):
    return {key: deepcopy(finding[key]) for key in
            ("detector_id", "reason", "status", "spans", "verification_spans", "evidence") if key in finding}


def _literal_phrase_pair(content, finding):
    spans = finding.get("verification_spans", [])
    if (finding.get("detector_id") != "verified.redundant_phrase"
            or finding.get("status") != "confirmed_error" or len(spans) != 1 or not _valid(spans, content)):
        return None
    span = spans[0]
    match = _EXACT_PHRASE_PAIR.fullmatch(span["text"])
    if not match:
        return None
    if (span["start"] and content[span["start"] - 1] not in "，,、；;。！？\r\n：:（("
            or span["end"] < len(content) and content[span["end"]] not in "，,、；;。！？\r\n"):
        return None
    unit = match["unit"]
    starts = [span["start"], span["end"] - len(unit)]
    if [m.start() for m in re.finditer("(?=" + re.escape(unit) + ")", content)] != starts:
        return None
    return {"unit": unit, "separator": match["separator"], "terminator": "",
            "members": [{"start": start, "end": start + len(unit), "text": unit} for start in starts],
            "full_span": deepcopy(span), "source_occurrence_count": 2, "same_paragraph": True,
            "independent_detector": finding["detector_id"]}


def build_source_issues(content, findings):
    """Annotate supported source findings; retain their original proof status."""
    issues = []
    for finding in findings:
        detector = finding.get("detector_id")
        if detector not in _DETECTORS:
            continue
        family, compatible_types = _DETECTORS[detector]
        if (detector == "verified.redundant_sentence" and finding.get("status") == "confirmed_error"
                and any("\n" in span["text"] or "\r" in span["text"] for span in finding.get("verification_spans", []))):
            finding["source_rule_proofs"] = [_rule_proof(finding)]
            finding["source_original_verdict"] = {key: finding[key] for key in ("error_type", "status", "validation", "reason")}
            finding["surface_spans"] = finding.pop("verification_spans")
            finding.update(status="needs_review", validation="paragraph_repetition_requires_review",
                           reason="相邻段落存在字面相同内容；仅确认来源位置，需核对标题、摘要、示例等段落功能，不能自动判定冗余。")
        if (detector != "structured.reversed_date_range" and finding.get("status") != "confirmed_error"
                and finding.get("validation") != "paragraph_repetition_requires_review"):
            continue
        anchors = finding.get("verification_spans") or [e for e in finding.get("evidence", []) if e.get("kind") == "source_text"]
        anchors = [{key: s.get(key) for key in ("start", "end", "text")} for s in anchors]
        if not _valid(anchors, content):
            continue
        checks = [deepcopy(e) for e in finding.get("evidence", []) if e.get("kind") == "deterministic"]
        pair = None
        if family == "literal_adjacent_sentence_pair":
            pair = (literal_sentence_pair(content, finding) or _paragraph_sentence_pair(content, finding)
                    or _discourse_prefixed_paragraph_pair(content, finding))
            if pair is None:
                continue
            if pair.get("localization_only"):
                family = "literal_adjacent_paragraph_pair"
            finding.setdefault("source_rule_proofs", [_rule_proof(finding)])
            # Display may need the shared date/subject of a semicolon list;
            # proof, pair members and identity remain narrow and unchanged.
            display_spans = _repeat_display_spans(content, pair)
            if finding.get("spans") != display_spans:
                finding["source_original_spans"] = deepcopy(finding.get("spans"))
                finding["spans"] = display_spans
            finding["paired_source_spans"] = deepcopy(pair["members"])
            checks.append({"check": "unique_literal_sentence_pair", **deepcopy(pair)})
        if detector == "verified.suspended_punctuation":
            family, kind, field = _gap_field(content, anchors[0]["start"])
            if kind != finding["error_type"]:
                finding["source_original_error_type"] = finding["error_type"]
                finding["error_type"] = kind
                finding["reason"] = f"原文明确命名的“{field['field']}”字段后直接结束，字段值为空；不推定应填内容。"
            if family == "missing_unnamed_clause":
                # A comma followed by a terminator proves only a surface gap;
                # it cannot decide missing financial content versus stray comma.
                finding["source_original_verdict"] = {key: finding[key] for key in ("error_type", "status", "validation", "reason")}
                finding.update(status="needs_review", validation="surface_gap_requires_review",
                               reason="原文逗号后直接结束，存在表面空槽；需核对是多余标点还是必要分句缺失，不推定缺失字段。")
                finding["surface_spans"] = finding.pop("verification_spans")
                checks.append({"check": "surface_gap_only", "business_field_missing_confirmed": False})
            checks.append({"check": "source_missing_field_boundary", **field})
        structure = {"family": family, "anchors": anchors, "canonical_type": finding["error_type"],
                     "proof_status": finding["status"], "checks": checks}
        if pair:
            structure["literal_pair"] = deepcopy(pair)
        finding["source_issue_key"] = _key(family, anchors)
        finding["source_structure"] = structure
        issues.append({"key": finding["source_issue_key"], "structure": structure,
                       "compatible_types": compatible_types, "finding": finding})
    # Phrase and sentence parsers can prove the same two members while differing
    # only in the final delimiter. Reuse one structure; do not create a second
    # option that would make a uniquely bound model allegation look ambiguous.
    for finding in findings:
        options = [issue for issue in issues
                   if issue["structure"]["family"] == "literal_adjacent_sentence_pair"
                   and _same_phrase_members(content, finding, issue["structure"]["literal_pair"])]
        if not options:
            pair = _literal_phrase_pair(content, finding)
            if pair is not None:
                family = "literal_adjacent_phrase_pair"
                anchors = [deepcopy(pair["full_span"])]
                finding["source_rule_proofs"] = [_rule_proof(finding)]
                finding["paired_source_spans"] = deepcopy(pair["members"])
                finding["source_issue_key"] = _key(family, anchors)
                finding["source_structure"] = {"family": family, "anchors": anchors,
                    "canonical_type": finding["error_type"], "proof_status": finding["status"],
                    "checks": [{"check": "unique_literal_phrase_pair", **deepcopy(pair)}], "literal_pair": pair}
                issues.append({"key": finding["source_issue_key"], "structure": finding["source_structure"],
                               "compatible_types": {"冗余语句"}, "finding": finding})
            continue
        if len(options) != 1:
            continue
        issue = options[0]
        source = issue["finding"]
        source["source_rule_proofs"].append(_rule_proof(finding))
        finding["source_original_spans"] = deepcopy(finding["spans"])
        finding["source_original_verification_spans"] = deepcopy(finding["verification_spans"])
        for key in ("spans", "verification_spans", "paired_source_spans", "source_structure", "source_rule_proofs"):
            finding[key] = deepcopy(source[key])
        finding["source_issue_key"] = issue["key"]
    return issues


def _describes_family(candidate, issue):
    reason = str(candidate.get("reason", ""))
    anchors = issue["structure"]["anchors"]
    family = issue["structure"]["family"]
    if family in _REPEAT_FAMILIES:
        pair = issue["structure"]["literal_pair"]
        exact = len(candidate.get("spans", [])) == 1 and candidate["spans"][0] == pair["full_span"]
        return _single_repeat_allegation(reason, pair["unit"], require_unit_quote=not exact)
    if family == "missing_date_numeric_slot":
        # An impossible filled date, chronology, or a different missing field
        # must not inherit this date blank's proof through a broad quote.
        if re.search(r"晚于|早于|先于|后于|倒序|逆序|颠倒|先后|前后|起止|冲突|矛盾|不一致|不匹配|"
                     r"不可能|不存在|无效|非法|不合法|超出|超过|应改为|而且|并且|同时|另外|此外|"
                     r"另有|还有|还存在|也存在|又|也缺|还缺", reason):
            return False
        # A reason may restate this incomplete date, but cannot quietly attach
        # a second concrete date assertion under the same blank's proof.
        source_dates = "".join(re.sub(r"\s+", "", a["text"]) for a in anchors)
        mentioned_dates = re.findall(r"(?:19|20)\d{2}年(?:\s*\d{1,2}月(?:\s*\d{1,2}日)?)?", reason)
        if any(re.sub(r"\s+", "", date) not in source_dates for date in mentioned_dates):
            return False
        return bool(re.search(r"日期|年月|月份|日数|几日|几号|具体.{0,3}日|\d{4}年", reason)
                    and re.search(r"缺|遗漏|漏填|空|不完整|未(?:填|给|列|说明|提供)", reason)
                    and not re.search(r"收入|利润|评级|名称|证券代码|市盈率|每股收益", reason))
    numeric_fields = re.findall(r"销售额均值|营业收入|销售额|收入|归母净利润|净利润|毛利率|净利率|增长率|数量|均价|每股收益|市盈率", reason)
    if family == "empty_attribute" and numeric_fields:
        return False
    exact_token = len(candidate.get("spans", [])) == len(anchors) and {
        (s["start"], s["end"]) for s in candidate["spans"]} == {(s["start"], s["end"]) for s in anchors}
    if exact_token:
        return True
    # A model allegation about another explicitly named field must not inherit
    # this structure's proof merely because its full-sentence quote overlaps.
    field_names = (re.findall(r"营业收入|销售额|收入|归母净利润|净利润", reason)
                   if family == "aligned_sequence_cardinality" else
                   numeric_fields + re.findall(r"信用评级|评级|证券代码", reason)
                   if family in {"empty_attribute", "missing_numeric_slot", "missing_unnamed_clause"} else [])
    source_text = "".join(s["text"] for s in issue["finding"].get("spans", []))
    if field_names and any(name not in source_text for name in field_names):
        return False
    if family in {"empty_attribute", "missing_numeric_slot", "missing_unnamed_clause"}:
        return bool(re.search(r"缺|遗漏|漏填|空|未(?:填|给|列|说明|提供)", reason))
    if family == "aligned_sequence_cardinality":
        return bool(re.search(r"数量|个数|项数|序列|对应|依次|分别", reason)
                    and re.search(r"不一致|不等|不匹配|不符|多[出于了]|缺|列出|[二三四五六七八九十\d]+[个项]", reason))
    if family == "rank_within_explicit_total":
        return bool(re.search(r"排名|位居|位列|名次|第.+[名位]", reason))
    if family == "reversed_legal_date_range":
        return bool(re.search(r"倒|起止|先后|区间|期间|开始|起始", reason)
                    or any(s["text"] in reason for s in anchors))
    return False


def align_source_issue(candidate, issues, content):
    """Resolve exactly one alleged source issue; ambiguous candidates stay raw."""
    spans = candidate.get("spans", [])
    if not _valid(spans, content):
        return False
    if candidate.get("error_type") == "冗余语句":
        covered_pairs = [issue for issue in issues if issue["structure"]["family"] in _REPEAT_FAMILIES
                         and all(any(s["start"] <= a["start"] and a["end"] <= s["end"] for s in spans)
                                 for a in issue["structure"]["anchors"])]
        # An exact proved outer pair can still contain a separate repeated
        # phrase. Binding that exact pair must not swallow or duplicate the
        # inner issue; broad quotes over two unrelated pairs remain ambiguous.
        exact_pair = [issue for issue in covered_pairs if spans == issue["structure"]["anchors"]
                      and _describes_family(candidate, issue)]
        if len({issue["key"] for issue in covered_pairs}) > 1 and len(exact_pair) != 1:
            candidate["source_alignment"] = {"status": "ambiguous_source_structures",
                                             "candidate_issue_keys": [i["key"] for i in covered_pairs]}
            return False
    options = []
    for issue in issues:
        if candidate.get("error_type") not in issue["compatible_types"] or not _describes_family(candidate, issue):
            continue
        if all(any(s["start"] <= a["start"] and a["end"] <= s["end"] for s in spans)
               for a in issue["structure"]["anchors"]):
            options.append(issue)
    if len(options) > 1:
        # Repeated empty tokens cannot distinguish two slots. A unique explicit
        # source quote can select one different token without guessing an offset.
        explicit = [issue for issue in options if all(a["text"] in str(candidate.get("reason", ""))
                    for a in issue["structure"]["anchors"])]
        if len(explicit) == 1:
            options = explicit
    if len(options) != 1:
        if options:
            candidate["source_alignment"] = {"status": "ambiguous_source_structures", "candidate_issue_keys": [i["key"] for i in options]}
        return False
    issue = options[0]
    source, structure = issue["finding"], issue["structure"]
    candidate["source_alignment"] = {"status": "unique_source_structure", "original_error_type": candidate["error_type"],
                                     "original_reason": candidate.get("reason", ""), "source_issue_key": issue["key"]}
    candidate.update(error_type=structure["canonical_type"], reason=source["reason"], status=structure["proof_status"],
                     source_issue_key=issue["key"], source_structure=deepcopy(structure),
                     spans=deepcopy(source["spans"]), evidence=deepcopy(source["evidence"]),
                     validation="deterministic" if structure["proof_status"] == "confirmed_error" else "source_structure_needs_review")
    if source.get("verification_spans"):
        candidate["verification_spans"] = deepcopy(source["verification_spans"])
        candidate["verified_by"] = source["detector_id"]
        candidate["confirmation"] = "unique_source_structure"
    return True


def merge_represented_candidates(old, new):
    """Preserve each raw occurrence even when one issue represents several."""
    represented, seen = [], set()
    for value in old.get("represented_model_candidates", []) + new.get("represented_model_candidates", []):
        key = value["candidate_ref"]
        if key not in seen:
            seen.add(key)
            represented.append(deepcopy(value))
    return represented
