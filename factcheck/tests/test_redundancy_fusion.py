"""Source-only repeat localization and identity; synthetic text, no benchmark gold."""
from copy import deepcopy
import json

import pytest

from yjcheck.review_hints import all_review_hints
from yjcheck.text_fusion import anchor_repeated_sentence_pair, build_source_issues
from yjcheck.text_review import _redundant_duplicate_rules, detect_text


UNIT = "公司主要提供工业自动控制设备，并面向海内外客户销售。"
PRICE = "港口甲动力煤库提价580元/吨，周环比下跌8元/吨；"
OTHER = "港口乙动力煤库提价640元/吨，周环比持平。"


def candidate(spans, reason="同一完整句子连续重复出现，未新增信息。"):
    return {"error_type": "冗余语句", "spans": spans, "reason": reason}


def pair_candidate(unit=UNIT, **extra):
    return candidate([{"text": unit, **extra}, {"text": unit, **extra}])


def run(text, candidates, *, detector="hybrid", normalize_spans=True):
    def chat(messages, *, purpose):
        return {"content": json.dumps({"errors": candidates}, ensure_ascii=False), "trace": {}}
    return detect_text(text, document_id="synthetic-repeat-only", detector=detector, chat=chat,
                       normalize_spans=normalize_spans)


def repeats(report):
    return [e for e in report["errors"] if e["error_type"] == "冗余语句"]


def recover(text, raw, ranges=None, proofs=None):
    return anchor_repeated_sentence_pair(raw, text, ranges if ranges is not None else [(0, len(text))],
                                         _redundant_duplicate_rules(text) if proofs is None else proofs)


@pytest.mark.parametrize("gap", ["", " ", "\t"])
def test_exact_two_missing_offsets_are_bound_to_a_unique_source_pair(gap):
    text = UNIT + gap + UNIT
    raw = pair_candidate()
    original = deepcopy(raw)
    report = run(text, [raw])
    issues = repeats(report)
    assert len(issues) == len(all_review_hints(report)["errors"]) == 1
    issue = issues[0]
    assert issue["status"] == "confirmed_error" and issue["source_structure"]["family"] == "literal_adjacent_sentence_pair"
    assert issue["spans"] == [{"start": 0, "end": len(text), "text": text}]
    assert issue["paired_source_spans"] == [
        {"start": 0, "end": len(UNIT), "text": UNIT},
        {"start": len(UNIT) + len(gap), "end": len(text), "text": UNIT}]
    assert report["raw_candidates"][0]["candidate"] == original
    representation = issue["represented_model_candidates"][0]
    assert representation["candidate"] == original
    assert representation["source_anchor_proof"]["anchor_method"] == "unique_adjacent_sentence_pair"
    assert representation["source_anchor_proof"]["members"] == issue["paired_source_spans"]
    assert report["model_candidate_representation"][0]["error_id"] == issue["id"]
    assert not report["rejected_candidates"] and report["coverage"]["candidate_quality_complete"]


def test_model_direct_shares_localization_but_does_not_inherit_confirmation():
    text = UNIT * 2
    report = run(text, [pair_candidate()], detector="model_direct")
    assert len(repeats(report)) == len(all_review_hints(report)["errors"]) == 1
    issue = repeats(report)[0]
    assert issue["status"] == "needs_review" and "verification_spans" not in issue
    assert issue["anchor_method"] == "unique_adjacent_sentence_pair"
    assert len(issue["paired_source_spans"]) == 2 and issue["spans"][0]["text"] == text
    assert not report["rejected_candidates"]


@pytest.mark.parametrize("text", [UNIT * 3, UNIT * 2 + UNIT[:-1], UNIT + "另有不同业务安排。" + UNIT])
def test_three_occurrences_and_separated_quotes_abstain(text):
    assert recover(text, pair_candidate()) is None
    report = run(text, [pair_candidate()])
    assert len(report["rejected_candidates"]) == 1
    assert any(e.get("invalid_anchor") for e in all_review_hints(report)["errors"])
    assert not any(e.get("represented_model_candidates") for e in repeats(report))


def test_two_visible_occurrences_do_not_hide_a_third_in_the_full_document():
    text = UNIT * 3
    assert recover(text, pair_candidate(), ranges=[(0, len(UNIT) * 2)]) is None


@pytest.mark.parametrize("ranges", [[(0, len(UNIT))], [(len(UNIT), len(UNIT) * 2)],
                                      [(0, len(UNIT) - 1), (len(UNIT), len(UNIT) * 2)]])
def test_both_members_must_be_fully_model_visible(ranges):
    assert recover(UNIT * 2, pair_candidate(), ranges=ranges) is None


def test_overlapping_visible_contexts_do_not_duplicate_source_occurrences():
    text = UNIT * 2
    pair = recover(text, pair_candidate(), ranges=[(0, len(text)), (0, len(text))])
    assert pair and pair["source_occurrence_count"] == 2


@pytest.mark.parametrize("raw", [candidate([{"text": UNIT}]),
                                   candidate([{"text": UNIT}] * 3),
                                   pair_candidate(UNIT[:-1]), pair_candidate(start=None),
                                   pair_candidate(start=0, end=len(UNIT))])
def test_single_incomplete_members_and_explicit_offsets_do_not_enter_recovery(raw):
    assert recover(UNIT * 2, raw) is None


def test_existing_valid_explicit_offsets_keep_the_original_protocol():
    text = UNIT * 2
    spans = [{"start": 0, "end": len(UNIT), "text": UNIT},
             {"start": len(UNIT), "end": len(text), "text": UNIT}]
    report = run(text, [candidate(spans)], detector="model_direct", normalize_spans=False)
    assert report["errors"][0]["spans"] == spans
    assert "paired_source_spans" not in report["errors"][0]
    assert not report["rejected_candidates"]


def test_different_periods_are_distinct_facts_not_a_literal_pair():
    one, two = "公司2024年主营收入稳定增长。", "公司2025年主营收入稳定增长。"
    raw = candidate([{"text": one}, {"text": two}])
    report = run(one + two, [raw])
    assert len(repeats(report)) == 1 and repeats(report)[0]["status"] == "needs_review"
    assert "source_issue_key" not in repeats(report)[0]


def test_recovery_requires_existing_confirmed_independent_proof():
    text = UNIT * 2
    assert recover(text, pair_candidate(), proofs=[]) is None
    proofs = _redundant_duplicate_rules(text)
    for proof in proofs:
        proof["status"] = "needs_review"
    assert recover(text, pair_candidate(), proofs=proofs) is None
    assert recover("错误示例：" + text, pair_candidate()) is None


def test_wide_quote_selects_only_an_explicit_unique_repeated_source_unit():
    text = "截至4月30日，" + PRICE * 2 + OTHER
    raw = candidate([{"text": text}], f"“{PRICE[:-1]}”连续重复，未新增信息。")
    report = run(text, [raw])
    issues = repeats(report)
    assert len(issues) == len(all_review_hints(report)["errors"]) == 1
    issue = issues[0]
    assert issue["status"] == "confirmed_error"
    assert issue["spans"][0]["text"] == text
    assert issue["verification_spans"][0]["text"] == PRICE * 2
    assert OTHER not in issue["verification_spans"][0]["text"]
    assert issue["represented_model_candidates"][0]["candidate"] == raw
    assert issue["source_alignments"][0]["original_reason"] == raw["reason"]
    assert len(report["model_candidate_representation"]) == 1


@pytest.mark.parametrize("reason", [
    "本段连续重复，没有新增信息。",
    f"“{PRICE[:-1]}”重复，港口乙动力煤价格也错误。",
    f"“{PRICE[:-1]}”重复，而且港口乙的描述也重复。",
    f"“{PRICE[:-1]}”与“{OTHER[:-1]}”都重复。",
    "“动力煤价格”连续重复，未新增信息。",
    f"“{PRICE[:-1]}”没有重复。",
])
def test_wide_quote_without_unique_exact_allegation_does_not_swallow_other_claims(reason):
    text = "截至4月30日，" + PRICE * 2 + OTHER
    raw = candidate([{"text": text}], reason)
    report = run(text, [raw])
    issues = repeats(report)
    assert len(issues) == len(all_review_hints(report)["errors"]) == 2
    model = next(e for e in issues if e["detector_id"] == "hybrid.model")
    assert model["status"] == "needs_review" and model["reason"] == reason
    assert "source_issue_key" not in model


def test_wide_quote_with_two_proved_repeat_issues_remains_unmerged():
    text = PRICE * 2 + UNIT * 2
    raw = candidate([{"text": text}], f"“{PRICE[:-1]}”连续重复，未新增信息。")
    report = run(text, [raw])
    assert len(repeats(report)) == 3
    model = next(e for e in repeats(report) if e["detector_id"] == "hybrid.model")
    assert model["source_alignment"]["status"] == "ambiguous_source_structures"
    assert model["status"] == "needs_review"


def test_distinct_model_redundancy_outside_the_pair_is_retained():
    text = "截至4月30日，" + PRICE * 2 + "另有产品产品预计投产。"
    selected = candidate([{"text": text}], f"“{PRICE[:-1]}”连续重复，未新增信息。")
    other = candidate([{"text": text}], "产品产品包含独立重复词。")
    report = run(text, [selected, other])
    issues = repeats(report)
    assert len(issues) == 2
    assert {m["candidate_ref"] for e in issues for m in e.get("represented_model_candidates", [])} == {"model:0:0", "model:0:1"}
    assert len({m["error_id"] for m in report["model_candidate_representation"]}) == 2
    assert any(e["reason"] == other["reason"] and e["status"] == "needs_review" for e in issues)


def test_recovered_hint_does_not_hide_an_unrelated_rejected_hint():
    text = UNIT * 2
    report = run(text, [pair_candidate(), candidate([{"text": "原文根本没有的引文。"}])])
    assert len(repeats(report)) == 1 and len(report["rejected_candidates"]) == 1
    hints = all_review_hints(report)["errors"]
    assert len(hints) == 2 and sum(bool(e.get("invalid_anchor")) for e in hints) == 1


@pytest.mark.parametrize("prefix,suffix", [
    ("前一公司收入增长。", "下一公司收入下降。"),
    ("前一公司收入增长。\n", "\n下一公司收入下降。"),
    ("第一项目：其他产品价格持平。", "第三项目：其他产品价格下降。"),
])
def test_semicolon_context_keeps_only_one_closed_statement(prefix, suffix):
    statement = "第二项目：截至4月30日，" + PRICE * 2 + OTHER
    text = prefix + statement + suffix
    issue, = repeats(run(text, []))
    assert issue["spans"] == [{"start": len(prefix), "end": len(prefix + statement), "text": statement}]
    assert issue["verification_spans"][0]["text"] == PRICE * 2
    assert all(OTHER not in span["text"] for span in issue["paired_source_spans"])


@pytest.mark.parametrize("text", ["甲" * 801 + "，" + PRICE * 2 + OTHER,
                                 "截至4月30日，" + PRICE * 2 + OTHER[:-1]])
def test_long_or_unfinished_statement_does_not_expand_repeat_context(text):
    issue, = repeats(run(text, []))
    assert issue["spans"] == issue["verification_spans"]
    assert issue["spans"][0]["text"] == PRICE * 2


def test_two_full_sentences_never_absorb_neighbor_context():
    text = "公司此前安排其他业务。" + UNIT * 2 + "公司随后开发新产品。"
    issue, = repeats(run(text, [pair_candidate()]))
    assert issue["spans"] == issue["verification_spans"]
    assert issue["spans"][0]["text"] == UNIT * 2


RISK = "主要原料价格大幅波动"


def test_phrase_sentence_model_share_exact_two_members_and_keep_every_proof():
    text = "风险提示：" + (RISK + "；") * 2 + "汇率波动；需求下降。"
    raw = candidate([{"text": text}], f"“{RISK}”连续重复，同一风险不必要地出现两次。")
    report = run(text, [raw])
    issue, = repeats(report)
    assert len(all_review_hints(report)["errors"]) == 1
    assert issue["spans"][0]["text"] == text
    assert issue["verification_spans"][0]["text"] == (RISK + "；") * 2
    assert issue["detector_ids"] == ["hybrid.model", "verified.redundant_phrase", "verified.redundant_sentence"]
    assert {proof["detector_id"] for proof in issue["source_rule_proofs"]} == {
        "verified.redundant_phrase", "verified.redundant_sentence"}
    original_phrase = next(proof for proof in issue["source_rule_proofs"] if proof["detector_id"].endswith("phrase"))
    assert original_phrase["verification_spans"][0]["text"] == RISK + "；" + RISK
    assert issue["represented_model_candidates"][0]["candidate"] == raw
    assert len(report["model_candidate_representation"]) == 1
    assert not report["rejected_candidates"]
    findings = _redundant_duplicate_rules(text)
    issues = build_source_issues(text, findings)
    assert len(issues) == 1
    assert len({f["source_issue_key"] for f in findings}) == 1


def test_two_repeat_pairs_with_same_display_context_remain_two_issues():
    another = "国际汇率出现大幅波动"
    text = "风险提示：" + (RISK + "；") * 2 + (another + "；") * 2 + "需求下降。"
    report = run(text, [])
    issues = repeats(report)
    assert len(issues) == len(all_review_hints(report)["errors"]) == 2
    assert len({e["source_issue_key"] for e in issues}) == 2
    assert all(e["spans"][0]["text"] == text for e in issues)
    assert {e["verification_spans"][0]["text"] for e in issues} == {(RISK + "；") * 2, (another + "；") * 2}


def test_nested_phrase_inside_a_repeated_sentence_is_an_independent_issue():
    sentence = "公司产品包括：工业设备、工业设备，预计明年投产。"
    report = run(sentence * 2, [])
    issues = repeats(report)
    assert len(issues) == len(all_review_hints(report)["errors"]) == 3
    assert sum(bool(e.get("source_issue_key")) for e in issues) == 1
    assert sum(e["verification_spans"][0]["text"] == "工业设备、工业设备" for e in issues) == 2


def test_phrase_only_proof_does_not_acquire_sentence_identity():
    text = "风险提示：材料波动、材料波动、需求下降。"
    issue, = repeats(run(text, []))
    assert issue["detector_id"] == "verified.redundant_phrase"
    assert issue["source_structure"]["family"] == "literal_adjacent_phrase_pair"
    assert issue["paired_source_spans"][0]["text"] == "材料波动"


@pytest.mark.parametrize("detector", ["hybrid", "model_direct"])
def test_finite_same_subject_reason_restores_pair_localization(detector):
    raw = pair_candidate()
    raw["reason"] = "同一主体、时点、产能状态的原句连续重复两次，未新增信息。"
    report = run(UNIT * 2, [raw], detector=detector)
    issue, = repeats(report)
    assert not report["rejected_candidates"] and report["coverage"]["candidate_quality_complete"]
    assert issue["represented_model_candidates"][0]["source_anchor_proof"]["members"] == issue["paired_source_spans"]
    assert issue["status"] == ("confirmed_error" if detector == "hybrid" else "needs_review")
    assert ("verification_spans" in issue) is (detector == "hybrid")
    assert len(all_review_hints(report)["errors"]) == 1


@pytest.mark.parametrize("reason", [
    "同一主体、时点、产能状态的原句连续重复两次，而且开工晚于竣工。",
    "不同主体、时点、产能状态的原句连续重复两次，未新增信息。",
    "同一主体、时点、价格错误的原句连续重复两次，未新增信息。",
    "同一主体、时点、产能状态的原句没有重复。",
])
def test_descriptor_reason_does_not_admit_extra_or_negated_allegations(reason):
    raw = pair_candidate()
    raw["reason"] = reason
    report = run(UNIT * 2, [raw])
    assert len(report["rejected_candidates"]) == 1
    assert len(all_review_hints(report)["errors"]) == 2
    assert not repeats(report)[0].get("represented_model_candidates")


def test_fused_risk_pair_keeps_other_risk_allegation_and_unanchored_hint():
    text = "风险提示：" + (RISK + "；") * 2 + "汇率波动；需求下降。"
    mixed = candidate([{"text": text}], f"“{RISK}”连续重复，同一风险不必要地出现两次，而且汇率波动也重复。")
    invalid = candidate([{"text": "原文不存在的风险提示。"}])
    report = run(text, [mixed, invalid])
    assert len(repeats(report)) == 2
    model = next(e for e in repeats(report) if e["detector_id"] == "hybrid.model")
    assert model["reason"] == mixed["reason"] and model["status"] == "needs_review"
    assert "source_issue_key" not in model
    assert len(report["rejected_candidates"]) == 1
    assert len(all_review_hints(report)["errors"]) == 3


@pytest.mark.parametrize("gap", ["\n", "\n\n", "\r\n", "\r\n\r\n"])
@pytest.mark.parametrize("detector", ["hybrid", "model_direct"])
@pytest.mark.parametrize("reason", [
    "同一句子在原文中连续重复出现，无新增信息。",
    "同一句话在相邻段落中完全重复，未提供新增信息，属于冗余语句。",
])
def test_whole_adjacent_paragraphs_restore_location_but_never_auto_confirm(gap, detector, reason):
    text = "背景介绍。\n\n" + UNIT + gap + UNIT + "\n\n后续新增业务。"
    raw = pair_candidate()
    raw["reason"] = reason
    report = run(text, [raw], detector=detector)
    issue, = repeats(report)
    assert issue["status"] == "needs_review" and "verification_spans" not in issue
    assert issue["spans"][0]["text"] == UNIT + gap + UNIT
    assert len(all_review_hints(report)["errors"]) == 1 and not report["rejected_candidates"]
    represented = issue["represented_model_candidates"][0]
    assert represented["candidate"] == raw
    assert represented["source_anchor_proof"]["localization_only"] is True
    assert represented["source_anchor_proof"]["anchor_method"] == "unique_adjacent_paragraph_pair"
    assert all(member["text"] == UNIT for member in represented["source_anchor_proof"]["members"])
    if detector == "hybrid":
        assert issue["source_structure"]["proof_status"] == "needs_review"
        assert issue["source_original_verdict"]["status"] == "confirmed_error"
        assert issue["surface_spans"][0]["text"] == UNIT + gap + UNIT


@pytest.mark.parametrize("prefix", ["示例：\n", "摘要：\n", "正文标题：\n", "错误示例：\n"])
def test_paragraph_role_labels_do_not_turn_identical_passages_into_confirmed_errors(prefix):
    report = run(prefix + UNIT + "\n\n" + UNIT, [pair_candidate()])
    assert all(e["status"] == "needs_review" for e in repeats(report))
    assert all("verification_spans" not in e for e in repeats(report))


@pytest.mark.parametrize("text", [
    UNIT + "\n\n" + UNIT + "另外扩建一条生产线。",
    UNIT + "\n\n" + UNIT + "\n\n" + UNIT,
    UNIT + "\n\n正文：\n" + UNIT,
    "摘要：" + UNIT + "\n\n" + UNIT,
])
@pytest.mark.parametrize("reason", [
    "同一完整句子连续重复出现，未新增信息。",
    "同一句话在相邻段落中完全重复，未提供新增信息，属于冗余语句。",
])
def test_paragraph_new_information_extra_roles_or_third_member_do_not_recover(text, reason):
    raw = pair_candidate()
    raw["reason"] = reason
    report = run(text, [raw])
    assert report["rejected_candidates"]
    assert not any(e.get("represented_model_candidates") for e in repeats(report))
    assert not any(e["status"] == "confirmed_error" for e in repeats(report))


@pytest.mark.parametrize("reason", [
    "同一句子在原文中连续重复出现，而且开工晚于竣工。",
    "同一句话在相邻段落中完全重复，未提供新增信息，而且开工晚于竣工。",
    "同一句话在相邻段落中完全重复，未提供新增信息，而且收入单位也错误。",
    "同一句话在相邻段落中没有完全重复，提供了新增信息。",
    "同一句话在不同段落角色中完全重复，未提供新增信息。",
])
def test_paragraph_mixed_reason_is_preserved_as_unrepresented_hint(reason):
    raw = pair_candidate()
    raw["reason"] = reason
    report = run(UNIT + "\n\n" + UNIT, [raw])
    assert len(report["rejected_candidates"]) == 1
    hints = all_review_hints(report)["errors"]
    assert len(hints) == 2 and any(e["reason"] == raw["reason"] for e in hints)
    assert all(e["status"] == "needs_review" for e in hints)


def prefixed_candidate(reason=None, **extra):
    return candidate([{"text": "本周订单增长；" + UNIT, **extra}, {"text": UNIT, **extra}],
                     reason or f"“{UNIT[:-1]}”同一信息连续重复。")


@pytest.mark.parametrize("detector", ["hybrid", "model_direct"])
def test_unequal_quotes_can_bind_only_a_unique_pair_of_proved_members(detector):
    text = "本周订单增长；" + UNIT * 2
    raw = prefixed_candidate()
    report = run(text, [raw], detector=detector)
    issue, = repeats(report)
    assert issue["status"] == ("confirmed_error" if detector == "hybrid" else "needs_review")
    assert issue["spans"][0]["text"] == UNIT * 2
    representation = issue["represented_model_candidates"][0]
    assert representation["candidate"] == raw
    proof = representation["source_anchor_proof"]
    assert proof["anchor_method"] == "unique_member_covering_quotes"
    assert [span["text"] for span in proof["quoted_source_spans"]] == [s["text"] for s in raw["spans"]]
    assert [span["member_index"] for span in proof["quoted_source_spans"]] == [0, 1]
    assert len(all_review_hints(report)["errors"]) == 1 and not report["rejected_candidates"]


@pytest.mark.parametrize("raw", [
    prefixed_candidate("本段连续重复，没有新增信息。"),
    prefixed_candidate(f"“{UNIT[:-1]}”重复，而且本周订单增长也不正确。"),
    prefixed_candidate(f"“{UNIT[:-1]}”没有重复。"),
    prefixed_candidate(start=None),
    candidate([{"text": "本周订单增长；" + UNIT}, {"text": UNIT[:-1]}], f"“{UNIT[:-1]}”连续重复。"),
])
def test_unequal_quote_recovery_does_not_guess_or_absorb_other_allegations(raw):
    assert recover("本周订单增长；" + UNIT * 2, raw) is None


def test_member_prefix_cannot_extend_to_previous_sentence_or_across_paragraph():
    raw = candidate([{"text": "本周订单增长。" + UNIT}, {"text": UNIT}], f"“{UNIT[:-1]}”连续重复。")
    assert recover("本周订单增长。" + UNIT * 2, raw) is None
    assert recover("本周订单增长；" + UNIT + "\n\n" + UNIT, prefixed_candidate()) is None


def test_second_sentence_new_information_is_not_a_same_member_pair():
    different = UNIT[:-1] + "，预计明年扩建。"
    raw = candidate([{"text": "本周订单增长；" + UNIT}, {"text": different}], f"“{UNIT[:-1]}”连续重复。")
    report = run("本周订单增长；" + UNIT + different, [raw])
    model = next(e for e in repeats(report) if e["detector_id"] == "hybrid.model")
    assert model["status"] == "needs_review" and "source_issue_key" not in model
    assert model["represented_model_candidates"][0]["candidate"] == raw


def test_unique_comma_phrase_proof_binds_model_but_does_not_change_type():
    unit = "构建起核心产品壁垒"
    text = "公司持续进行技术研发，满足客户定制化\n\n需求，" + unit + "，" + unit + "，保持稳定经营。"
    raw = candidate([{"text": text}], f"“{unit}”连续重复出现，未增加新信息。")
    report = run(text, [raw])
    issue, = repeats(report)
    assert issue["error_type"] == "冗余语句" and issue["source_structure"]["family"] == "literal_adjacent_phrase_pair"
    assert issue["verification_spans"][0]["text"] == unit + "，" + unit
    assert [m["text"] for m in issue["paired_source_spans"]] == [unit, unit]
    assert issue["spans"][0]["text"].startswith("需求，")
    assert issue["represented_model_candidates"][0]["candidate"] == raw
    assert len(all_review_hints(report)["errors"]) == 1


def test_phrase_identity_does_not_absorb_another_allegation_or_suffix_expansion():
    unit = "构建起核心产品壁垒"
    text = "公司计划：" + unit + "，" + unit + "，并预计明年扩建。"
    mixed = candidate([{"text": text}], f"“{unit}”重复，而且明年扩建时间也有错误。")
    report = run(text, [mixed])
    assert len(repeats(report)) == 2
    assert any(e["reason"] == mixed["reason"] and e["status"] == "needs_review" for e in repeats(report))
    expanded = "公司计划：" + unit + "，" + unit + "并形成第二项业务。"
    raw = candidate([{"text": expanded}], f"“{unit}”连续重复。")
    findings = _redundant_duplicate_rules(expanded)
    assert not build_source_issues(expanded, findings)
    assert not any(f.get("source_issue_key") for f in findings)


def test_two_comma_repeat_pairs_in_one_quote_remain_ambiguous():
    one, two = "构建起核心产品壁垒", "持续拓展海外客户渠道"
    text = "公司计划：" + one + "，" + one + "，" + two + "，" + two + "。"
    raw = candidate([{"text": text}], f"“{one}”连续重复。")
    report = run(text, [raw])
    assert len(repeats(report)) == 3
    model = next(e for e in repeats(report) if e["detector_id"] == "hybrid.model")
    assert model["source_alignment"]["status"] == "ambiguous_source_structures"
    assert model["status"] == "needs_review"


CAUSE_UNIT = "生产费用下降主要系设备维护支出减少所致。"


@pytest.mark.parametrize("detector", ["hybrid", "model_direct"])
def test_discourse_prefixed_adjacent_members_restore_only_source_location(detector):
    text = "前文经营情况。" + "其中" + CAUSE_UNIT + "\n" + CAUSE_UNIT + "公司新增业务预计明年投产。"
    raw = candidate([{"text": "其中" + CAUSE_UNIT}, {"text": CAUSE_UNIT}],
                    "同一生产费用下降原因连续重复，未提供新增信息。")
    other = {"error_type": "时间矛盾", "spans": [{"text": text}], "reason": "新增业务的投产期间需核实。"}
    report = run(text, [raw, other], detector=detector)
    issue, = repeats(report)
    assert issue["status"] == "needs_review" and "verification_spans" not in issue
    assert issue["spans"][0]["text"] == "其中" + CAUSE_UNIT + "\n" + CAUSE_UNIT
    representation, = issue["represented_model_candidates"]
    assert representation["candidate"] == raw
    proof = representation["source_anchor_proof"]
    assert proof["localization_only"] is True and proof["discourse_prefix"]["text"] == "其中"
    assert proof["source_member_contexts"][0]["text"] == "其中" + CAUSE_UNIT
    assert [q["member_index"] for q in proof["quoted_source_spans"]] == [0, 1]
    assert not report["rejected_candidates"]
    assert len(all_review_hints(report)["errors"]) == 2
    assert next(e for e in report["errors"] if e["error_type"] == "时间矛盾")["reason"] == other["reason"]


@pytest.mark.parametrize("gap", ["", "\n\n"])
@pytest.mark.parametrize("detector", ["hybrid", "model_direct"])
@pytest.mark.parametrize("reverse", [False, True])
def test_wide_quote_covering_pair_keeps_narrow_quote_joint_without_guessing_offset(gap, detector, reverse):
    text = UNIT + gap + UNIT
    quotes = [{"text": text}, {"text": UNIT}]
    raw = candidate(quotes[::-1] if reverse else quotes, f"“{UNIT[:-1]}”连续重复，未提供新增信息。")
    report = run(text, [raw], detector=detector)
    issue, = repeats(report)
    proof = issue["represented_model_candidates"][0]["source_anchor_proof"]
    assert proof["anchor_method"] == "unique_pair_covering_quote"
    assert proof["member_assignment"] == "joint_pair_not_individual_offset"
    narrow = next(q for q in proof["quoted_source_matches"] if q["text"] == UNIT)
    assert narrow["source_matches"] == issue["paired_source_spans"]
    assert proof["wide_quote_index"] == (1 if reverse else 0)
    assert not report["rejected_candidates"] and len(all_review_hints(report)["errors"]) == 1
    assert issue["status"] == ("confirmed_error" if detector == "hybrid" and not gap else "needs_review")


def test_same_statement_prefix_may_cover_pair_but_next_sentence_may_not():
    text = "本周订单增长；" + UNIT * 2
    raw = candidate([{"text": text}, {"text": UNIT}], f"“{UNIT[:-1]}”连续重复。")
    assert recover(text, raw)["anchor_method"] == "unique_pair_covering_quote"
    extended = text + "另有新业务投入。"
    raw["spans"][0]["text"] = extended
    assert recover(extended, raw) is None


@pytest.mark.parametrize("prefix", ["公司", "摘要：", "正文：", "例如", "2025年"])
def test_arbitrary_prefix_is_not_a_discourse_equivalence(prefix):
    text = prefix + CAUSE_UNIT + "\n" + CAUSE_UNIT
    raw = candidate([{"text": prefix + CAUSE_UNIT}, {"text": CAUSE_UNIT}],
                    "同一生产费用下降原因连续重复，未提供新增信息。")
    assert recover(text, raw) is None


@pytest.mark.parametrize("reason", [
    "同一生产费用下降原因连续重复，而且利润单位有错。",
    "同一生产费用下降原因连续重复，而且开工晚于竣工。",
    "同一研发费用下降原因连续重复，未提供新增信息。",
    "同一生产费用下降原因没有重复。",
])
def test_source_cause_description_keeps_mixed_wrong_subject_and_negated_claims(reason):
    text = "其中" + CAUSE_UNIT + "\n" + CAUSE_UNIT
    raw = candidate([{"text": "其中" + CAUSE_UNIT}, {"text": CAUSE_UNIT}], reason)
    report = run(text, [raw])
    assert len(report["rejected_candidates"]) == 1
    assert len(all_review_hints(report)["errors"]) == 2
    assert all(e["status"] == "needs_review" for e in all_review_hints(report)["errors"])


@pytest.mark.parametrize("source", [
    "其中" + CAUSE_UNIT + "\n" + CAUSE_UNIT + "\n" + CAUSE_UNIT,
    "其中" + CAUSE_UNIT + "\n" + CAUSE_UNIT + CAUSE_UNIT[:-1],
    "其中" + CAUSE_UNIT + "\n正文：" + CAUSE_UNIT,
    "其中" + CAUSE_UNIT + "\n" + CAUSE_UNIT[:-1] + "，并实现收入增长。",
])
def test_nested_member_quotes_never_select_between_third_occurrences_or_changed_members(source):
    raw = candidate([{"text": "其中" + CAUSE_UNIT}, {"text": CAUSE_UNIT}],
                    "同一生产费用下降原因连续重复，未提供新增信息。")
    assert recover(source, raw) is None


@pytest.mark.parametrize("suffix", [UNIT, UNIT[:-1]])
def test_wide_pair_quote_does_not_hide_an_unseen_third_occurrence(suffix):
    text = UNIT * 2 + suffix
    raw = candidate([{"text": UNIT * 2}, {"text": UNIT}], f"“{UNIT[:-1]}”连续重复。")
    assert recover(text, raw, ranges=[(0, len(UNIT) * 2)]) is None
