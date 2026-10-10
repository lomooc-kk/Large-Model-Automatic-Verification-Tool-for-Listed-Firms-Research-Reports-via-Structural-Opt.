"""Complete source members, never a shared prefix/suffix presented as a pair."""
import pytest

from yjcheck.text_review import _redundant_duplicate_rules, detect_text


UNIT = "公司本期经营承压"


def sentence_findings(text):
    return [finding for finding in _redundant_duplicate_rules(text)
            if finding["detector_id"] == "verified.redundant_sentence"]


@pytest.mark.parametrize("gap", ["", " ", "\n", "\r\n", "\n\n"])
@pytest.mark.parametrize("continuation", [
    "，原因是新增关税导致出口下降。", ",但新业务利润增长。", "并新增业务风险。", "：主要由于费用增加。",
])
def test_second_member_prefix_with_new_information_is_not_a_full_duplicate(gap, continuation):
    text = UNIT + "。" + gap + UNIT + continuation
    assert not sentence_findings(text)
    report = detect_text(text, document_id="synthetic-prefix")
    assert not any(e["error_type"] == "冗余语句" for e in report["errors"])


@pytest.mark.parametrize("terminator", ["。", "！", "？", "；"])
@pytest.mark.parametrize("gap", ["", " ", "\t", "\n", "\r\n"])
def test_two_full_members_keep_exact_narrow_proof(terminator, gap):
    text = UNIT + terminator + gap + UNIT + terminator
    finding, = sentence_findings(text)
    assert finding["verification_spans"] == [{"start": 0, "end": len(text), "text": text}]


@pytest.mark.parametrize("tail", ["", " ", "\t", "\r\n", " \n\t"])
def test_unpunctuated_second_member_is_supported_only_at_document_end(tail):
    text = UNIT + "。" + UNIT + tail
    finding, = sentence_findings(text)
    assert finding["verification_spans"][0]["text"] == UNIT + "。" + UNIT


@pytest.mark.parametrize("tail", ["\n后续新增业务。", "\n\n另一条完整事实。", "\t另一信息。"])
def test_line_boundary_is_not_silently_treated_as_document_end(tail):
    assert not sentence_findings(UNIT + "。" + UNIT + tail)


@pytest.mark.parametrize("prefix", ["甲", "2024年", "集团旗下", "别的公司认为", "A", "1", "🙂"])
def test_first_member_cannot_start_inside_a_longer_sentence(prefix):
    assert not sentence_findings(prefix + UNIT + "。" + UNIT + "。")


@pytest.mark.parametrize("prefix", ["截至4月30日，", "风险提示：", "其他完整句。", "其他段。\n", "\t"])
def test_existing_shared_context_or_explicit_boundary_is_preserved(prefix):
    text = prefix + UNIT + "；" + UNIT + "；"
    finding, = sentence_findings(text)
    proof = finding["verification_spans"][0]
    assert proof["text"] == UNIT + "；" + UNIT + "；"
    assert proof["start"] == len(prefix)


def test_discourse_prefix_only_localizes_existing_cross_paragraph_pair():
    source = "其中研发费用减少主要系研发物料消耗减少所致。\n研发费用减少主要系研发物料消耗减少所致。"
    report = detect_text(source, document_id="synthetic-discourse-prefix")
    issues = [e for e in report["errors"] if e["error_type"] == "冗余语句"]
    assert issues
    assert all(e["status"] == "needs_review" for e in issues)
    assert all("verification_spans" not in e for e in issues)


@pytest.mark.parametrize("prefix", ["其中", "公司", "摘要：公司", "根据报告其中"])
def test_unbounded_same_line_or_other_prefix_is_not_an_exception(prefix):
    assert not sentence_findings(prefix + UNIT + "。" + UNIT + "。")


def test_cross_paragraph_discourse_prefix_does_not_bypass_second_member_end():
    assert not sentence_findings("其中" + UNIT + "。\n" + UNIT + "，新增原因仍应保留。")


def test_literal_sentence_pair_does_not_absorb_following_new_sentence():
    pair = UNIT + "。" + UNIT + "。"
    text = pair + "下一句提供新增财务信息。"
    finding, = sentence_findings(text)
    assert finding["verification_spans"][0] == {"start": 0, "end": len(pair), "text": pair}
